"""
Trainable Fuzzy Feature Gate
-----------------------------
نسخه‌ی یادگیرنده و سطح-ویژگی (Feature-level) ماژول فازی که قبلاً روی
Confidence خروجی نهایی اعمال می‌شد.

تناظر مفهومی با نسخه‌ی post-hoc:

    Post-hoc (fuzzy_confidence_module.py)     ->  این نسخه (Trainable, Feature-level)
    ---------------------------------------------------------------------------
    Confidence خروجی Head                     ->  قدرت فعال‌سازی نرمال‌شده‌ی هر موقعیت
                                                    (mean فعال‌سازی روی کانال‌ها)
    Relative Size (از باکس واقعی)              ->  مقیاس هرمی که ماژول رویش نصب شده
                                                    (small=P3, medium=P4, large=P5)
    Class Difficulty (جدول ثابت 1-AP)          ->  بردار قابل‌یادگیری در سطح کانال،
                                                    مقداردهی اولیه با میانگین همان جدول

    قوانین فازی، توابع عضویت مثلثی، و ساختار AND=min حفظ شده‌اند؛ فقط اکنون
    کل زنجیره differentiable است و به‌صورت element-wise gate روی نقشه‌ویژگی
    اعمال می‌شود (مشابه فلسفه‌ی SE-Net/CBAM، اما با قوانین تفسیرپذیر).

نحوه‌ی استفاده صحیح (حیاتی):
    این ماژول باید به‌عنوان attribute واقعی مدل ثبت شود (نه فقط forward_hook)
    تا پارامترهایش در model.parameters() ظاهر شوند و توسط optimizer آپدیت شوند.
    نمونه‌ی کامل پایین فایل آمده است.
"""

import torch
import torch.nn as nn


class TrainableFuzzyFeatureGate(nn.Module):
    def __init__(self, num_channels: int, size_level: str, init_difficulty: float = 0.5):
        super().__init__()
        assert size_level in ("small", "medium", "large")
        self.size_level = size_level
        self.size_boost = {"small": 1.0, "medium": 0.5, "large": 0.0}[size_level]

        # بردار سختی قابل‌یادگیری در سطح کانال (پروکسی class_difficulty)
        # نکته‌ی حیاتی: مقداردهی اولیه نباید دقیقاً یکسان برای همه‌ی کانال‌ها و
        # دقیقاً روی نقطه‌ی متقارن (مثل ۰.۵) باشد، چون آنجا مشتق تابع عضویت
        # میانی صفر است (قله‌ی متقارن) و گرادیان اولیه صفر می‌شود. برای شکستن
        # تقارن، کمی نویز تصادفی کوچک به مقدار اولیه اضافه می‌شود.
        init_logit_center = torch.logit(torch.tensor(init_difficulty).clamp(0.01, 0.99))
        jitter = torch.randn(num_channels) * 0.15
        self.channel_difficulty_logit = nn.Parameter(init_logit_center + jitter)

        # مقادیر خروجی برچسب‌های فازی: [low, medium, high, very_high] - Sugeno-style قابل یادگیری
        self.output_values = nn.Parameter(torch.tensor([0.2, 0.5, 0.75, 0.95]))

        # بازه‌ی نهایی gate. نسخه‌ی قبلی (0.95-1.05) روی backbone منجمد تقریباً
        # خنثی عمل کرد (gate ≈ 1.0 همیشه) - این نسخه کمی بازه را باز می‌کند
        # تا فازی فرصت اثرگذاری واقعی‌تری داشته باشد (بدون ریسک قبلی، چون
        # backbone دیگر در حال یادگیری نیست).
        self.gate_min = 0.9
        self.gate_range = 0.2

    @staticmethod
    def _soft_membership(x, centers, temperature=0.08):
        """
        نسخه‌ی نرم (differentiable در همه‌جا) توابع عضویت، بر پایه‌ی
        فاصله‌ی گاوسی + Softmax به‌جای مثلث سخت + clamp.
        مزیت: هرگز گرادیان صفر نمی‌شود (بر خلاف نسخه‌ی مثلثی/clamp که
        در نقاط قله یا نواحی صفرشده گرادیان را از دست می‌داد).
        جمع سه خروجی همیشه دقیقاً ۱ است (مثل یک توزیع احتمال روی سه برچسب).
        centers: تانسور/لیست با ۳ مقدار مرکز مجموعه‌های فازی (مثلاً [0, 0.5, 1.0])
        """
        # x: هر شکلی، centers را broadcast می‌کنیم
        dists = [-(x - c) ** 2 / temperature for c in centers]
        stacked = torch.stack(dists, dim=0)  # (3, ...)
        weights = torch.softmax(stacked, dim=0)
        return weights[0], weights[1], weights[2]

    def _confidence_membership(self, act):
        return self._soft_membership(act, centers=[0.0, 0.5, 1.0])

    def _difficulty_membership(self, diff):
        return self._soft_membership(diff, centers=[0.0, 0.5, 1.0])

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # --- پروکسی Confidence: قدرت فعال‌سازی نرمال‌شده هر موقعیت مکانی ---
        act = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        act_min = act.amin(dim=(2, 3), keepdim=True)
        act_max = act.amax(dim=(2, 3), keepdim=True)
        act = (act - act_min) / (act_max - act_min + 1e-6)

        conf_low, conf_med, conf_high = self._confidence_membership(act)  # (B,1,H,W)

        # --- پروکسی Difficulty: بردار یادگیرنده سطح کانال ---
        diff = torch.sigmoid(self.channel_difficulty_logit).view(1, C, 1, 1)  # (1,C,1,1)
        d_easy, d_med, d_hard = self._difficulty_membership(diff)

        sb = self.size_boost
        ov = self.output_values  # [low, medium, high, very_high]

        # --- همان ۱۱ قانون اصلی نسخه‌ی نهایی post-hoc، به‌صورت برداری ---
        rules = [
            (conf_high * d_easy, 3),
            (conf_high * d_med, 2),
            (conf_high * d_hard, 3),
            (conf_med * d_easy, 1),
            (conf_med * d_med, 1),
            (conf_med * d_hard, 1),
            (conf_low * d_easy, 0),
            (conf_low * d_med, 0),
            (conf_low * d_hard, 0),
            (conf_low * d_hard * sb, 1),   # تقویت: کوچک + سخت + کانفیدنس پایین
            (conf_med * d_hard * sb, 2),   # تقویت: کوچک + سخت + کانفیدنس متوسط
        ]

        weighted_sum = sum(strength * ov[idx] for strength, idx in rules)
        weight_total = sum(strength for strength, _ in rules) + 1e-6
        fuzzy_out = weighted_sum / weight_total  # (B,C,H,W) به‌خاطر broadcast دو طرف

        gate = self.gate_min + self.gate_range * fuzzy_out
        return x * gate


class FuzzyFeatureGateManager(nn.Module):
    """
    نگه‌دارنده‌ی سه ماژول فازی برای سه مقیاس P3/P4/P5 و نصب آن‌ها روی مدل
    از طریق forward_hook - اما چون این کلاس خودش nn.Module است و ماژول‌ها
    را به‌عنوان submodule واقعی (self.gates) نگه می‌دارد، اگر این کلاس را
    خودتان attribute مدل کنید (model.model.fuzzy_manager = manager)،
    پارامترها به‌طور خودکار در model.parameters() ظاهر می‌شوند.
    """

    def __init__(self, channel_sizes: dict, init_difficulty: dict = None):
        """
        channel_sizes: {"small": C_p3, "medium": C_p4, "large": C_p5}
        init_difficulty: مقدار میانگین سختی برای هر مقیاس (اختیاری)
        """
        super().__init__()
        init_difficulty = init_difficulty or {"small": 0.5, "medium": 0.5, "large": 0.5}
        self.gates = nn.ModuleDict({
            level: TrainableFuzzyFeatureGate(channel_sizes[level], level, init_difficulty[level])
            for level in ("small", "medium", "large")
        })
        self._hooks = []

    def install(self, model, layer_names: dict):
        """
        layer_names: {"small": module, "medium": module, "large": module}
        (خود ماژول‌های نن، نه اسم رشته‌ای - همان الگویی که قبلاً برای
        FuzzyHead/FuzzySE استفاده شده بود)
        """
        for level, layer_module in layer_names.items():
            gate = self.gates[level]

            def make_hook(g):
                def hook(module, inp, output):
                    return g(output)
                return hook

            h = layer_module.register_forward_hook(make_hook(gate))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


if __name__ == "__main__":
    # ------------------------------------------------------------
    # تست سریع: بررسی صحت shape خروجی و این‌که backward() بدون خطا کار می‌کند
    # (یعنی گرادیان واقعاً تا پارامترهای فازی می‌رسد)
    # ------------------------------------------------------------
    torch.manual_seed(0)

    channel_sizes = {"small": 64, "medium": 128, "large": 256}
    manager = FuzzyFeatureGateManager(channel_sizes)

    dummy_inputs = {
        "small": torch.randn(2, 64, 80, 80, requires_grad=True),
        "medium": torch.randn(2, 128, 40, 40, requires_grad=True),
        "large": torch.randn(2, 256, 20, 20, requires_grad=True),
    }

    total_loss = 0.0
    for level, x in dummy_inputs.items():
        gate_module = manager.gates[level]
        out = gate_module(x)
        assert out.shape == x.shape, f"Shape mismatch for {level}: {out.shape} vs {x.shape}"
        total_loss = total_loss + out.mean()

    total_loss.backward()

    print("✅ Forward pass موفق بود، shapeها تطابق دارند.")
    for level in channel_sizes:
        grad_norm = manager.gates[level].channel_difficulty_logit.grad.norm().item()
        print(f"  {level}: نُرم گرادیان channel_difficulty = {grad_norm:.6f} "
              f"({'✅ گرادیان می‌رسد' if grad_norm > 0 else '❌ گرادیان صفر است!'})")

    # بررسی این‌که پارامترها واقعاً در .parameters() هستند (شبیه‌سازی چک قبل از train)
    n_params = sum(p.numel() for p in manager.parameters())
    print(f"\nتعداد کل پارامترهای قابل‌یادگیری در manager: {n_params}")
