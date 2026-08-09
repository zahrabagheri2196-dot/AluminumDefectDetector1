from ultralytics import YOLO
import cv2
import os


class Detector:

    def init(self):

        self.baseline_model = YOLO(
            "models/yolov11_baseline.pt"
        )

        self.fuzzy_model = YOLO(
            "models/yolov11_fuzzy.pt"
        )


    def detect(self, image_path, model_type):

        if model_type == "YOLO11":

            model = self.baseline_model

        else:

            model = self.fuzzy_model


        image = cv2.imread(image_path)


        results = model(image)


        detections = []


        for result in results:


            for box in result.boxes:


                cls_id = int(box.cls[0])

                confidence = float(box.conf[0])


                x1,y1,x2,y2 = map(
                    int,
                    box.xyxy[0]
                )


                class_name = model.names[cls_id]


                # رسم باکس

                cv2.rectangle(
                    image,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    3
                )


                text = (
                    f"{class_name} "
                    f"{confidence*100:.1f}%"
                )


                cv2.putText(
                    image,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )


                item={

                    "class":class_name,

                    "confidence":confidence,

                    "box":
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]

                }


                # اگر مدل فازی باشد

                if model_type=="FUZZY":

                    item["reliability"]=round(
                        confidence*100,
                        2
                    )


                detections.append(item)



        # ذخیره تصویر خروجی

        os.makedirs(
            "results",
            exist_ok=True
        )


        cv2.imwrite(
            "results/result.jpg",
            image
        )


        return detections
