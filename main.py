import sys

from PyQt6.QtWidgets import *

from PyQt6.QtGui import *

from PyQt6.QtCore import *

from detector import Detector



class MainWindow(QMainWindow):


    def init(self):

        super().__init__()


        self.setWindowTitle(
            "Aluminum Profile Defect Detection"
        )


        self.resize(
            1200,
            800
        )


        self.detector = Detector()


        self.image_path=None



        # تصویر

        self.image_label = QLabel()

        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )



        # دکمه ها

        self.load_button = QPushButton(
            "Load Image"
        )


        self.run_button = QPushButton(
            "Run Detection"
        )


        self.save_button = QPushButton(
            "Save Result"
        )



        # انتخاب مدل

        self.yolo_radio = QRadioButton(
            "YOLO11 Baseline"
        )


        self.fuzzy_radio = QRadioButton(
            "YOLO11 + Fuzzy"
        )


        self.yolo_radio.setChecked(True)



        # خروجی متن

        self.output = QTextEdit()

        self.output.setReadOnly(True)



        layout = QVBoxLayout()



        layout.addWidget(
            self.load_button
        )


        layout.addWidget(
            self.image_label
        )


        layout.addWidget(
            self.yolo_radio
        )


        layout.addWidget(
            self.fuzzy_radio
        )


        layout.addWidget(
            self.run_button
        )


        layout.addWidget(
            self.output
        )



        widget = QWidget()

        widget.setLayout(layout)


        self.setCentralWidget(
            widget
        )



        self.load_button.clicked.connect(
            self.load_image
        )


        self.run_button.clicked.connect(
            self.run_detection
        )



    def load_image(self):


        file,_ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.png *.bmp)"
        )


        if file:


            self.image_path=file


            pixmap=QPixmap(file)


            self.image_label.setPixmap(
                pixmap.scaled(
                    800,
                    500
                )
            )



    def run_detection(self):


        if self.image_path is None:

            return



        if self.yolo_radio.isChecked():

            model="YOLO11"

        else:

            model="FUZZY"



        results=self.detector.detect(
            self.image_path,
            model
        )



        # نمایش تصویر نتیجه

        pixmap=QPixmap(
            "results/result.jpg"
        )


        self.image_label.setPixmap(
            pixmap.scaled(
                800,
                500
            )
        )



        text=""


        for r in results:


            text += f"""

Defect:
{r['class']}


Confidence:
{r['confidence']*100:.2f}%


Bounding Box:
{r['box']}

"""


            if "reliability" in r:


                text += f"""

Reliability:
{r['reliability']}%

"""


            text+="------------------"



        self.output.setText(text)



app = QApplication(sys.argv)


window = MainWindow()

window.show()


sys.exit(
    app.exec()
)
