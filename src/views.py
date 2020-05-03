import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QLabel, QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider, QDialog, QGridLayout
from pyqtgraph import ImageView
import cv2
import pyqtgraph as pg
from recognizer import Recognizer


class StartWindow(QDialog):
# class StartWindow(QMainWindow):
    def __init__(self, camera = None):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()

        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        # testPushButton = QPushButton("This is a test")


        # cam_layout = QVBoxLayout(self.central_widget)
        cam_layout = QVBoxLayout()
        # cam_layout.addWidget(self.win)

        # Had this uncommented.
        # cam_layout.addWidget(testPushButton)
        cam_layout.addStretch(1)
        # self.setCentralWidget(self.central_widget)

        main_layout = QGridLayout()
        # main_layout.addLayout(cam_layout, 0, 0, 1, 1)
        main_layout.addWidget(self.win, 0, 0)
        main_layout.addLayout(cam_layout, 0, 1, 1, 1)

        # main_layout.setRowStretch(1, 1)
        # main_layout.setRowStretch(2, 1)

        # Had these uncommented.
        # main_layout.setColumnStretch(0, 2)
        # main_layout.setColumnStretch(1, 1)
 
        self.setLayout(main_layout)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(10)

        self.recognizer = Recognizer()
        self.recognizer.load_models()


        self.setWindowTitle("Recognizer")

    def update(self):
        frame = self.camera.get_frame()
        encodings, landmarks = self.recognizer.find_faces(frame)
        for face in encodings.keys():
            frame = self.recognizer.draw_face(frame, face, landmarks[face])
            person = self.recognizer.recognize(encodings[face])
            print(person)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # self.img.setImage(np.flip(gray).T)
        self.img.setImage(np.rot90(gray, 3))
        # self.img.setImage(np.rot90(frame, 3))
        # self.img.setImage(np.rot90(np.flip(frame).T), 3)


class RecognitionThread(QThread):
    def __init__(self, recognizer):
        super().__init__()
        self._encodings = {}

    def run(self):
        self.camera.acquire_movie(200)

    def get_encodings(self):
        return self._encodings

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
