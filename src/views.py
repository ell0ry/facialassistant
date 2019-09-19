import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider
from pyqtgraph import ImageView
from recognizer import Recognizer


class StartWindow(QMainWindow):
    def __init__(self, camera = None):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()
        # self.button_frame = QPushButton('Acquire Frame', self.central_widget)
        # self.button_movie = QPushButton('Start Movie', self.central_widget)
        self.image_view = ImageView()
        # self.slider = QSlider(Qt.Horizontal)
        # self.slider.setRange(0,10)

        self.layout = QVBoxLayout(self.central_widget)
        # self.layout.addWidget(self.button_frame)
        # self.layout.addWidget(self.button_movie)
        self.layout.addWidget(self.image_view)
        # self.layout.addWidget(self.slider)
        self.setCentralWidget(self.central_widget)

        # self.button_frame.clicked.connect(self.update_image)
        # self.button_movie.clicked.connect(self.start_movie)
        # self.slider.valueChanged.connect(self.update_brightness)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)

        self.recognizer = Recognizer()
        self.recognizer.load_models()

    def update(self):
        frame = self.camera.get_frame()
        found = self.recognizer.find_faces(frame)
        for face in found.keys():
            frame = self.recognizer.draw_face(frame, face)
            person = self.recognizer.recognize(found[face])
            print(person)
        self.image_view.setImage(frame.T)

    def update_brightness(self, value):
        value /= 10
        self.camera.set_brightness(value)

    # def start_movie(self):
        # self.movie_thread = MovieThread(self.camera)
        # self.movie_thread.start()


class RecognitionThread(QThread):
    def __init__(self, recognizer):
        super().__init__()

    def run(self):
        self.camera.acquire_movie(200)

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())
