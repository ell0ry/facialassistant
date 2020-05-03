from PyQt5.QtWidgets import QApplication

from cam_model import Camera
from views import StartWindow

# Change this on other devices based on the /dev/video#
_CAM_ID = 0

camera = Camera(_CAM_ID)
camera.initialize()

app = QApplication([])
start_window = StartWindow(camera)
start_window.show()
app.exit(app.exec_())
