import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
import cv2
import numpy as np
import os

from Astra import Astra
from Detection import detector

class Communicate(QObject):
    calibration_complete = pyqtSignal()

class GraspingGUI(QWidget):
    def __init__(self, camera, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = camera
        self.initUI()
        self.calibration_points = []
        self.state = 'calibration'
        self.comm = Communicate()

    def initUI(self):
        """Initialize the UI components."""
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        self.pickup_btn = QPushButton('Pickup', self)
        self.stop_btn = QPushButton('Stop', self)
        self.recalibrate_btn = QPushButton('Recalibrate', self)

        self.pickup_btn.clicked.connect(self.pickup)
        self.stop_btn.clicked.connect(self.stop)
        self.recalibrate_btn.clicked.connect(self.recalibrate)

        control_layout.addWidget(self.pickup_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.recalibrate_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        self.setWindowTitle('Grasping GUI')
        self.setGeometry(100, 100, 800, 600)

    def mousePressEvent(self, event):
        """Handle mouse press events for calibration.

        Args:
            event (QMouseEvent): The mouse event.
        """
        if self.state == 'calibration':
            x = event.pos().x()
            y = event.pos().y()
            self.calibration_points.append((x, y))
            if len(self.calibration_points) == 8:
                self.comm.calibration_complete.emit()  # Emit signal

    def update_image(self):
        """Fetch and update the displayed image."""
        rgb_image, _ = self.camera.frames()
        self.display_image(rgb_image)

    def display_image(self, image):
        """Convert and display the image in the QLabel.

        Args:
            image (numpy.ndarray): The image to display.
        """
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def pickup(self):
        """Handle the pickup button click event."""
        print("Pickup button clicked")

    def stop(self):
        """Handle the stop button click event."""
        print("Stop button clicked")

    def recalibrate(self):
        """Handle the recalibrate button click event."""
        print("Recalibrate button clicked")

def main():
    app = QApplication(sys.argv)
    
    camera_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Astra')
    color_intr_file = os.path.join(camera_file_path, 'Astra_Color.intr')
    ir_intr_file = os.path.join(camera_file_path, 'Astra_IR.intr')
    
    camera = Astra(color_intr_path=color_intr_file, ir_intr_path=ir_intr_file)
    camera.start()
    
    gui = GraspingGUI(camera)	
    # gui.comm.calibration_complete.connect(lambda: handle_calibration_complete(gui.calibration_points))
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
