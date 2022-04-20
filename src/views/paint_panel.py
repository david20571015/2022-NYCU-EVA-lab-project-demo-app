import io
import os
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from canvas import PaintScene, PaintView

class PaintPanel(QWidget):
    def __init__(self, count = 4, width = 256, height = 256) -> None:
        super().__init__()
        self.paint_layout = QtWidgets.QGridLayout()
        self.panel_cnt = count
        self.width = width
        self.height = height
        self._paint_view = []
        self.paint_scene = []
        self.upload_btn = []
        for i in range(self.panel_cnt):
            self._paint_view.append(PaintView(self.width, self.height))
            self._paint_view[i].setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            self.paint_scene.append(PaintScene(0, 0, self.width, self.height, None))
            self._paint_view[i].setScene(self.paint_scene[i])
            self.paint_layout.addWidget(self._paint_view[i], 0, i)

            self.upload_btn.append(QtWidgets.QPushButton("Upload "+str(i+1)))
            self.paint_layout.addWidget(self.upload_btn[i], 1, i)

        self.last_color = self.paint_scene[0].pen_color

        self.setLayout(self.paint_layout)
        self._make_connections()

    def _make_connections(self):
        from functools import partial
        for id in range(len(self.upload_btn)):
            self.upload_btn[id].clicked.connect(partial(self.upload_image, id))

    def upload_image(self, id):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "choose image", ".", "Images (*.png *.jpg)")

        if file_name == "":
            print("cancelled uploading")
            return

        pim = QPixmap(file_name)
        pim = pim.scaled(self.width, self.height, Qt.IgnoreAspectRatio)
        self.paint_scene[id].addPixmap(pim)

    @property
    def pen_size(self):
        return self.paint_scene[0].pen_size
    def hide_pen_preview(self):
        for paint_scene in self.paint_scene:
            paint_scene.hide_pen_preview()
    def strokeAdded_connect(self, func):
        for paint_scene in self.paint_scene:
            paint_scene.strokeAdded.connect(func)
    def strokeRemoved_connect(self, func):
        for paint_scene in self.paint_scene:
            paint_scene.strokeRemoved.connect(func)
    def brushChanged_connect(self, func):
        for paint_scene in self.paint_scene:
            paint_scene.brushChanged.connect(func)
    def increment_pen_size(self, size = 10):
        for paint_scene in self.paint_scene:
            paint_scene.increment_pen_size(size)
    def set_pen_color(self, color):
        for paint_scene in self.paint_scene:
            paint_scene.set_pen_color(color)
    def set_pen_size(self, size):
        for paint_scene in self.paint_scene:
            paint_scene.set_pen_size(size)
    def choose_eraser(self):
        self.last_color = self.paint_scene[0].pen_color
        self.set_pen_color(QtGui.QColor(255, 255, 255, 255))

    def save_img(self, dir):
        """
        saves image to file
        """
        from PyQt5.QtWidgets import QFileDialog
        isExist = os.path.exists(dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(dir)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Canvas", dir+"/Stroke",
                    "Images (*.png *.jpg)")

        if file_path == "":
            return
        else:
            imgs = self.get_img()
            idx = 1
            for img in imgs:
                img.save(file_path + str(idx) + ".png")
                idx += 1

    def get_img(self):
        """
        gets all images from PaintPanel

        Returns:
            img: returns a list of Image data from canvas
        """
        imgs = []
        for paint_scene in self.paint_scene:
            img = QtGui.QImage(self.width, self.height, QtGui.QImage.Format_RGB32)
            paint = QtGui.QPainter(img)
            paint.setRenderHint(QtGui.QPainter.Antialiasing)
            paint_scene.render(paint)
            paint.end()

            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QBuffer.ReadWrite)
            img.save(buffer, "png")
            pil_im = Image.open(io.BytesIO(buffer.data()))
            imgs.append(pil_im)

        return imgs

    def clear(self):
        border = QtGui.QPen()
        border.setWidthF(0.01)
        border.setColor(QtGui.QColor(255, 255, 255, 255))
        for paint_scene in self.paint_scene:
            paint_scene.addRect(0, 0, self.width, self.height, border, QtGui.QBrush(
                QtGui.QColor(255, 255, 255, 255), QtCore.Qt.SolidPattern))

    def choose_brush(self):
        self.set_pen_color(self.last_color)
