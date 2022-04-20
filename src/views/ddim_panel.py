# -*- coding: utf-8 -*-
import os
import io
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageQt

class DDIMPanel(QWidget):
    def __init__(self, count = 4, width = 256, height = 256) -> None:
        super().__init__()
        self.panel_layout = QtWidgets.QHBoxLayout()
        self.labels = []

        for i in range(count):
            self.labels.append(QtWidgets.QLabel())
            self.labels[i].setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage()))
            self.labels[i].setFixedSize(width, height)
            self.panel_layout.addWidget(self.labels[i])

        self.setLayout(self.panel_layout)

    def save_img(self, dir):
        """
        saves image to file
        """
        from PyQt5.QtWidgets import QFileDialog
        isExist = os.path.exists(dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(dir)

        file_path, _ = QFileDialog.getSaveFileName(self, "Save ddim", dir+"/DDIM",
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
        gets all images from DDIMPanel

        Returns:
            img: returns a list of QImage data from canvas
        """
        imgs = []
        for label in self.labels:
            pix_img = label.pixmap()
            img = ImageQt.fromqpixmap(pix_img)

            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QBuffer.ReadWrite)
            img.save(buffer, "png")

            pil_im = Image.open(io.BytesIO(buffer.data()))
            imgs.append(pil_im)

        return imgs
    
    