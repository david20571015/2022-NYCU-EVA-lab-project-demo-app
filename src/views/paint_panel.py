from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtGui, QtCore
from canvas import PaintScene, PaintView


class PaintPanel(QWidget):
    def __init__(self, count = 4, width = 256, height = 256) -> None:
        super().__init__()
        self.paint_layout = QtWidgets.QHBoxLayout()
        self.panel_cnt = count
        self.width = width
        self.height = height
        self._paint_view = []
        self.paint_scene = []
        
        for i in range(self.panel_cnt):
            self._paint_view.append(PaintView(self.width, self.height))
            self._paint_view[i].setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            self.paint_scene.append(PaintScene(0, 0, self.width, self.height, None))
            self._paint_view[i].setScene(self.paint_scene[i])
            self.paint_layout.addWidget(self._paint_view[i])

        self.last_color = self.paint_scene[0].pen_color

        self.setLayout(self.paint_layout)

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

    # REFERENCE
    def save_img(self):
        """
        saves image to file
        """
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Canvas", "Render",
                    "Images (*.png *.jpg)")

        if file_path == "":
            return
        else:
            imgs = self.get_img()
            idx = 1
            for img in imgs:
                img.save(file_path + str(idx) + ".jpg")
                idx += 1
    
    # REFERENCE
    def get_img(self):
        """
        gets all images from PaintPanel

        Returns:
            img: returns a list of QImage data from canvas
        """
        imgs = []
        for paint_scene in self.paint_scene:
            img = QtGui.QImage(self.width, self.height, QtGui.QImage.Format_RGB32)
            paint = QtGui.QPainter(img)
            paint.setRenderHint(QtGui.QPainter.Antialiasing)
            paint_scene.render(paint)
            paint.end()
            imgs.append(img)
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
