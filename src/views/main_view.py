import os
import sys
import torch
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from torchvision.utils import save_image
from PIL import Image
from views.main_view_ui import Ui_MainWindow


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class MainView(QMainWindow):
    def __init__(self, args, model, main_controller):
        super().__init__()
        self.args = args
        self.export_dir = os.path.join("result", self.args.name)

        # Set app color
        self._dark_mode()

        # Combine model, view, and controller
        self._model = model
        self._main_controller = main_controller
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self, self.args)

        self._create_actions()
        self._make_shortcut()
        self._make_connections()

        self._ui.paint_scene.hide_pen_preview()

        self.blending_path = []
        # console panel output
        # sys.stdout = Stream(newText=self.onUpdateText)
        isExist = os.path.exists(self.export_dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(self.export_dir)

    def _create_actions(self):      
        # MainView actions
        self.increase_size_action = QtWidgets.QAction('Increase Size', self)
        self.addAction(self.increase_size_action)

        self.decrease_size_action = QtWidgets.QAction('Decrease Size', self)
        self.addAction(self.decrease_size_action)

    def _make_shortcut(self):

        # UI shortcuts
        self._ui.export_action.setShortcut("Ctrl+S")
        self._ui.clear_all_action.setShortcut("Ctrl+C")
        self._ui.run_action.setShortcut("Ctrl + R")
        
        # MainView shortcuts
        self.increase_size_action.setShortcut(']')
        self.decrease_size_action.setShortcut('[')

    def _make_connections(self):
        self._ui.export_action.triggered.connect(lambda: self.save_img("result", 5))
        self._ui.clear_all_action.triggered.connect(lambda: self._ui.paint_scene.clear())
        self._ui.preference_action.triggered.connect(lambda: self._ui.preference_view.show_event(self.geometry().center()))
        self._ui.run_action.triggered.connect(lambda: self.run_model())
        self._ui.run_btn.clicked.connect(lambda: self.run_model())

        self._model.ddim_changed.connect(self.ddim_update)
        self._model.image_blending_changed.connect(self.image_blending_update)
        self._model.finished.connect(self.exit_model)

        self._ui.brush_action.triggered.connect(lambda: self._ui.paint_scene.choose_brush())
        self._ui.eraser_action.triggered.connect(lambda: self._ui.paint_scene.choose_eraser())

        self._ui.paint_scene.brushChanged_connect(self._update_brush_ui)
        self._ui.palette_action.triggered.connect(self.update_pen_color)

        self.increase_size_action.triggered.connect(lambda: self._ui.paint_scene.increment_pen_size(10))
        self.decrease_size_action.triggered.connect(lambda: self._ui.paint_scene.increment_pen_size(-10))

        self._ui.size_slider.valueChanged.connect(lambda: self.set_pen_size(self._ui.size_slider.value()))
        
    def save_img(self, name, step):
        # check data is exist
        if len(self.blending_path) <= 1 :
            return

        # open buffer images
        imgs = []
        for path in self.blending_path:
            imgs.append(Image.open(path))

        # concate images
        result = self.get_concat_h(imgs[0], imgs[1])
        for i in range(2, len(imgs)):
            result = self.get_concat_h(result, imgs[i])

        # save strip image
        save_path = os.path.join(self.export_dir, name+".png")
        result.save(save_path)

        # form a circle with first image
        for i in range(self.args.out_width):
            result = self.get_concat_h(result, imgs[i%(self.args.canvas*2)])
        width = imgs[0].width * self.args.out_width
        height = imgs[0].height

        # save gif
        gif_buffer = []
        for i in range(0, result.width-width, step):
            buffer = Image.new('RGB', (width , height))
            region = result.crop((i, 0, i+width, height))
            buffer.paste(region, (0, 0))
            gif_buffer.append(buffer)

        gif_path = os.path.join(self.export_dir, name+".gif")
        gif_buffer[0].save(fp=gif_path, format='GIF', append_images=gif_buffer[1:],
            save_all=True, duration=self.args.duration, loop=0)
    def demo_gif(self):
        # check data is exist
        if len(self.blending_path) <= 1 :
            return

        # open buffer images
        imgs = []
        for path in self.blending_path:
            imgs.append(Image.open(path))

        # concate images
        result = self.get_concat_h(imgs[0], imgs[1])
        for i in range(2, len(imgs)):
            result = self.get_concat_h(result, imgs[i])
        # form a circle with first image
        for i in range(self.args.out_width):
            result = self.get_concat_h(result, imgs[i%(self.args.canvas*2)])
        width = imgs[0].width * self.args.out_width
        height = imgs[0].height

        # save gif
        gif_buffer = []
        for i in range(0, result.width-width, 20):
            buffer = Image.new('RGB', (width , height))
            region = result.crop((i, 0, i+width, height))
            buffer.paste(region, (0, 0))
            gif_buffer.append(buffer)

        gif_path = os.path.join(self.export_dir, "tmp.gif")
        gif_buffer[0].save(fp=gif_path, format='GIF', append_images=gif_buffer[1:],
            save_all=True, duration=self.args.duration*4, loop=0)

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def run_model(self):
        self.blending_path.clear()
        strokes = self._ui.paint_scene.get_img()
        for id, img in enumerate(strokes):
            save_path = os.path.join(self.export_dir, "stroke_"+str(id)+".png")
            img.save(save_path)

        self._model.set_strokes(strokes)
        self._ui.run_btn.setDisabled(True)
        self._model.start()
        self._ui.run_btn.setText("Infer ddim 1 ...")

    def exit_model(self):
        import threading
        self.view_thread = threading.Thread(target = self.demo_gif())
        self.view_thread.start()
        self.view_thread.join()
        gif = QMovie(self.export_dir+ "/tmp.gif")
        self._ui.blending_scene.labels.setMovie(gif)
        gif.start()
        self._ui.run_btn.setDisabled(False)
        self._ui.run_btn.setText("Run")

    @pyqtSlot(str, int, torch.Tensor)
    def ddim_update(self, src, id, imgs_tensor):
        if id < self.args.canvas-1:
            self._ui.run_btn.setText("Infer ddim "+str(id+2)+" ...")
        else:
            self._ui.run_btn.setText("Infer blending ...")

        save_path = os.path.join(self.export_dir, src+str(id)+".png")
        save_image(imgs_tensor, save_path)
        pim = QPixmap(save_path)
        
        self._ui.ddim_scene.labels[id].setPixmap(pim)

        
    @pyqtSlot(str, int, torch.Tensor)
    def image_blending_update(self, src, id, imgs_tensor):
        
        save_path = os.path.join(self.export_dir, src+str(id)+".png")
        save_image(imgs_tensor, save_path)
        # pim = QPixmap(save_path)
        self.blending_path.append(save_path)

        # self._ui.blending_scene.labels[id].setPixmap(pim)   

    def _update_brush_ui(self):
        self._ui.size_slider.setValue(self._ui.paint_scene.pen_size)

    def set_pen_size(self, size):
        """
        Sets pen size from slider input

        Args:
            size (int): diameter of pen
        """
        self._ui.paint_scene.set_pen_size(size)
        self._update_brush_ui()

    def set_pen_color(self, color):
        """
        sets pen color

        Args:
            color (QColor): color to set
        """
        self._ui.paint_scene.set_pen_color(color)
        self._update_brush_ui()

    def update_pen_color(self):
        color = self._ui.color_dialog.getColor(options=QtWidgets.QColorDialog.ShowAlphaChannel)
        self._ui.paint_scene.set_pen_color(color)

    def onUpdateText(self, text):
        from PyQt5.QtGui import QTextCursor
        cursor = self._ui.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self._ui.process.setTextCursor(cursor)
        self._ui.process.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return standard output to defaults.
        sys.stdout = sys.__stdout__
        self._ui.preference_view.close()
        super().closeEvent(event)

    def _dark_mode(self):
        from PyQt5.QtGui import QPalette
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt

        # using a palette to switch to dark colors:
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.black)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)
