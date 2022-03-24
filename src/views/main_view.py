from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from views.main_view_ui import Ui_MainWindow
from layers import LayerPanel
from delegate import TreeDelegate

import sys

class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

class MainView(QMainWindow):
    def __init__(self, args, model, main_controller):
        super().__init__()

        # Set app color
        self._dark_mode()

        # Combine model, view, and controller
        self._model = model
        self._main_controller = main_controller
        self.layers_tree = LayerPanel(dragToggleColumns=[0], columns=['', ''])
        self.layers_tree.setItemDelegate(TreeDelegate())
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self, args)

        self._create_actions()
        self._make_shortcut()
        self._make_connections()

        self._ui.paint_scene.hide_pen_preview()

        # console panel output
        # sys.stdout = Stream(newText=self.onUpdateText)

    def _create_actions(self):
        
        # MainView actions
        self.undo_action = QtWidgets.QAction('Undo', self)
        self.addAction(self.undo_action)

        self.redo_action = QtWidgets.QAction('Redo', self)
        self.addAction(self.redo_action)

        self.delete_action = QtWidgets.QAction('Delete', self)
        self.addAction(self.delete_action)

        self.group_action = QtWidgets.QAction('Group', self)
        self.addAction(self.group_action)

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
        # self.undo_action.setShortcut('Ctrl+Z')
        # self.redo_action.setShortcut('Shift+Ctrl+Z')
        # self.delete_action.setShortcut('Backspace')
        # self.group_action.setShortcut('Ctrl+G')
        self.increase_size_action.setShortcut(']')
        self.decrease_size_action.setShortcut('[')

    def _make_connections(self):
        self._ui.export_action.triggered.connect(lambda: self._ui.paint_scene.save_img())
        self._ui.clear_all_action.triggered.connect(lambda: self._ui.paint_scene.clear())
        self._ui.run_action.triggered.connect(lambda: self._model.run(self._ui.paint_scene.get_img()))
        self._ui.run_btn.clicked.connect(lambda: self._model.run(self._ui.paint_scene.get_img()))

        self._model.ddim_update.connect(self.ddim_changed)
        self._model.image_blending_update.connect(self.image_blending_changed)

        self._ui.brush_action.triggered.connect(lambda: self._ui.paint_scene.choose_brush())
        # self._ui.line_action.triggered.connect(lambda: self._ui.paint_scene.choose_line())
        self._ui.eraser_action.triggered.connect(lambda: self._ui.paint_scene.choose_eraser())

        # self._ui.paint_scene.strokeAdded_connect(self.create_layer_item)
        # self._ui.paint_scene.strokeRemoved_connect(self.remove_layer_item)
        self._ui.paint_scene.brushChanged_connect(self._update_brush_ui)
        self._ui.palette_action.triggered.connect(self.update_pen_color)

        self.increase_size_action.triggered.connect(lambda: self._ui.paint_scene.increment_pen_size(10))
        self.decrease_size_action.triggered.connect(lambda: self._ui.paint_scene.increment_pen_size(-10))

        self._ui.size_slider.valueChanged.connect(lambda: self.set_pen_size(self._ui.size_slider.value()))
        
        # self.delete_action.triggered.connect(self.delete_layer)
        # self.group_action.triggered.connect(self.group_layers)

        # self.layers_tree.itemChanged.connect(self.layer_change)
        # self.layers_tree.layerOrderChanged.connect(self.update_layer_index)

    # TODO:
    @pyqtSlot(list)
    def ddim_changed(self, imgs):
        pass

    # TODO:
    @pyqtSlot(list)
    def image_blending_changed(self, imgs):
        pass

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
        # color = self._ui.color_dialog.getColor(self._ui.paint_scene.pen_color,
        #                                    self, "Color",
        #                                    QtWidgets.QColorDialog.ShowAlphaChannel)
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

    # def create_layer_item(self, stroke_id, layer_name):
    #     """
    #     Creates layer item in layer panel using stroke data

    #     Args:
    #         stroke_id (int): unique index of stroke
    #         layer_name (str): name of stroke layer

    #     """
    #     stroke_info = ['', layer_name]
    #     layer = Layer(stroke_info, stroke_index=stroke_id)

    #     highest_group = None
    #     if self.layers_tree.selectedItems():
    #         iterator = QtWidgets.QTreeWidgetItemIterator(self.layers_tree)
    #         while iterator.value():
    #             item = iterator.value()
    #             if isinstance(item, Folder) and item in self.layers_tree.selectedItems():
    #                 highest_group = item
    #                 break
    #             iterator += 1
    #     if highest_group:
    #         highest_group.insertChild(0, layer)
    #     else:
    #         self.layers_tree.insertTopLevelItem(0, layer)
    #     self.update_layer_index()

    # def remove_layer_item(self, stroke_id):
    #     """
    #     deletes layer item in layer panel

    #     Args:
    #         stroke_id (int): unique index of stroke to be removed

    #     """
    #     iterator = QtWidgets.QTreeWidgetItemIterator(self.layers_tree)

    #     while iterator.value():
    #         item = iterator.value()
    #         if isinstance(item, Layer):
    #             layer_data = item.data(1, QtCore.Qt.UserRole)[0]
    #             if layer_data['stroke_index'] == stroke_id:
    #                 parent = item.parent()
    #                 if parent:
    #                     idx = parent.indexOfChild(item)
    #                     parent.takeChild(idx)
    #                 else:
    #                     idx = self.layers_tree.indexOfTopLevelItem(item)
    #                     self.layers_tree.takeTopLevelItem(idx)
    #         if isinstance(item, Folder):
    #             layer_data = item.data(1, QtCore.Qt.UserRole)[0]

    #             if item.group_index == stroke_id:
    #                 parent = item.parent()
    #                 if parent:
    #                     idx = parent.indexOfChild(item)
    #                     parent.takeChild(idx)
    #                 else:
    #                     idx = self.layers_tree.indexOfTopLevelItem(item)
    #                     self.layers_tree.takeTopLevelItem(idx)
    #         iterator += 1

    # def layer_change(self, item, column):
    #     """
    #     updates stroke information, used when updating visibility or layer name

    #     Args:
    #         item (QTreeWidgetItem): item associated with stroke
    #         column (int): column to change
    #     """
    #     if column == 0:
    #         if isinstance(item, Layer):
    #             self._ui.paint_scene.toggle_layer_visibility(item.stroke_index,
    #                                                      item.visible)

    #         elif isinstance(item, Folder):
    #             for i in range(item.childCount()):
    #                 if item.visible is True:
    #                     item.child(i).setFlags(QtCore.Qt.ItemIsSelectable |
    #                                            QtCore.Qt.ItemIsEditable |
    #                                            QtCore.Qt.ItemIsEnabled |
    #                                            QtCore.Qt.ItemIsDragEnabled)
    #                 else:
    #                     item.child(i).setFlags(QtCore.Qt.NoItemFlags)
    #                 self._ui.paint_scene.toggle_layer_visibility(item.child(i).stroke_index, item.visible)

    #     elif column == 1:
    #         if isinstance(item, Layer):
    #             self._ui.paint_scene.update_layer_name(item.stroke_index,
    #                                                item.text(1))

    # def delete_layer(self):
    #     """
    #     Deletes selected layers
    #     """
    #     for item in self.layers_tree.selectedItems():
    #         # remove item.stroke_index
    #         if isinstance(item, Layer):
    #             if item.parent():
    #                 command = DeleteStroke(self, item, group=item.parent())
    #                 self._ui.paint_scene.undo_stack.push(command)
    #             else:
    #                 command = DeleteStroke(self, item)
    #                 self._ui.paint_scene.undo_stack.push(command)

    #         if isinstance(item, Folder):
    #             command = DeleteGroup(self, item)
    #             self._ui.paint_scene.undo_stack.push(command)

    # def group_layers(self):
    #     """
    #     groups seleted layers

    #     """
    #     if self.layers_tree.selectedItems():
    #         grab_items = []
    #         for item in self.layers_tree.selectedItems():
    #             if isinstance(item, Layer):
    #                 grab_items.append(item.stroke_index)

    #         command = GroupStrokes(self, grab_items)
    #         self._ui.paint_scene.undo_stack.push(command)

    # def update_layer_index(self):
    #     """
    #     iterates through layer panel & updates stacking order of strokes

    #     """
    #     iterator = QtWidgets.QTreeWidgetItemIterator(self.layers_tree)
    #     while iterator.value():
    #         item = iterator.value()
    #         target_index = self.layers_tree.indexFromItem(item).row()
    #         try:
    #             new_indx = len(self._ui.paint_scene.strokes) - target_index
    #             self._ui.paint_scene.set_stroke_zindex(item._stroke_index, new_indx)
    #         except AttributeError:
    #             pass

    #         if isinstance(item, Layer):
    #             # layer_data = item.data(1, QtCore.Qt.UserRole)[0]
    #             layer_data = item.data(1, QtCore.Qt.UserRole)[0]
    #             parent = item.parent()
    #             if not parent:
    #                 layer_data['layerType'] = 0
    #             else:
    #                 layer_data['layerType'] = 2

    #             varient = QtCore.QVariant((layer_data,))
    #             item.setData(1, QtCore.Qt.UserRole, varient)

    #         elif isinstance(item, Folder):
    #             for i in range(item.childCount()):
    #                 if item.visible is True:
    #                     item.child(i).setFlags(QtCore.Qt.ItemIsSelectable |
    #                                            QtCore.Qt.ItemIsEditable |
    #                                            QtCore.Qt.ItemIsEnabled |
    #                                            QtCore.Qt.ItemIsDragEnabled)
    #                 else:
    #                     item.child(i).setFlags(QtCore.Qt.NoItemFlags)
    #                 self._ui.paint_scene.toggle_layer_visibility(item.child(i).stroke_index, item.visible)
    #         iterator += 1