# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from views.ddim_panel import DDIMPanel
from views.blending_panel import BlendingPanel
from views.paint_panel import PaintPanel
from views.preference_view import PreferenceView

from resources import icons

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, args):

        # Get monitor resolution
        self.screen_size = QtWidgets.QDesktopWidget().screenGeometry(-1)
        MainWindow.setGeometry((int)(self.screen_size.width()*.2), (int)(self.screen_size.height()*.1), 
            (int)(self.screen_size.width()*.6), (int)(self.screen_size.height()*.8))

        # Workspace layout
        self.main_splitter = QtWidgets.QSplitter(Qt.Vertical)
        MainWindow.setCentralWidget(self.main_splitter)

        # section -- paint panel 
        self.paint_label = QtWidgets.QLabel("Paint Panel")
        self.paint_label.setAlignment(Qt.AlignCenter)
        self.paint_label.setFont(QtGui.QFont("AnyStyle", 24))
        # paint scene
        self.paint_scene = PaintPanel(args.canvas, args.width, args.height)
        # paint panel run button
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setFont(QtGui.QFont("AnyStyle", 18))
        # paint panel 
        self.paint_panel = QtWidgets.QWidget()
        # paint panel layout
        self.paint_layout = QtWidgets.QVBoxLayout()
        self.paint_panel.setLayout(self.paint_layout)
        self.paint_layout.addWidget(self.paint_label)
        self.paint_layout.addWidget(self.paint_scene)
        self.paint_layout.addWidget(self.run_btn)

        # ddim panel, or you can create your QWidget
        # section -- ddim panel 
        self.ddim_label = QtWidgets.QLabel("Diffusion Results")
        self.ddim_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.ddim_label.setFont(QtGui.QFont("AnyStyle", 24))
        # ddim scene
        self.ddim_scene = DDIMPanel(args.canvas, args.width, args.height)
        # ddim panel 
        self.ddim_panel = QtWidgets.QWidget()
        # ddim panel layout
        self.ddim_layout = QtWidgets.QVBoxLayout()
        self.ddim_panel.setLayout(self.ddim_layout)
        self.ddim_layout.addWidget(self.ddim_label)
        self.ddim_layout.addWidget(self.ddim_scene)

        # image_blending panel, or you can create your QWidget
        # section -- blending panel 
        self.blending_label = QtWidgets.QLabel("Blending Results")
        self.blending_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.blending_label.setFont(QtGui.QFont("AnyStyle", 24))
        # blending scene
        self.blending_scene = BlendingPanel()
        # blending panel 
        self.blending_panel = QtWidgets.QWidget()
        # blending panel layout
        self.blending_layout = QtWidgets.QVBoxLayout()
        self.blending_panel.setLayout(self.blending_layout)
        self.blending_layout.addWidget(self.blending_label)
        self.blending_layout.addWidget(self.blending_scene)

        # console panel
        # self.process = QtWidgets.QTextEdit()
        # self.process.setMaximumHeight(self.screen_size.height() - (int)(args.height*1.3))
        # self.process.moveCursor(QtGui.QTextCursor.Start)
        # self.process.ensureCursorVisible()
        # self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)

        # add ddim panel and image_blending panel(sample code)
        self.main_splitter.addWidget(self.paint_panel)
        self.main_splitter.addWidget(self.ddim_panel)
        self.main_splitter.addWidget(self.blending_panel)
        # self.main_splitter.addWidget(self.process)

        # creating menu bar
        self.main_menu = QtWidgets.QMenuBar()

        # Adding menu options
        self.file_menu = self.main_menu.addMenu("File")
        self.edit_menu = self.main_menu.addMenu("Edit")
        self.run_menu = self.main_menu.addMenu("Run")
        MainWindow.setMenuBar(self.main_menu)

        # creating tool bar
        self.edit_tool_bar = QtWidgets.QToolBar()
        self.edit_tool_bar.setIconSize(QtCore.QSize(96, 96))
        MainWindow.addToolBar(Qt.LeftToolBarArea, self.edit_tool_bar)
        
        # creating export option
        self.export_action = QtWidgets.QAction("&Export")
        self.file_menu.addAction(self.export_action)

        # creating clear option
        self.clear_all_action = QtWidgets.QAction("&Clear all")
        self.file_menu.addAction(self.clear_all_action)

        # creating preference option
        self.preference_action = QtWidgets.QAction("Preference")
        self.edit_menu.addAction(self.preference_action)
        self.preference_view = PreferenceView(args)

        # creating run option
        self.run_action = QtWidgets.QAction("&Run")
        self.run_menu.addAction(self.run_action)

        # creating edit option -- brush
        self.brush_action = QtWidgets.QAction(QtGui.QIcon(":brush_white.png"), "&Brush")
        self.edit_tool_bar.addAction(self.brush_action)

        # creating edit option -- palette
        self.palette_action = QtWidgets.QAction(QtGui.QIcon(":palette_white.png"), "&Palette")
        self.edit_tool_bar.addAction(self.palette_action)

        # creating edit option -- eraser
        self.eraser_action = QtWidgets.QAction(QtGui.QIcon(":eraser_white.png"), "&Eraser")
        self.edit_tool_bar.addAction(self.eraser_action)

        # creating edit option -- brush size
        self.increment_action = QtWidgets.QAction(QtGui.QIcon(":increment_white.png"), "&Increment")
        self.edit_tool_bar.addAction(self.increment_action)

        self.size_slider = QtWidgets.QSlider(Qt.Vertical)
        self.size_slider.setRange(20, 250)
        self.size_slider.setValue(30)
        self.size_slider.setTickInterval(10)
        self.size_slider.setSingleStep(10)
        self.size_slider.setMaximumHeight((int)(self.screen_size.height()*.1))
        self.edit_tool_bar.addWidget(self.size_slider)

        self.decrement_action = QtWidgets.QAction(QtGui.QIcon(":decrement_white.png"), "&Decrement")
        self.edit_tool_bar.addAction(self.decrement_action)

        self.color_dialog = QtWidgets.QColorDialog()

