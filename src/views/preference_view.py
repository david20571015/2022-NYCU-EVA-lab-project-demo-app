from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtCore

class PreferenceView(QWidget):
    def __init__(self, args):
        super(PreferenceView, self).__init__()
        self.setWindowTitle("Preference")
        
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        # Inference options
        self.inference_label = QtWidgets.QLabel("Inference")
        self.inference_label.setAlignment(QtCore.Qt.AlignTop|QtCore.Qt.AlignHCenter)
        self.main_layout.addWidget(self.inference_label)

        self.infer_all = QtWidgets.QCheckBox('infer all combinations of input', self)
        if args.i == "true":
            self.infer_all.toggle() 
        self.main_layout.addWidget(self.infer_all)

        self.gpu_label = QtWidgets.QLabel("GPU")
        self.gpu_label.setAlignment(QtCore.Qt.AlignTop|QtCore.Qt.AlignHCenter)
        self.main_layout.addWidget(self.gpu_label)
        # REFERENCE: more gpu options
        # GPU options

        # save button
        self.save_btn = QtWidgets.QPushButton("Save")
        self.main_layout.addWidget(self.save_btn)

        self.setLayout(self.main_layout)
        self.setFixedSize(self.sizeHint())
        self._dark_mode()

        self._make_connections(args)
        
    def show_event(self, center):
        geo = self.geometry()
        geo.moveCenter(center)
        self.setGeometry(geo)
        self.show()
    
    def _make_connections(self, args):
        self.save_btn.clicked.connect(lambda: self.save(args))

    def save(self, args):
        args.i = "true" if self.infer_all.isChecked() else "false"

        self.hide()

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