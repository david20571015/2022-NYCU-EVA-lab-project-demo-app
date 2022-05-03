import sys
from PyQt5.QtWidgets import QApplication
from model.model import Model
from controllers.main_ctrl import MainController
from views.main_view import MainView
import argparse

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--name','-n', type=str, default="project 1", help="name of the project")
    parser.add_argument('--canvas', type=int, default=4, help="number of canvas")
    parser.add_argument('--width', type=int, default=256, help="width of input image")
    parser.add_argument('--height', type=int, default=256, help="height of input image")
    parser.add_argument('--out_width', type=int, default=4, help="times of gif width in input image")
    parser.add_argument('--duration', type=int, default=20, help="duration of gif in milliseconds")
    parser.add_argument('-i', type=str, default="true", help="infer all combinations of input")
    args = parser.parse_args()
    return args

class App(QApplication):

    def __init__(self, args):
        app_name = ["2022 NYCU EVA lab project demo app"]
        super(App, self).__init__(app_name)

        # Force the style to be the same on all OSs:
        self.setStyle("Fusion")

        # Connect everything together
        self.model = Model()
        self.main_ctrl = MainController(self.model)
        self.main_view = MainView(args, self.model, self.main_ctrl)
        self.main_view.show()

if __name__ == '__main__':
    args = parse_args_and_config()
    app = App(args)
    sys.exit(app.exec_())
