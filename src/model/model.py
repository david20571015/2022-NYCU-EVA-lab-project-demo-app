from PyQt5.QtCore import QObject, pyqtSignal


class Model(QObject):

    ddim_update = pyqtSignal(list)
    image_blending_update = pyqtSignal(list)

    def __init__(self):
        super().__init__()

    # TODO: ddim and image_blending
    def run(self, imgs):
        pass
    
    # TODO: ddim
    def ddim(self, imgs):
        # self.ddim_update.emit(self.ddim_results)
        pass

    # TODO: image_blending
    def image_blending(self, imgs):
        # self.image_blending_update.emit(self.image_blending_results)
        pass


        