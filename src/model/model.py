import torch
# from PyQt5.QtCore import QObject, pyqtSignal

from diffusion import Diffusion
from blending import Blending


class Model(QObject):

    def __init__(self):
        super().__init__()
        ddim_update = pyqtSignal(list)
        image_blending_update = pyqtSignal(list)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.diffusion_model = Diffusion()
        self.blending_model = Blending(skip=[0, 1, 2, 3, 4],
                                       attention=[1]).to(self.device)

    # TODO: diffusion
    def diffusion(self, images):
        # [batch_size, 3, 256, 256]
        return self.diffusion_model.inference(images)

    # TODO: ddim and image_blending
    def run(self, images):
        pass

    # TODO: image_blending
    def image_blending(self, left_image, right_image):
        # [1, 3, 256, 256], [1, 3, 256, 256]
        # left_image = torch.Tensor(left_image).to(self.device)
        # right_image = torch.Tensor(right_image).to(self.device)

        pred_image, *_ = self.blending_model(left_image, right_image)

        return pred_image
