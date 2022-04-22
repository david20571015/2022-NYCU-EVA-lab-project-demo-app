import torch
from PyQt5.QtCore import QObject, pyqtSignal

from .diffusion import Diffusion
from .blending import Blending


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
        import os.path
        blending_weighits_path = os.path.join(os.path.dirname(__file__),
                                              './blending/weights/Gen_100')
        self.blending_model.load_state_dict(torch.load(blending_weighits_path))

    # TODO: ddim and image_blending
    def run(self, images):
        pass

    # TODO: diffusion
    def diffusion(self, images):
        # [batch_size, 3, 256, 256]
        diffusion_image = self.diffusion_model.inference(images,
                                                         sample_step=2,
                                                         total_noise_levels=300)
        return (diffusion_image + 1) * 0.5
        # [batch_size, 3, 256, 256]
        # return self.diffusion_model.inference(images, sample_step=2, total_noise_levels=300).numpy()

    # TODO: image_blending
    def image_blending(self, images):
        # images is a list which contains images with shape [1, 3, 256, 256]
        # image = torch.Tensor(image).to(self.device)

        if len(images) < 2:
            return images

        image_list = []

        for left_image, right_image in zip(images[:-1], images[1:]):
            image_list.append(left_image)
            pred_image, *_ = self.blending_model(left_image, right_image)
            _, mid_image, _ = torch.chunk(pred_image, 3, dim=-1)
            image_list.append(mid_image)

        image_list.append(images[-1])

        # [2 * batch_size - 1, 256, 256]
        return torch.cat(image_list, 0)

        # [2 * batch_size - 1, 256, 256]
        # return torch.cat(image_list, 0).numpy()
