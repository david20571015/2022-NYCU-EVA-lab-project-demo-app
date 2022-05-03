import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import threading
from .diffusion import Diffusion
from .blending import Blending


class Model(QThread):

    finished = pyqtSignal()
    ddim_changed = pyqtSignal(str, int, torch.Tensor)
    image_blending_changed = pyqtSignal(str, int, torch.Tensor)
    
    def __init__(self):
        super().__init__()
        import time
        t0= time.time()
        self.models_thread = threading.Thread(target = self.load_models)
        self.models_thread.start()
        t1 = time.time() - t0
        print("Loading time: ", t1) # CPU seconds elapsed (floating point)

    def load_models(self):
        self.diffusion_model = Diffusion()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.blending_model = Blending(skip=[0, 1, 2, 3, 4],
                                attention=[1]).to(self.device)

        import os.path
        blending_weighits_path = os.path.join(os.path.dirname(__file__),
                                              './blending/weights/Gen_5')
        self.blending_model.load_state_dict(torch.load(blending_weighits_path))

    def set_strokes(self, images):
        self.strokes_images = []
        for img in images:
            tf_buffer = TF.to_tensor(img)
            unsqueeze_image = tf_buffer.unsqueeze(0)
            self.strokes_images.append(unsqueeze_image)

    # ddim and image_blending
    def run(self):

        # check model loading
        self.models_thread.join()

        # ddim inference
        self.ddim_images = []
        for id, img in enumerate(self.strokes_images):
            ddim_result = self.diffusion(img)
            # clone = ddim_result.squeeze().cpu().clone().detach()
            self.ddim_images.append(ddim_result)
            self.ddim_changed.emit("ddim_", id, ddim_result)
            
        # image blending inference
        for i in range(len(self.ddim_images)):
            blending_result = self.image_blending(self.ddim_images[i], self.ddim_images[(i+1)%len(self.ddim_images)])
            # 0, 2, 4, 6
            self.image_blending_changed.emit("blending_", 2*i, self.ddim_images[i])
            # 1, 3, 5, 7
            self.image_blending_changed.emit("blending_", 1+2*i, blending_result)

        # recovery UI function
        self.finished.emit()

    # diffusion
    def diffusion(self, image):
        result_image = self.diffusion_model.inference(image,
                                                        sample_step=2,
                                                        total_noise_levels=300)
        return (result_image + 1) * 0.5

        # [batch_size, 3, 256, 256]
        # return self.diffusion_model.inference(images, sample_step=2, total_noise_levels=300).numpy()

    # image_blending
    def image_blending(self, left_image, right_image):
        # [1, 3, 256, 256], [1, 3, 256, 256]
        # left_image = torch.Tensor(left_image).to(self.device)
        # right_image = torch.Tensor(right_image).to(self.device)

        pred_image, *_ = self.blending_model(left_image, right_image)
        _, mid_image, _ = torch.chunk(pred_image, 3, dim=-1)
        
        # [3, 256, 768]
        return mid_image

        # [3, 256, 768]
        # return pred_image.squeeze().numpy()