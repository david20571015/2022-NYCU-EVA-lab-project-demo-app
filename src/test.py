import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image
import pathlib

from model import Model
dir = str(pathlib.Path(__file__).parent.resolve())
model = Model()

with Image.open(dir + '/model/img1.png') as left_img:
    left_image = TF.to_tensor(left_img)

with Image.open(dir + '/model/img2.png') as right_img:
    right_image = TF.to_tensor(right_img)

print(f'left_image.shape: {left_image.shape}')  # [3, 256, 256]
print(f'right_image.shape: {right_image.shape}')  # [3, 256, 256]

images = torch.stack((left_image, right_image))
print(f'images.shape: {images.shape}')  # [2, 3, 256, 256]

outputs = model.diffusion(images)
print(f'outputs.shape: {outputs.shape}')  # [2, 3, 256, 256]
save_image(outputs, dir + '/model/outputs.png')

left_diff_image, right_diff_image = torch.split(outputs, 1)
print(f'left_diff_image.shape: {left_diff_image.shape}')  # [1, 3, 256, 256]
print(f'right_diff_image.shape: {right_diff_image.shape}')  # [1, 3, 256, 256]

pred_image = model.image_blending(left_diff_image, right_diff_image)
print(f'pred_image.shape: {pred_image.shape}')  # [3, 256, 768]
save_image(pred_image, dir + '/model/pred_image.png')
