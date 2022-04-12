import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

from model import Model

model = Model()

with Image.open('./model/img1.png') as image1, \
     Image.open('./model/img2.png') as image2, \
     Image.open('./model/img3.png') as image3:
    img1 = TF.to_tensor(image1)
    img2 = TF.to_tensor(image2)
    img3 = TF.to_tensor(image3)

print(f'img1.shape: {img1.shape}')  # [3, 256, 256]
print(f'img2.shape: {img2.shape}')  # [3, 256, 256]
print(f'img3.shape: {img3.shape}')  # [3, 256, 256]

images = torch.stack((img1, img2, img3))
print(f'images.shape: {images.shape}')  # [3, 3, 256, 256]

diffusion_image = model.diffusion(images)
print(f'diffusion_image.shape: {diffusion_image.shape}')  # [3, 3, 256, 256]
save_image(diffusion_image, './model/diffusion_image.png')

blending_image = model.image_blending(torch.split(diffusion_image, 1))
print(f'blending_image.shape: {blending_image.shape}')  # [5, 3, 256, 256]
save_image(blending_image, './model/blending_image.png')
