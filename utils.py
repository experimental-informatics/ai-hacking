
import torch
from PIL import Image,ImageDraw 
import numpy as np
from RealESRGAN import RealESRGAN 
import glob
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from pathlib import Path
import time
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO


def dummy(images, **kwargs):
		return images, False   


class sd:
    def __init__(self, mode=None, dirName=time.strftime("%Y%m%d-%H%M%S")):
        self.dirName = dirName
        self.DIR = Path.cwd() / self.dirName
        self.DIR.mkdir(parents=True,exist_ok=True)
 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if mode == 'upscale':
            self.model = RealESRGAN(self.device, scale=4)
            self.model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        elif mode == 'img2img':
            self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id_or_path,
                revision="fp16", 
                torch_dtype=torch.float16, 
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.safety_checker = dummy
    
    def upscale_RealESRGAN(self,img):
        image = self.model.predict(img)
        return image

    def img2img(self,img,prompt=''):
        image = self.pipe(prompt=prompt, init_image=img, strength=0.55, guidance_scale=6).images[0]
        return image
   


def upscale_RealESRGAN():
    return


def upscale_StableDiffusion():
    return


def img2img_StableDiffusion():
    return


def text2img_StableDiffusion():
    return


def img2text_gpt2():
    return


def img2text_clip():
    return

def img2segmentations():
    return 