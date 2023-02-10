import torch
from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI 

# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__.callback
# https://github.com/huggingface/diffusers/pull/1150

def decode(latents, name):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        print(latents.size())
        image = m.pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = m.pipe.numpy_to_pil(image)

        # do something with the Images
        for i, img in enumerate(image):
            img.save(f"callback/{name}_img{i}.jpg") 


def decode_from(step, timestep, latents):
    # convert latents to image
    decode(latents, "step")


def save_latents0(step, timestep, latents):
    global latents0
    latents0 = latents

def save_latents1(step, timestep, latents):
    global latents1
    latents1 = latents

dirName="callback"
m = AI('img2img_stablediffusion')

latents0 = None
latents1 = None

width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save(dirName + "/0.jpg")
m.run(img=image, callback=save_latents0, num_inference_steps=512, strength=1.0, callback_steps=10)

url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save(dirName + "/1.jpg")
m.run(img=image, callback=save_latents1, num_inference_steps=512, strength=1.0, callback_steps=10)

latentsC = torch.lerp(latents0, latents1, 0.5)
decode(latents0, '0')
decode(latents1, '1')
decode(latentsC, 'C')
