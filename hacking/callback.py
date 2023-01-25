import torch
from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI 

# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__.callback
# https://github.com/huggingface/diffusers/pull/1150

def decode_from(step, timestep, latents):
    # convert latents to image
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
            img.save(f"{m.dir}/iter_{step}_img{i}.jpg")


m = AI('img2img_stablediffusion',dirName=("callback"))

width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save(m.dir / "0.jpg")
m.run(img=image, callback=decode_from, num_inference_steps=512, strength=1.0)
