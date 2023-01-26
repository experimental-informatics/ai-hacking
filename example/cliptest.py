from PIL import Image
from clip_interrogator import Config, Interrogator
import requests
import torch

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


def fix_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

width = 512
height = 512

url = f"https://picsum.photos/{width}/{height}"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
img = img.resize((width, height))

with open('cliploop.txt', 'w') as f:
    f.write('init')

for i in range(10000):
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    t = ci.interrogate(img)
    print(t)


    model_id = "runwayml/stable-diffusion-v1-5"

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    ci = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    ci = ci.to("cuda")


    prompt = t
    

    with open('cliploop.txt', 'a') as f:
        f.write(f"\n{str(i).zfill(4)}@{fix_filename(t)}")

    img = ci(prompt).images[0]
    img.save(f"cliploop/{str(i).zfill(4)}.png")
