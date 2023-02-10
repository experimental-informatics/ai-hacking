import PIL
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

def dummy(images, **kwargs):
    return images, False   

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.enable_sequential_cpu_offload()
pipe.safety_checker = dummy
size = 512

while True:
    img = Image.open(requests.get(f"https://thispersondoesnotexist.com/image", stream=True).raw).convert("RGB").resize((size, size))
    prompt = "make it tiktok"
    pipe(prompt, image=img, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=15).images[0].save("happy.png")
    img.save("sad.png")
