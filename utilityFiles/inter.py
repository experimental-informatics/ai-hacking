import PIL
from PIL import Image
import requests
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import AutoProcessor, CLIPModel

def dummy(images, **kwargs):
    return images, False   

size = 256
img0 = Image.open(requests.get(f"https://thispersondoesnotexist.com/image", stream=True).raw).convert("RGB").resize((size, size))
img1 = Image.open(requests.get(f"https://thispersondoesnotexist.com/image", stream=True).raw).convert("RGB").resize((size, size))

model_id_or_path = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16, 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipe.to(device)

pipe.safety_checker = dummy

image = img0
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
print(image_features.size())

image = pipe.vae.decode(image_features).sample
