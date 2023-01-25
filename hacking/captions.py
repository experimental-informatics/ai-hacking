import torch
from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI

img2txt = AI('img2text_gpt2',dirName=("captions/"))
txt2img = AI('img2img_stablediffusion',dirName=("captions/"))

width = 512
height = 512

url = f"https://picsum.photos/{width}/{height}"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
img = img.resize((width, height))

i = 0
filename = str(i) + '.jpg'
img.save(m.dir / filename)

while i < 1000:
    i = i + 1
    txt = img2txt.run(img)
    img = img2txt.run(img=img)
    filename = str(i) + '.jpg'
    img.save(m.dir / filename)