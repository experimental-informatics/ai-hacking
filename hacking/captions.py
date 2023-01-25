import torch
from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI

def fix_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()

img2txt = AI('img2txt_gpt2', dirName=("captions/"))
# txt2img = AI('txt2img_stablediffusion', dirName=("captions/"))

width = 512
height = 512

url = f"https://picsum.photos/{width}/{height}"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
img = img.resize((width, height))

# i = 0
# filename = str(i) + '.jpg'
# img.save(img2txt.dir / filename)

# while i < 1000:
#     i = i + 1
#     txt = img2txt.run(img)
#     print(txt)
#     img = txt2img.run(prompt=txt)
#     filename = f"{str(i)}-{fix_filename(txt)}.jpg"
#     img.save(img2txt.dir / filename)