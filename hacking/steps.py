from PIL import Image
import requests
import sys 
sys.path.append("..") 
from utils.utils import AI 
import time
import os


stepsCounter = [32,64,128,256,512]

for k in stepsCounter:
    m = AI('img2img_stablediffusion',dirName=("steps/" + str(k)))

    width = 512
    height = 512
    url = f"https://picsum.photos/{width}/{height}"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((width, height))

    i = 0
    filename = str(i).zfill(5) + '.jpg'
    image.save(m.dir / filename)

    while i < 70:
        i = i + 1
        #image = image.resize((width / 4, height / 4))
        image = m.run(img=image,num_inference_steps=k)
        filename = str(i).zfill(5) + '.jpg'
        image.save(m.dir / filename)
           

for k in stepsCounter:
    m = AI('img2img_stablediffusion',dirName=("steps_apple/" + str(k)))

    width = 512
    height = 512
    url = f"https://picsum.photos/{width}/{height}"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((width, height))

    i = 0
    filename = str(i).zfill(5) + '.jpg'
    image.save(m.dir / filename)

    while i < 70:
        i = i + 1
        #image = image.resize((width / 4, height / 4))
        image = m.run(prompt="apple",img=image,num_inference_steps=k)
        filename = str(i).zfill(5) + '.jpg'
        image.save(m.dir / filename)
           