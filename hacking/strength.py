from PIL import Image
import requests
import sys 
sys.path.append("..") 
from utils.utils import AI 
import time
import os



m = AI('img2img_stablediffusion',dirName=("sampling/"))
width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((width, height))

m.run(img=image)
'''
for k in range(5,15):
    m = AI('img2img_stablediffusion',dirName=("findingRed/" + str(k)))

    width = 512
    height = 512
    url = f"https://picsum.photos/{width}/{height}"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((width, height))

    i = 0
    filename = str(i) + '.jpg'
    image.save(m.dir / filename)

    while i < 1000:
        i = i + 1
        #image = image.resize((width / 4, height / 4))
        image = m.run(img=image,guidance_scale=k,num_inference_steps=50)
        filename = str(i) + '.jpg'
        image.save(m.dir / filename)
    os.popen(f"ffmpeg -framerate 12 -pattern_type glob -i '{m.dirName}/*.jpg'  'findingRedVideo/'{str(k)}.mp4")
           
'''