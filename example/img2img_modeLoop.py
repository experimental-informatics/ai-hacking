from PIL import Image
import requests


import sys 
sys.path.append("..") 
from utils.utils import AI 

m1 = AI('img2img_stability',dirName="loop")
m2 = AI('img2img_dalle',dirName="loop")

width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((width, height))

i = 0
filename = str(i) + '.jpg'
image.save(m1.dir / filename)

while i < 10:
    i = i + 1
    #image = image.resize((width / 4, height / 4))

    if(i%2==0):
        image = m1.run(img=image)
    else:
        image = m2.run(img=image)
    filename = str(i) + '.jpg'
    image.save(m1.dir / filename)