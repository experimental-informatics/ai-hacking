from PIL import Image
import requests


import sys # added!
sys.path.append("..") # added!
from utils.utils import AI 

m = AI('upscale_stablediffusion')

width = 128
height = 128
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((width, height))

i = 0
filename = str(i) + '.jpg'
image.save(m.dir / filename)

while i < 10:
    i = i + 1
    image = image.resize((int(width / 4), int(height / 4)))
    image = m.run(img=image)
    filename = str(i) + '.jpg'
    print(filename)
    image.save(m.dir / filename)