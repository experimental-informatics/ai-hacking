import utils
from PIL import Image
import requests
from utils import AI

txt2img = AI('txt2img_stablediffusion', dirName='txt2img_loop')
img2text = AI('img2text_gpt2', dirName='txt2img_loop')

width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((width, height))

i = 0
filename = str(i) + '.jpg'
image.save(m.dir / filename)

while i < 10:
    i = i + 1
    txt = img2text.run(img=image)
    image = image.resize((width / 4, height / 4))
    image = m.run(img=image)
    filename = str(i) + '.jpg'
    image.save(m.dir / filename)