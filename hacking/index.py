import utils
from PIL import Image
import requests
m = utils.sd('img2img',dirName='hello')

width = 512
height = 512
url = f"https://picsum.photos/{width}/{height}"
print(url)
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((width, height))

fileName = '0.png'
m.img2img(image).save(m.DIR / fileName)