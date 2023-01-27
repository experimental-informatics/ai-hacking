import requests
from PIL import Image
import sys
sys.path.append("..") 
from utils.utils import AI



m = AI('img2txt_clip')

while True:
    url = input('input: ')
    try:
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        text = m.run(img)
        print(text)
    except:
        continue
