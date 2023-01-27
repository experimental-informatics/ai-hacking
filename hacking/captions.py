import torch
from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI
#import the py file utils in folder utils as name utils
import utils.utils as utils


dirName = utils.makeDir("captions_clip/"+utils.getTimestamp())


img = utils.loremImage()
i = 0
filename = f"{str(i).zfill(4)}.jpg"
img.save(dirName / filename)

with open(dirName / 'clip.txt', 'w', encoding='utf-8') as f:
    f.write(f"{filename}@init\n")


while i < 1000:
    
    i = i + 1

    model = AI('img2txt_clip')
    txt = model.run(img)
    print(txt)

    filename = f"{str(i).zfill(4)}-{utils.fixFilename(txt)}.jpg"

    with open(dirName / 'clip.txt', 'a',encoding='utf-8') as f:
        f.write(f"{filename}@{txt}\n")
    
    model = AI('txt2img_stablediffusion')
    img = model.run(prompt=txt)

    img.save(dirName / filename)