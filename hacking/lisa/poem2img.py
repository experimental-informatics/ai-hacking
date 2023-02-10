# generating i images of poem5 with stability. used first one for img2img video

from PIL import Image
import requests
import sys
sys.path.append("../..") 
from utils.utils import AI
#import the py file utils in folder utils as name utils
import utils.utils as utils

f = open ("gen_text/poem9.txt")

text = f.read() 

dirName = utils.makeDir("lisa_generation/"+"poem9"+utils.getTimestamp())


i = 0 
model = AI('txt2img_stablity')
while i < 300:
    
    #text = poem 
    
    img = model.run(prompt=text,seed=i)

    filename = f"{str(i).zfill(99)}.jpg"
    img.save(dirName / filename)

    i = i + 1

f.close()


