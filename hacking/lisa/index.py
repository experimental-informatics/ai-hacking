from PIL import Image
import requests
import sys
sys.path.append("../..") 
from utils.utils import AI
#import the py file utils in folder utils as name utils
import utils.utils as utils


dirName = utils.makeDir("lisa_generation/"+utils.getTimestamp())


i = 0 
model = AI('txt2img_stablity')
while i < 100:
    
    text='One horrid contraption struck unforgivably on yet another in the blistering room of fury emanations unbeknown.'
    

   
    img = model.run(prompt=text,seed=i)

    filename = f"{str(i).zfill(9)}.jpg"
    img.save(dirName / filename)

    i = i + 1