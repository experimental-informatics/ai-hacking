from PIL import Image, ImageDraw
import requests

import os
import sys 
sys.path.append("..") 
from utils.utils import AI

m1 = AI('img2img_stablediffusion',dirName="croploop")

