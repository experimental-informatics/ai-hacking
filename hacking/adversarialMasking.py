from PIL import Image, ImageDraw, ImageOps
import requests

import os
import sys 
sys.path.append("..") 
from utils.utils import AI 




for k in range(2):
    m1 = AI('img2img_stablediffusion',dirName="adversarialMasking/"+str(k))
    m2 = AI('img2img_dalle',dirName="adversarialMasking/"+str(k))

    width = 512
    height = 512
    url = f"https://picsum.photos/{width}/{height}"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((width, height))

    i = 0
    filename = str(i) + '.jpg'
    image.save(m1.dir / filename)
    
    border = 256
    while i < 256:

        i = i + 1
        border = border-1
        print(k,i)
        im = Image.new('RGB', (512, 512), (0,0,0))
        draw = ImageDraw.Draw(im)
        shift = border
        coloring = 255
        draw.rectangle((shift,shift,width-shift,height-shift), (coloring,coloring,coloring))
        im = im.convert('L')

        #msk = ImageOps.invert(im)
        #image = image.resize((width / 4, height / 4))
        
        if(i%2==0):
            image_temp = m1.run(img=image)
            image = Image.composite(image,image_temp, im)
        else:
            image_temp = m2.run(img=image)
            image = Image.composite(image,image_temp, im)

        filename = str(i) + '.jpg'
        image.save(m1.dir / filename)


    
