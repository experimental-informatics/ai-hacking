from PIL import Image
import requests

import os
import sys 
sys.path.append("..") 
from utils.utils import AI 

for k in range(5):
    m1 = AI('img2img_stability',dirName="adversarial/"+str(k))
    m2 = AI('img2img_dalle',dirName="adversarial/"+str(k))

    width = 512
    height = 512
    url = f"https://picsum.photos/{width}/{height}"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image.resize((width, height))

    i = 0
    filename = str(i) + '.jpg'
    image.save(m1.dir / filename)
    
    while i < 100:
        i = i + 1

        print(k,i)
        #image = image.resize((width / 4, height / 4))

        if(i%2==0):
            image = m1.run(img=image)
        else:
            image = m2.run(img=image)
        filename = str(i) + '.jpg'
        image.save(m1.dir / filename)


    os.popen(f"ffmpeg -framerate 8 -pattern_type glob -i '{m1.dirName}/*.jpg'  'adversarialVideo/'{str(k)}.mp4")
