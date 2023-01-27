from PIL import Image
import requests
import sys
sys.path.append("..") 
from utils.utils import AI
import utils.utils as utils

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

happiness = 0.0
sadness = 100.0
i = 0
while True:
    i=i+1
    #img = utils.loremImage()

    url = 'https://thispersondoesnotexist.com/image'
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    inputs = processor(text=["jonny depp"], images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    
    now = logits_per_image.item()
   
    print(f"{i}\tnow:{now}\tmax:{happiness}\tmin:{sadness}")
    if(now > happiness):
        happiness = now
        img.save('happestImage.jpg')
    if(now < sadness):
        sadness = now
        img.save('sadestImage.jpg')
    img.save('this.jpg')
        
    