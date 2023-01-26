import os
import time
from io import BytesIO
from base64 import b64decode
import torch
import numpy as np
import glob
from pathlib import Path
from PIL import Image, ImageDraw

from clip_interrogator import Config, Interrogator
import transformers
from RealESRGAN import RealESRGAN 
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline
#from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, StableDiffusionPipeline
import openai as openai
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from torchvision.transforms import GaussianBlur

import config


def dummy(images, **kwargs):
    return images, False   

def modes():
    print("possible modes:\n\n")
    print("upscale_realesrgan",end=', ')
    print("upscale_stablediffusion",end=', ')
    print("txt2img_stablediffusion",end=', ')
    print("img2img_stablediffusion",end=', ')
    print("img2img_stability",end=', ')
    print("txt2img_stablity",end=', ')
    print("txt2img_dalle",end=', ')
    print("img2img_dalle",end=', ')
    print("img2txt_gpt2",end='\n')
    print("img2txt_clip",end='\n')

class AI:
    def __init__(self, mode=None, stability_engine='stable-diffusion-v1-5', dirName=time.strftime("%Y%m%d-%H%M%S")):
        self.dirName = dirName
        self.dir = Path.cwd() / self.dirName
        self.dir.mkdir(parents=True,exist_ok=True)
        self.mode = mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if mode == 'upscale_realesrgan':
            self.model = RealESRGAN(self.device, scale=4)
            self.model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        elif mode == 'upscale_stablediffusion':
            self.model_id = "stabilityai/stable-diffusion-x4-upscaler"
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to("cuda")
            self.pipe.safety_checker = dummy
        elif mode == 'txt2img_stablediffusion':
            self.model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to("cuda")
            self.pipe.safety_checker = dummy
        elif mode == 'img2img_stablediffusion':
            self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id_or_path,
                revision="fp16", 
                torch_dtype=torch.float16, 
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.safety_checker = dummy
        elif mode == 'img2img_stability' or mode == 'txt2img_stablity':
            # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
            # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0

            os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
            os.environ['STABILITY_KEY'] = config.stability_api_key
            self.stability_api = client.StabilityInference(
                key=os.environ['STABILITY_KEY'], # API Key reference.
                verbose=True, # Print debug messages.
                engine=stability_engine, # Set the engine to use for generation. 
            )
        elif mode == 'txt2img_dalle' or  mode == 'img2img_dalle':
           print("initualizing openai")
        elif mode == "img2txt_gpt2":
            self.pipe = transformers.pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        elif mode == "img2txt_clip":
            self.pipe = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        else:
            modes()


    def run(self,img="",prompt="",mask=None,negative_prompt="",width=512,height=512,seed=992446758,strength=0.5, guidance_scale=5, num_inference_steps=25, callback=None):
        if self.mode == 'upscale_realesrgan':
            return self.upscale_realesrgan(img=img)
        elif self.mode == 'upscale_stablediffusion':
            return self.upscale_stablediffusion(img=img,prompt=prompt, callback=callback)
        elif self.mode == 'txt2img_stablediffusion':
            return self.txt2img_stablediffusion(prompt=prompt,negative_prompt=negative_prompt,strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, callback=callback)
        elif self.mode == 'img2img_stablediffusion':
            return self.img2img_stablediffusion(img=img,prompt=prompt,negative_prompt=negative_prompt,strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, callback=callback)
        elif self.mode == 'img2img_stability':
            return self.img2img_stability(img=img,prompt=prompt,mask=mask,width=width,height=height,seed=seed,strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        elif self.mode == 'txt2img_stablity':
            return self.txt2img_stability(prompt=prompt,width=width,height=height,seed=seed,strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
        elif self.mode == 'txt2img_dalle':
            return self.txt2img_dalle(prompt=prompt)
        elif self.mode == 'img2img_dalle':
            return self.img2img_dalle(img=img,mask=mask,prompt=prompt)
        elif self.mode == 'img2txt_gpt2':
            return self.img2txt_gpt2(img)
        elif self.mode == 'img2txt_clip':
            return self.img2txt_clip(img)
        
        else:
            modes()
            
    def upscale_realesrgan(self,img):
        img = self.model.predict(img)
        return img

    def upscale_stablediffusion(self, img, prompt="", callback=None):
        img = self.pipe(prompt=prompt, image=img, callback=callback).images[0]
        return img

    def img2img_stablediffusion(self, img, prompt='', negative_prompt='', strength=0.5, guidance_scale=5, num_inference_steps=25, callback=None):
        img = self.pipe(prompt=prompt, init_image=img, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, callback=callback).images[0]
        return img

    def txt2img_stablediffusion(self, prompt, negative_prompt='', strength=0.5, guidance_scale=5, num_inference_steps=25, callback=None):
        img = self.pipe(prompt=prompt, negative_prompt=negative_prompt, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, callback=callback).images[0]
        return img

    def img2img_stability(self,img,prompt='',mask=None,width=512,height=512,seed=992446758,strength=0.5, guidance_scale=5.0, num_inference_steps=25):
        
        

        if mask!=None:
            maskPreTemp = mask.copy().resize(img.size)

            img2_grayscale = maskPreTemp.convert('L')
            img2_a = np.array(img2_grayscale)
            npmask = np.array(img2_grayscale)
            npmask[img2_a<150] = 0  # This is the area that will get painted, will show up as grey.
            npmask[img2_a>=150] = 1 # This is the protected area, will show up white. Protected areas won't be affected by our generating.

            np_strength = .8  # This controls the strength of our prompt relative to the init image.

            d = int(255 * (1-np_strength))
            npmask *= 255-d # Converts our range from [0,1] to [0,255]
            npmask += d
            maskTemp = Image.fromarray(npmask)

            blur = GaussianBlur(11,20)
            maskTemp = blur(maskTemp)
            
            #img.show()
            #maskTemp.show()
            answers = self.stability_api.generate(
                prompt=prompt,
                init_image=img,
                mask_image=maskTemp,
                start_schedule = strength,
                seed=seed, # If a seed is provided, the resulting generated image will be deterministic.
                                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                                # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
                steps=num_inference_steps, # Amount of inference steps performed on image generation. Defaults to 30. 
                cfg_scale=guidance_scale, # Influences how strongly your generation is guided to match your prompt.
                            # Setting this value higher increases the strength in which it tries to match your prompt.
                            # Defaults to 7.0 if not specified.
                width=width, # Generation width, defaults to 512 if not included.
                height=height, # Generation height, defaults to 512 if not included.
                samples=1, # Number of images to generate, defaults to 1 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                            # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                            # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
            )
        else:
            answers = self.stability_api.generate(
                prompt=prompt,
                init_image=img,
                start_schedule = strength,
                seed=seed, # If a seed is provided, the resulting generated image will be deterministic.
                                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                                # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
                steps=num_inference_steps, # Amount of inference steps performed on image generation. Defaults to 30. 
                cfg_scale=guidance_scale, # Influences how strongly your generation is guided to match your prompt.
                            # Setting this value higher increases the strength in which it tries to match your prompt.
                            # Defaults to 7.0 if not specified.
                width=width, # Generation width, defaults to 512 if not included.
                height=height, # Generation height, defaults to 512 if not included.
                samples=1, # Number of images to generate, defaults to 1 if not included.
                sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                            # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                            # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
            )
        for resp in answers:
            for artifact in resp.artifacts:
                #if artifact.finish_reason == generation.FILTER:
                #    warnings.warn(
                #        "Your request activated the API's safety filters and could not be processed."
                #        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(BytesIO(artifact.binary))
        return img    
    def txt2img_stability(self,prompt='',width=512,height=512,seed=992446758,strength=0.5, guidance_scale=5.0, num_inference_steps=25):
        answers = self.stability_api.generate(
            prompt=prompt,
            start_schedule = strength,
            seed=seed, # If a seed is provided, the resulting generated image will be deterministic.
                            # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                            # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
            steps=num_inference_steps, # Amount of inference steps performed on image generation. Defaults to 30. 
            cfg_scale=guidance_scale, # Influences how strongly your generation is guided to match your prompt.
                        # Setting this value higher increases the strength in which it tries to match your prompt.
                        # Defaults to 7.0 if not specified.
            width=width, # Generation width, defaults to 512 if not included.
            height=height, # Generation height, defaults to 512 if not included.
            samples=1, # Number of images to generate, defaults to 1 if not included.
            sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
        )
        for resp in answers:
            for artifact in resp.artifacts:
                #if artifact.finish_reason == generation.FILTER:
                #    warnings.warn(
                #        "Your request activated the API's safety filters and could not be processed."
                #        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(BytesIO(artifact.binary))
        return img
         
    def txt2img_dalle(self, prompt=''):
        openai.api_key = config.openai_api_key

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        image_data = b64decode(response["data"][0]["b64_json"])
                
        stream = BytesIO(image_data)
        img = Image.open(stream)
        return img
    def img2img_dalle(self,img,prompt=' ',mask=None):
        openai.api_key = config.openai_api_key
        


        if mask!=None:

            if prompt == "":
                prompt = " "


            tempImg = img.copy()
            tempImg.putalpha(mask.copy().convert('L').resize(tempImg.size))


            temp = BytesIO()
            img.save(temp,format="png")

            tempMask = BytesIO()
            tempImg.save(tempMask,format="png")


            response = openai.Image.create_edit(
            image = temp.getvalue(),
            mask = tempMask.getvalue(),
            prompt = prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
            )
        else: # general variation

            temp = BytesIO()
            img.save(temp,format="png")

            response = openai.Image.create_variation(
            image = temp.getvalue(),
            n=1,
            size="512x512",
            response_format="b64_json",
            )
        
        image_data = b64decode(response["data"][0]["b64_json"])

        stream = BytesIO(image_data)
        img = Image.open(stream)
        return img
        
    def img2txt_gpt2(self, img):
        return self.pipe(img)[0]["generated_text"]

    def img2txt_clip(self, img):
        return self.pipe.interrogate(img)


    # def img2text_clip():
    #     return
     
    # def img2segmentations():
    #     return 



