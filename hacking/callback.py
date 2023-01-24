from diffusers import StableDiffusionPipeline
import torch

# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__.callback
# https://github.com/huggingface/diffusers/pull/1150

def dummy(images, **kwargs):
    return images, False  

def decode_from(step, timestep, latents):
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image.save(f"./callback/{step}_{timestep}.jpg")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = dummy

img = pipe(prompt="a dog", guidance_scale=10, num_inference_steps=100, callback=decode_from).images[0]

