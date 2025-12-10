import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"

import torch
torch.cuda.empty_cache()
from diffusers import StableDiffusionImageVariationPipeline
from diffusers.utils import load_image

# switch to "mps" for apple devices
#pipe = DiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", dtype=torch.bfloat16, device_map="cuda")
pipe = StableDiffusionImageVariationPipeline.from_pretrained("/home/kojo/Code/sd-image-variations-diffusers/", local_files_only=True)
#pipe = pipe.to("cuda")

#prompt = "Turn this cat into a dog"
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(image=input_image, num_images_per_prompt=3).images[0]

image.save("result.jpg")
