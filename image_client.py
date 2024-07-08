import asyncio
import base64
import json
import os
import io
import logging
import time
import configparser
from typing import AsyncIterable, List, Generator, Union, Optional
import torch
from PIL import Image

import requests
import sseclient
import random

from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from pydantic import BaseModel
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

logging.basicConfig(level=logging.DEBUG)

class CompletionRequest(BaseModel):
    prompt: str
    n: Optional[int] = 42
    image: Optional[UploadFile] = None
    response_format: Optional[str] = "b64_json"
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "smooth"
    style: Optional[str] = "0.6"
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('be-stable-diffusion-xl-base-1.0', 'repo')
adapter_id = config.get('ip-adapter', 'repo')
host = config.get('settings', 'host')
port = config.getint('settings', 'port')
upload_url = config.get('settings', 'upload_url')
path_url = config.get('settings', 'path_url')

scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None)
seed = 42
generator = torch.Generator("cpu").manual_seed(seed)

stable_diffusion.load_ip_adapter(adapter_id, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

stable_diffusion.to(f"cuda")

#stable_diffusion.enable_vae_slicing()
#stable_diffusion.enable_sequential_cpu_offload()
stable_diffusion.enable_model_cpu_offload()
print("*** Loaded.. now Inference...:")

app = FastAPI(title="StableDiffusion")

def upload_image(image_file, upload_url, filename, wallet_address):
    image_buffer = io.BytesIO()
    image_file.save(image_buffer, format='JPEG')
    image_buffer.seek(0)
    files = {'file': (filename+'.jpg', image_buffer, 'image/jpeg')}
    data = {'walletAddress': wallet_address}
    response = requests.post(upload_url, files=files, data=data)

    if response.status_code == 200:
        print(f'Successfully uploaded image')
        return response
    else:
        print(f'Failed to upload image. Status code: {response.status_code}')
        return None

def image_request(prompt: str, size: str, response_format: str, seed: int = 42, ipimage: Optional[Image.Image] = None, tempmodel: str = 'XL'):

    negprompt = ""
    w, h = map(int, size.split('x'))
    generator = torch.Generator("cpu").manual_seed(seed)

    args_dict = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "height": h,
        "width": w,
        "generator": generator,
        "num_inference_steps": 20
    }
    ipimagenone = load_image("512x512bb.jpeg")
    # Conditionally add the ip_adapter_image argument
    if ipimage is not None:
        args_dict["ip_adapter_image"] = ipimage
        stable_diffusion.set_ip_adapter_scale(0.5)
    else:
        args_dict["ip_adapter_image"] = ipimagenone
        stable_diffusion.set_ip_adapter_scale(0.0)


    image = stable_diffusion(**args_dict).images[0]

    random_string = str(random.randint(100000, 999999))
    filename = "ai_seed"+str(seed)+"_"+random_string

    # Send appropriate response based on response_format

    if response_format == "url":
        response = upload_image(image, upload_url,filename,"ai")
        if response.status_code == 200:
            print("Generation and upload successful.", filename)

        response_data = {
            "created": int(time.time()),
            "data": [
                {
                    "url": path_url+filename+".jpg",
                }
            ]
            }
        
    elif response_format == "b64_json":
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        response_data = {
            "created": int(time.time()),
            "data": base64.b64encode(image_buffer.read()),
        }


    return response_data



@app.post('/v1/images/generations')
async def main(request: CompletionRequest):

    response_data = None
    try:
        response_data = image_request(request.prompt, request.size, request.response_format, request.n)
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data

@app.post('/v1/images/edits')
async def edits(inrequest: Request):
    form_data = await inrequest.form()
    imageup: UploadFile = form_data.get("image")

    # Extract other fields and create a dictionary to create a CompletionRequest instance
    completion_request_data = {
        "prompt": form_data.get("prompt"),
        "n": int(form_data.get("n")) if form_data.get("n") else None,
        "model": form_data.get("model"),
        "response_format": form_data.get("response_format"),
        "quality": form_data.get("quality"),
        "style": form_data.get("style"),
        "size": form_data.get("size"),
        "user": form_data.get("user")
    }

    # Create an instance of CompletionRequest
    request = CompletionRequest(**completion_request_data)

    tensor_image = None
    if imageup:
        image_data = await imageup.read()
        tensor_image = Image.open(io.BytesIO(image_data))
        tensor_image = tensor_image.resize((512, 512))
    else:
        tensor_image = None

    response_data = None
    try:
        response_data = image_request(request.prompt, request.size, request.response_format, request.n, tensor_image)
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data


@app.get('/ping')
async def get_status():
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
