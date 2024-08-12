import asyncio
import base64
import json
import os
import io
import logging
import time
import configparser
import argparse
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
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import subprocess

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
#quantization_config = BitsAndBytesConfig(load_in_8bit=True)

class CompletionRequest(BaseModel):
    prompt: str
    n: Optional[int] = 42
    image: Optional[UploadFile] = None
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "smooth"
    style: Optional[str] = "0.6"
    user: Optional[str] = None
    negprompt: Optional[str] = None

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')
parser.add_argument('--repo_str', type=str, default='stable-diffusion-3-medium', help='The model repository name')

args = parser.parse_args()
repo_str = args.repo_str

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
host = config.get('settings', 'host')
#port = config.getint('settings', 'port')
maxh = config.get(repo_str, 'maxh')
maxw = config.get(repo_str, 'maxw')
max_height = int(maxh)
max_width = int(maxw)
minh = config.get(repo_str, 'minh')
minw = config.get(repo_str, 'minw')
min_height = int(minh)
min_width = int(minw)
upload_url = config.get('settings', 'upload_url')
path_url = config.get('settings', 'path_url')

port = args.port if args.port is not None else config.getint('settings', 'port')

seed = 42
generator = torch.Generator("cpu").manual_seed(seed)
text_encoder = T5EncoderModel.from_pretrained(
    repo_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)
stable_diffusion = StableDiffusion3Pipeline.from_pretrained(
    repo_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    max_memory = {0:"4GB", 1:"12GB"},
    torch_dtype=torch.float16
)

#stable_diffusion.low_cpu_mem_usage=False
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

def image_request(prompt: str, size: str, response_format: str, seed: int = 42, negprompt: str ="", ipimage: Optional[Image.Image] = None, tempmodel: str = 'XL'):

    try:
        w, h = map(int, size.split('x'))
        # Ensure w and h do not exceed max_width and max_height
        w = min(w, max_width)
        h = min(h, max_height)
        w = max(w, min_width)
        h = max(h, min_height)
    except ValueError:
        #ignore
        return None
    generator = torch.Generator("cpu").manual_seed(seed)
    
    args_dict = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "height": h,
        "width": w,
        "guidance_scale": 7.5,
        "num_inference_steps": 20
    }

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
        response_data = image_request(request.prompt, request.size, request.response_format, request.n, request.negprompt)
    
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
        "response_format": form_data.get("response_format") if form_data.get("response_format") else "url",
        "quality": form_data.get("quality"),
        "style": form_data.get("style"),
        "size": form_data.get("size"),
        "negprompt": form_data.get("negprompt") if form_data.get("negprompt") else "",
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
        response_data = image_request(request.prompt, request.size, request.response_format, request.n, request.negprompt, tensor_image)
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data

@app.get("/nvidia-smi")
async def get_nvidia_smi():
    # Execute the nvidia-smi command
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    nvidia_smi_output = result.stdout.strip()  # Remove any extra whitespace
    # Split the output by lines and then by commas
    gpu_data = []
    for line in nvidia_smi_output.split("\n"):
        utilization, memory_used, memory_total = line.split(", ")
        # Strip the '%' and 'MiB' and convert to appropriate types
        utilization = float(utilization.strip(' %'))
        memory_used = int(memory_used.strip(' MiB'))
        memory_total = int(memory_total.strip(' MiB'))
        gpu_data.append({
           "utilization": utilization,
           "memory_used": memory_used,
           "memory_total": memory_total
        })
    return gpu_data



@app.get('/ping')
async def get_status():
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="error")
