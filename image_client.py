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
from diffusers import DiffusionPipeline, UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLControlNetPipeline, DDIMScheduler
from diffusers.utils import load_image
import subprocess
import numpy as np

from controlnet_aux import OpenposeDetector

from controlnet_aux.processor import Processor

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
parser.add_argument('--repo_str', type=str, default='be-stable-diffusion-xl-base-1.0', help='The model repository name')
parser.add_argument('--use_ctrlnet', action="store_true", help='Use ControlNet')

# Parse the arguments
args = parser.parse_args()
repo_str = args.repo_str

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
use_ctrlnet = args.use_ctrlnet
controlnet_id = None
sd_controlnet_id = None
controlnet = None

adapter_id = config.get(repo_str, 'iprepo')
adapter_folder = config.get(repo_str, 'ipfolder')
adapter_encoder = config.get(repo_str, 'ipencoder')
adapter_encoder_face = config.get(repo_str, 'ipencoder_face')
maxh = config.get(repo_str, 'maxh')
maxw = config.get(repo_str, 'maxw')
max_height = int(maxh)
max_width = int(maxw)
minh = config.get(repo_str, 'minh')
minw = config.get(repo_str, 'minw')
min_height = int(minh)
min_width = int(minw)
host = config.get('settings', 'host')
upload_url = config.get('settings', 'upload_url')
path_url = config.get('settings', 'path_url')

port = args.port if args.port is not None else config.getint('settings', 'port')

if use_ctrlnet:
    controlnet_id = config.get('control-net', 'repo')
    sd_controlnet_id = config.get('sd-control-net', 'repo')
    # Add ControlNet for openpose
    controlnet = ControlNetModel.from_pretrained(sd_controlnet_id, torch_dtype=torch.float16)

    # Add Openpose
    # openpose = OpenposeDetector.from_pretrained(controlnet_id, hand_and_face=True)
    openpose = Processor("openpose_full")

scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# Load Stable Diffusion ControlNet Pipeline
# 
if use_ctrlnet:
    stable_diffusion_ctrl = StableDiffusionXLControlNetPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None, controlnet=controlnet)
else:
    stable_diffusion_style = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None)
    stable_diffusion_face = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None)

seed = 42
generator = torch.Generator("cpu").manual_seed(seed)

stable_diffusion_style.load_ip_adapter(adapter_id, subfolder=adapter_folder, weight_name=adapter_encoder)
stable_diffusion_face.load_ip_adapter(adapter_id, subfolder=adapter_folder, weight_name=adapter_encoder_face)

#stable_diffusion.enable_vae_slicing()
#stable_diffusion.enable_sequential_cpu_offload()
stable_diffusion_style.enable_model_cpu_offload()
stable_diffusion_face.enable_model_cpu_offload()
print("*** Loaded.. now Inference...:")

image_array = np.zeros((max_height, max_width, 3), dtype=np.uint8)
placeholder_image = Image.fromarray(image_array)
ipimagenone = load_image(placeholder_image)
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


def image_request(prompt: str, size: str, response_format: str, seed: int = 42, negprompt: str = "", ipimage: Optional[Image.Image] = None, user: Optional[str] = None, weight: Optional[str] = '0.5', tempmodel: str = 'XL'):

    ipweight = 0.0
    key = ""
    try: 
        w, h = map(int, size.split('x'))
        # Ensure w and h do not exceed max_width and max_height
        w = min(w, max_width)
        h = min(h, max_height)
        w = max(w, min_width)
        h = max(h, min_height)

        if user is not None:
            key, value = user.split(':')
            # Try to convert the value to a float
            ipweight = float(value)
    except ValueError:
        #ignore
        return None 


    generator = torch.Generator("cpu").manual_seed(seed)
    args_dict = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "height": h,
        "width": w,
        "generator": generator,
        "num_inference_steps": 20
    }
    stable_diffusion = stable_diffusion_style
    if key == "face":
        stable_diffusion = stable_diffusion_face
    if key == "style":
        stable_diffusion = stable_diffusion_style
    if key == "cn":
        # For controlnet with openpose
        args_dict["ip_adapter_image"] = ipimagenone
        stable_diffusion.set_ip_adapter_scale(0.0)


        openpose_image = openpose(ipimage)
        args_dict["image"] = openpose_image
        args_dict["controlnet_conditioning_scale"] = 1.0
    
    else:
        # Everything else (Image generation and Image editing)
        # Conditionally add the ip_adapter_image argument
        if ipimage is not None:
            args_dict["ip_adapter_image"] = ipimage
            stable_diffusion.set_ip_adapter_scale(ipweight)

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

        if key == "cn":
            response_pose = upload_image(openpose_image, upload_url, filename + "_pose", "ai")
            if response_pose.status_code == 200:
                print("Generation and upload successful.", filename + "_pose")

            response_data = {
                "created": int(time.time()),
                "data": [
                    {
                        "url_image": path_url+filename+".jpg",
                        "url_pose": path_url+filename+"_pose"+".jpg",
                    }
                ]
            }

        else:
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

        if key == "cn":
            image_buffer_pose = io.BytesIO()
            openpose_image.save(image_buffer_pose, format='JPEG')
            image_buffer_pose.seek(0)

            response_data = {
                "created": int(time.time()),
                "data": [
                    { 
                        "b64_json_image": base64.b64encode(image_buffer.read()),
                        "b64_json_pose": base64.b64encode(image_buffer_pose.read()),
                    }
                ]       
            }

        else:
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
        "user": form_data.get("user") if form_data.get("user") else None,
        "negprompt": form_data.get("negprompt") if form_data.get("negprompt") else "",
    }

    # Create an instance of CompletionRequest
    request = CompletionRequest(**completion_request_data)

    tensor_image = None
    if imageup:
        w, h = map(int, request.size.split('x'))
        # Ensure w and h do not exceed max_width and max_height
        w = min(w, max_width)
        h = min(h, max_height)
 
        image_data = await imageup.read()
        tensor_image = Image.open(io.BytesIO(image_data))
        tensor_image = tensor_image.resize((w, h))
    else:
        tensor_image = None

    response_data = None
    try:
        response_data = image_request(request.prompt, request.size, request.response_format, request.n, request.negprompt, tensor_image, request.user, request.style)

    
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
