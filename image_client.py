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
from concurrent.futures import ThreadPoolExecutor

from controlnet_aux import OpenposeDetector

from controlnet_aux.processor import Processor
from collections import defaultdict

thread_pool = ThreadPoolExecutor(max_workers=2)  # Adjust the number of workers as needed
active_requests = 0

class CompletionRequest(BaseModel):
    prompt: str
    n: Optional[int] = 42
    model: Optional[str] = "sd15"
    image: Optional[Union[UploadFile, List[UploadFile]]] = None
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

scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
#scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# Load base pipeline
base_pipeline = DiffusionPipeline.from_pretrained(
    repo_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None
)

# Reuse components for different pipelines
stable_diffusion_style = DiffusionPipeline.from_pipe(base_pipeline)
stable_diffusion_face = DiffusionPipeline.from_pipe(base_pipeline)
stable_diffusion_mix = DiffusionPipeline.from_pipe(base_pipeline)

# ControlNet setup (if needed)
if use_ctrlnet:
    stable_diffusion_ctrl = StableDiffusionXLControlNetPipeline.from_pipe(
        base_pipeline,
        controlnet=controlnet
    )

seed = 42
generator = torch.Generator("cpu").manual_seed(seed)

stable_diffusion_style.load_ip_adapter(adapter_id, subfolder=adapter_folder, weight_name=adapter_encoder)
stable_diffusion_face.load_ip_adapter(adapter_id, subfolder=adapter_folder, weight_name=adapter_encoder_face)
stable_diffusion_mix.load_ip_adapter(adapter_id, subfolder=adapter_folder, weight_name=[adapter_encoder, adapter_encoder_face])

#stable_diffusion.enable_vae_slicing()
#stable_diffusion.enable_sequential_cpu_offload()
#stable_diffusion_style.enable_model_cpu_offload()
#stable_diffusion_face.enable_model_cpu_offload()
#stable_diffusion_mix.enable_model_cpu_offload()
stable_diffusion_style.to("cuda")
stable_diffusion_face.to("cuda")
stable_diffusion_mix.to("cuda")
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


def image_request(prompt: str, size: str, response_format: str, seed: int = 42, negprompt: str = "", ipimage: Union[Image.Image, List[Image.Image]] = None, user: Optional[str] = None, weight: Optional[str] = '0.5', tempmodel: str = 'XL'):
    key = ""
    weight_dict = defaultdict(list)
    image_counts = defaultdict(int)
    try:
        w, h = map(int, size.split('x'))
        # Ensure w and h do not exceed max_width and max_height
        w = min(w, max_width)
        h = min(h, max_height)
        w = max(w, min_width)
        h = max(h, min_height)
        if user is not None:
            user_items = user.split(',')
            for item in user_items:
                key, value = item.split(':')
                weight_dict[key].append(float(value))
                image_counts[key] += 1
    except ValueError:
        #ignore
        return None
    
    keys = ['face', 'style']
    ipweights = [0.0, 0.0]
    for key, values in weight_dict.items():
        if key in keys:
            index = keys.index(key)
            ipweights[index] = sum(values) / len(values)

    generator = torch.Generator("cpu").manual_seed(seed)
    args_dict = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "height": h,
        "width": w,
        "generator": generator,
        "num_inference_steps": 50
    }
    print(ipweights)
    print(keys)
    if isinstance(ipimage, list) and len(ipimage) > 0:
        stable_diffusion = stable_diffusion_mix
        
        # Split images based on the counts in the user string
        face_images = []
        style_images = []
        current_index = 0
        
        for key in keys:
            count = image_counts[key]
            if key == 'face':
                face_images.extend(ipimage[current_index:current_index+count])
            elif key == 'style':
                style_images.extend(ipimage[current_index:current_index+count])
            current_index += count
        
        # If there are no style images, use imagenone
        if not style_images:
            style_images = [ipimagenone]
        if not face_images:
            face_images = [ipimagenone]
        
        args_dict["ip_adapter_image"] = [face_images, style_images]
        
        stable_diffusion.set_ip_adapter_scale(ipweights)
    else:
        if "face" in keys:
            stable_diffusion = stable_diffusion_face
        elif "style" in keys:
            stable_diffusion = stable_diffusion_style
        else:
            stable_diffusion = stable_diffusion_style  # default

        if "cn" in keys:
            # For controlnet with openpose
            args_dict["ip_adapter_image"] = ipimagenone
            stable_diffusion.set_ip_adapter_scale(0.0)

            openpose_image = openpose(ipimage[0] if isinstance(ipimage, list) else ipimage)
            args_dict["image"] = openpose_image
            args_dict["controlnet_conditioning_scale"] = 1.0
        elif ipimage is not None:
            # Everything else (Image generation and Image editing)
            args_dict["ip_adapter_image"] = ipimage[0] if isinstance(ipimage, list) else ipimage
            stable_diffusion.set_ip_adapter_scale(ipweights[0] if ipweights else float(weight))

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
    global active_requests
    active_requests += 1
    response_data = None
    try:
        #response_data = image_request(request.prompt, request.size, request.response_format, request.n, request.negprompt)
        response_data = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            image_request,
            request.prompt,
            request.size,
            request.response_format,
            request.n,
            request.negprompt,
        )

    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}
    
    finally:
        active_requests -= 1
    return response_data

@app.post('/v1/images/edits')
async def edits(inrequest: Request):
    global active_requests
    active_requests += 1
    form_data = await inrequest.form()
    #imageup: UploadFile = form_data.get("image")
    image_uploads = form_data.getlist("image")

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

    #tensor_image = None
    tensor_images = []
    if image_uploads:
        w, h = map(int, request.size.split('x'))
        # Ensure w and h do not exceed max_width and max_height
        w = min(w, max_width)
        h = min(h, max_height)
 
        for image_upload in image_uploads:
            image_data = await image_upload.read()
            tensor_image = Image.open(io.BytesIO(image_data))
            tensor_image = tensor_image.resize((w, h))
            tensor_images.append(tensor_image)
        #image_data = await imageup.read()
        #tensor_image = Image.open(io.BytesIO(image_data))
        #tensor_image = tensor_image.resize((w, h))
    else:
        tensor_images = None

    response_data = None
    try:
        #response_data = image_request(request.prompt, request.size, request.response_format, request.n, request.negprompt, tensor_images, request.user, request.style)
        response_data = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            image_request,
            request.prompt,
            request.size,
            request.response_format,
            request.n,
            request.negprompt,
            tensor_images,
            request.user,
            request.style,
        )
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    finally:
        active_requests -= 1
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
    global active_requests
    return {"active": active_requests}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="error")
