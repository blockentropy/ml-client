import asyncio
import json
import os
import io
import logging
import time
import configparser
from typing import AsyncIterable, List, Generator, Union, Optional
import torch

import requests
import sseclient
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

logging.basicConfig(level=logging.DEBUG)

class CompletionRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('stable-diffusion-xl-base-1.0', 'repo')
host = config.get('settings', 'host')
port = config.getint('settings', 'port')
upload_url = config.get('settings', 'upload_url')

scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", safety_checker=None)
seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed)
stable_diffusion.to(f"cuda")

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

def image_request(prompt: str, tempmodel: str = 'XL'):

    negprompt = ""
    image = stable_diffusion(prompt, negative_prompt=negprompt, generator=generator, num_inference_steps=20).images[0]

    random_string = str(random.randint(100000, 999999))
    filename = "ai_seed"+str(seed)+"_"+random_string

    response = upload_image(image, upload_url,filename,"ai")
    if response.status_code == 200:
        print("Generation and upload successful.", filename)

    response_data = {
        "created": int(time.time()),
        "data": [
            {
                "url": "https://blockentropy.dev/uploads/ai/"+filename+"_full.jpg",
            }
        ]
        }
    return response_data

@app.post('/v1/image/generations')
async def main(request: CompletionRequest):

    response_data = None
    try:
        response_data = image_request(request.prompt)
    
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
