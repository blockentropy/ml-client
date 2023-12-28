######## Monkey Patch TQDM for updates
from tqdm.auto import tqdm as tqdm_base
import threading

class LoggingTqdm(tqdm_base):
    thread_local = threading.local()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = getattr(self.thread_local, 'url', None)
        self.signer = getattr(self.thread_local, 'signer', None)

    def update(self, n=1):
        super().update(n)
        if self.url:
            elapsed_time = time.time() - self.start_t  # Total elapsed time in seconds

            if self.n > 0 and elapsed_time > 0:  # Check if elapsed_time is greater than 0 to avoid division by zero
                rate = self.n / elapsed_time  # Compute the rate of progress
                remaining = (self.total - self.n) / rate if self.total else 0  # Estimate remaining time
                remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(remaining)) if self.total else "N/A"
            else:
                remaining_formatted = "?"

            elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            message = f"Generating Video: Processing"
            logStatus(self.url, self.signer, message, elapsed_formatted, remaining_formatted, self.n, self.total)


# Monkey patching tqdm
import tqdm.auto
tqdm.auto.tqdm = LoggingTqdm
from diffusers.utils import logging as diffusers_logging
########


import asyncio
import json
import os
import io
import logging
import time
import configparser
import argparse
from typing import AsyncIterable, List, Generator, Union, Optional
import torch
import sys
import cv2
import glob
import numpy as np

import requests
import sseclient
import random
from io import BytesIO
import imageio.v2 as imageio
import os.path as osp
from warnings import warn
from omegaconf import OmegaConf
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from starlette.datastructures import FormData
from PIL import Image

from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from pydantic import BaseModel
from diffusers import DiffusionPipeline, UniPCMultistepScheduler, MotionAdapter, AnimateDiffPipeline
from diffusers.models import AutoencoderKL
from diffusers.utils import export_to_gif
from diffusers.utils import load_image

#logging.basicConfig(level=logging.DEBUG)

def logStatus(url, signer, message, elapsed, remaining, currentStep, totalSteps):
    data = {
        "signer": signer,  
        "data": {
            "message": message,
            "elapsed": elapsed,
            "remaining": remaining,
            "currentStep": currentStep,
            "totalSteps": totalSteps
        }
    }

        # Make the POST request
    try:
        response = requests.post(url, json=data)
        print(data)
        print(url)
        if response.status_code != 200:
            print(f"Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred while sending data: {e}")


class CompletionRequest(BaseModel):
    prompt: str
    image: Optional[UploadFile] = None
    n: Optional[int] = 42
    model: Optional[str] = "be-anim-realvision"
    response_format: Optional[str] = "url"
    quality: Optional[str] = "smooth"
    style: Optional[str] = "0.6"
    size: Optional[str] = "512x512"
    user: Optional[str] = None


#class ServerLogHandler(logging.Handler):
#    def __init__(self, server_url, signer):
#        super().__init__()
#        self.server_url = server_url
#        self.signer = signer

#    def emit(self, record):
#        log_entry = self.format(record)
#        print(log_entry, self.server_url,self.signer)
#        print(log_entry)


parser = argparse.ArgumentParser(description='Run model on specified GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('--id', type=str, default='default', help='ML id to use')
parser.add_argument('--port', type=int, default='12350', help='ML id to use')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

amt_id = config.get('AMT','repo')
sys.path.append(amt_id)
from utils.utils import (
    read, write,
    img2tensor, tensor2img,
    check_dim_and_resize
)
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder

ckpt_path = amt_id+"/pretrained/amt-s.pth"
cfg_path = amt_id+"/cfgs/AMT-S.yaml"
network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
print(f'Loading [{network_name}] from [{ckpt_path}]...')
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
#model = model.to(dtype=torch.float16, device=f"cuda:{args.gpu}")
model = model.to(f"cuda:{args.gpu}")
model.eval()


repo_id = config.get('be-anim-realvision', 'repo')
adapter_id = config.get('adapter', 'repo')
ip_adapter_id = config.get('ip_adapter', 'repo')
host = config.get('settings', 'host')
port = config.getint('settings', 'port')
upload_url = config.get('settings', 'upload_url')
logging_url = config.get('settings', 'logging_url')
mlid = args.id

scheduler = UniPCMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(repo_id+'/vae')
adapter = MotionAdapter.from_pretrained(adapter_id)

stable_diffusion = AnimateDiffPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, scheduler=scheduler, motion_adapter=adapter, vae=vae)
seed = 42
generator = torch.Generator("cpu").manual_seed(seed)
stable_diffusion.to(f"cuda:{args.gpu}")
stable_diffusion.enable_vae_slicing()
#stable_diffusion.enable_model_cpu_offload()

print("*** Loaded.. now Inference...:")

app = FastAPI(title="Animdiff")

def upload_video(video, upload_url, filename, wallet_address):

    files = {'file': (filename+'.mp4', video, 'video/mp4')}
    response = requests.post(upload_url, files=files)

    if response.status_code == 200:
        print(f'Successfully uploaded image')
        return response
    else:
        print(f'Failed to upload image. Status code: {response.status_code}')
        return None


def upload_image(frames, upload_url, filename, wallet_address):
    buffer = BytesIO()

    # Save the frames as a GIF to the buffer
    frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )

    # Rewind the buffer to the start
    buffer.seek(0)

    files = {'file': (filename+'.gif', buffer, 'image/gif')}
    response = requests.post(upload_url, files=files)

    if response.status_code == 200:
        print(f'Successfully uploaded image')
        return response
    else:
        print(f'Failed to upload image. Status code: {response.status_code}')
        return None

def image_request(prompt: str, negprompt: str, image, tempmodel: str = 'XL'):

    diffusers_logging.set_verbosity_error()
    server_url = logging_url
    #handler = ServerLogHandler(server_url, "signer1")
    #logger = diffusers_logging.get_logger("diffusers")
    #logger.addHandler(handler)
    LoggingTqdm.thread_local.url = server_url
    LoggingTqdm.thread_local.signer = mlid
    logStatus(server_url, mlid, "Starting Video Generation....", 0, 0, 0, 0)

    stable_diffusion.load_lora_weights("/home/ed/Documents/huggingface/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
    #stable_diffusion.model.half()

    stable_diffusion.set_adapters(["zoom-out"], adapter_weights=[0.0])

    # Conditionally add the ip_adapter_image argument
    if image is not None:
        stable_diffusion.set_ip_adapter_scale(0.5)
    else:
        image = load_image("0000.png")
        stable_diffusion.set_ip_adapter_scale(0.0)


    output = stable_diffusion(
        prompt=(prompt),
        #negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        negative_prompt=negprompt,
        #negative_prompt="",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        ip_adapter_image=image,
        generator = generator,
    )
    frames = output.frames[0]
    inputs = []
    padder = InputPadder((512, 512))
    for frame in frames:
        np_frame = np.array(frame) 
        #frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
        frame_t = img2tensor(np_frame).to(f"cuda:{args.gpu}")
        frame_t = padder.pad(frame_t)
        inputs.append(frame_t)

    iters = 2
    scale = 1
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(f"cuda:{args.gpu}")
    print(f'Start frame interpolation:')
    total_frames = (iters + 1) * (iters + 2) / 2
    for i in range(iters):
        print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
        outputs = [inputs[0]]
        for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
            in_0 = in_0.to(f"cuda:{args.gpu}")
            in_1 = in_1.to(f"cuda:{args.gpu}")
            with torch.no_grad():
                imgt_pred = model(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']
            outputs += [imgt_pred.cpu(), in_1.cpu()]
        inputs = outputs

    outputs = padder.unpad(*outputs)
    # Create an in-memory buffer
    in_memory_vid = io.BytesIO()
    frame_rate = 24
    size = outputs[0].shape[2:][::-1]
    out_path = "./"
    #video_count = len(glob.glob(os.path.join(out_path, "*.mp4")))

    save_video_path = 'tempvid.mp4'
    writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"avc1"), 
                        frame_rate, size)
    for i, imgt_pred in enumerate(outputs):
        imgt_pred = tensor2img(imgt_pred)
        imgt_pred = cv2.cvtColor(imgt_pred, cv2.COLOR_RGB2BGR)
        writer.write(imgt_pred)        
    #print(f"Demo video is saved to [{save_video_path}]")
    # Finalize the video file in memory
    writer.release()

    random_string = str(random.randint(100000, 999999))
    filename = "ai_seed"+str(seed)+"_"+random_string
    response = upload_image(frames, upload_url,filename,"ai")
    with open(save_video_path, 'rb') as video_file:
        response2 = upload_video(video_file, upload_url,filename,"ai")
    if response.status_code == 200:
        print("Generation and upload successful.", filename)
        logStatus(server_url, mlid, "Generation and upload successful.", 0, 0, 0, 0)

    response_data = {
        "created": int(time.time()),
        "data": [
            {
                "url": "https://labs.blockentropy.ai/uploads/"+filename+".gif",
                "video": "https://labs.blockentropy.ai/uploads/"+filename+".mp4",
            }
        ]
        }
    del frame_t
    # Optionally, delete other tensors if they are no longer needed
    # del other_tensor

    # Clear CUDA cache
    torch.cuda.empty_cache()
    return response_data

@app.post('/v1/images/edits')
async def main(inrequest: Request):
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
    #request.model="be-anim-realvision"
    global stable_diffusion, generator
    response_data = None
    try:
        repo_id = config.get(request.model, 'repo')
        negprompt = config.get(request.model, 'negprompt')
        vae = AutoencoderKL.from_pretrained(repo_id+'/vae')
        stable_diffusion = AnimateDiffPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, scheduler=scheduler, motion_adapter=adapter, vae=vae)
        seed = int(request.n)
        #stable_diffusion.to(f"cuda:{args.gpu}")
        stable_diffusion.enable_vae_slicing()
        #stable_diffusion.enable_model_cpu_offload()
        generator = torch.Generator("cpu").manual_seed(seed)
        stable_diffusion.load_ip_adapter(ip_adapter_id, subfolder='models', weight_name="ip-adapter_sd15.bin")
        stable_diffusion.to(torch_dtype=torch.float16, device=f"cuda:{args.gpu}")
        #image = load_image(request.image)
        tensor_image = None
        if imageup:
            image_data = await imageup.read()
            tensor_image = Image.open(io.BytesIO(image_data))
            tensor_image = tensor_image.resize((512, 512))
        else:
            tensor_image = None
        response_data = image_request(request.prompt, negprompt, tensor_image)
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data

@app.post('/v1/images/generations')
async def maingen(request: CompletionRequest):

    global stable_diffusion, generator
    response_data = None
    try:
        repo_id = config.get(request.model, 'repo')
        negprompt = config.get(request.model, 'negprompt')
        vae = AutoencoderKL.from_pretrained(repo_id+'/vae')
        stable_diffusion = AnimateDiffPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, scheduler=scheduler, motion_adapter=adapter, vae=vae)
        seed = int(request.n)
        #stable_diffusion.to(f"cuda:{args.gpu}")
        stable_diffusion.enable_vae_slicing()
        #stable_diffusion.enable_model_cpu_offload()
        generator = torch.Generator("cpu").manual_seed(seed)
        stable_diffusion.load_ip_adapter(ip_adapter_id, subfolder='models', weight_name="ip-adapter_sd15.bin")
        stable_diffusion.to(torch_dtype=torch.float16, device=f"cuda:{args.gpu}")
        #image = load_image(request.image)
        tensor_image = None
        response_data = image_request(request.prompt, negprompt, tensor_image)
    
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

    uvicorn.run(app, host=host, port=args.port, log_level="debug")
