import asyncio
import json
import os
import logging
import time
import configparser
import argparse
from typing import AsyncIterable, List, Generator, Union, Optional
import torch

import requests
import sseclient
import tiktoken

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer, AutoModelForSequenceClassification
from threading import Thread, BoundedSemaphore

logging.basicConfig(level=logging.DEBUG)

class CompletionRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[List[int]]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

class CompletionRequestRerank(BaseModel):
    model: str
    input: Union[str, List[str], List[List[str]]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')

# Parse the arguments
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('bge-large-en-v1.5', 'repo')
host = config.get('settings', 'host')

port = args.port if args.port is not None else config.getint('settings', 'port')

model = AutoModel.from_pretrained(repo_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model.eval()

repo_id = config.get('bge-reranker-large', 'repo')
model_rerank = AutoModelForSequenceClassification.from_pretrained(repo_id).to("cuda")
tokenizer_rerank = AutoTokenizer.from_pretrained(repo_id)
model_rerank.eval()

print("*** Loaded.. now Inference...:")

app = FastAPI(title="BGE-embedding")

def embedding_request(input: Union[str, List[str], List[List[int]]], tempmodel: str = 'BGE'):

    if isinstance(input, list) and all(isinstance(i, list) for i in input):
        enc = tiktoken.get_encoding("cl100k_base")
        input = enc.decode(input[0])

    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to("cuda")
    print(encoded_input)

    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    embedding = sentence_embeddings.cpu().squeeze().numpy().tolist()
    prompt_tokens = sum(len(ids) for ids in encoded_input.input_ids)

    response_data = {
        "object": "list",
        "model": tempmodel,
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": embedding,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            }
        }
    return response_data


def embedding_rerank(input: Union[str, List[str], List[List[str]]], tempmodel: str = 'BGE'):

    encoded_input = tokenizer_rerank(input, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
    print(encoded_input)

    with torch.no_grad():
        scores = model_rerank(**encoded_input, return_dict=True).logits.view(-1, ).float()
    # normalize embeddings
    prompt_tokens = sum(len(ids) for ids in encoded_input.input_ids)

    response_data = {
        "object": "list",
        "model": tempmodel,
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "scores": scores.cpu().squeeze().numpy().tolist(),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            }
        }
    return response_data


@app.post('/v1/embeddings')
async def main(request: CompletionRequest):

    response_data = None
    try:
        response_data = embedding_request(request.input)
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data

@app.post('/v1/embeddings-rerank')
async def main(request: CompletionRequestRerank):

    response_data = None
    try:
        response_data = embedding_rerank(request.input)
    
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

    uvicorn.run(app, host=host, port=port, log_level="debug")
