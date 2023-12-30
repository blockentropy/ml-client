import asyncio
import json
import os
import logging
import time
import configparser
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


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('bge-large-en-v1.5', 'repo')
host = config.get('settings', 'host')
port = config.getint('settings', 'port')

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
    max_tokens = 10
    completion_tokens = max_tokens
    prompt_tokens = max_tokens

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
            "total_tokens": completion_tokens
            }
        }
    return response_data


def embedding_rerank(input: Union[str, List[str], List[List[str]]], tempmodel: str = 'BGE'):

    encoded_input = tokenizer_rerank(input, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
    print(encoded_input)

    with torch.no_grad():
        scores = model_rerank(**encoded_input, return_dict=True).logits.view(-1, ).float()
    # normalize embeddings
    max_tokens = 10
    completion_tokens = max_tokens
    prompt_tokens = max_tokens

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
            "total_tokens": completion_tokens
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

@app.get('/ping')
async def get_status():
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
