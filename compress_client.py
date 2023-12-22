import asyncio
import json
import os
import logging
import time
import configparser
import argparse
from typing import AsyncIterable, List, Generator, Union, Optional
import torch
import numpy as np

import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread, BoundedSemaphore
from llmlingua import PromptCompressor

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')

# Parse the arguments
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

class CompletionRequest(BaseModel):
    model: Optional[str] = "Starling-LM-7B-alpha"
    prompt: Union[str, List[str], List[List[str]]]
    user: Optional[str] = None

class CompletionRequestLong(BaseModel):
    model: Optional[str] = "Starling-LM-7B-alpha"
    prompt: Union[str, List[str], List[List[str]]]
    question: Union[str, List[str], List[List[str]]]
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('Starling-LM-7B-alpha', 'repo')
host = config.get('settings', 'host')
port = args.port if args.port is not None else config.getint('settings', 'port')

llm_lingua = PromptCompressor(repo_id, model_config={"revision": "main"})

print("*** Loaded.. now Rank...:")

app = FastAPI(title="Compressor-Rerank")

@app.post('/v1/compress')
async def main(request: CompletionRequest):

    response_data = None
    try:
        print(request.prompt)
        
        compressed_prompt = llm_lingua.compress_prompt(request.prompt, instruction="", question="", target_token=200)
        print(compressed_prompt)
        response_data = compressed_prompt
    
    except Exception as e:
        # Handle exception...
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}

    return response_data

@app.post('/v1/compresslong')
async def main(request: CompletionRequestLong):

    response_data = None
    try:
        print(request.prompt)
        print(request.question)
        
        compressed_prompt = llm_lingua.compress_prompt(request.prompt, instruction="", question=request.question, rank_method="longllmlingua",reorder_context="sort", dynamic_context_compression_ratio=0.2, condition_compare=True, context_budget="+100", token_budget_ratio=1.05)
        print(compressed_prompt)
        response_data = compressed_prompt
    
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
