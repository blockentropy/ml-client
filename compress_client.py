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
    model: Optional[str] = "Yi"
    input: Union[str, List[str], List[List[str]]]
    output: Union[str, List[str], List[List[str]]]
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

#repo_id = config.get('PairRM', 'repo')
host = config.get('settings', 'host')
port = args.port if args.port is not None else config.getint('settings', 'port')

llm_lingua = PromptCompressor()

print("*** Loaded.. now Rank...:")

app = FastAPI(title="Compressor-Rerank")

@app.post('/v1/compress')
async def main(request: CompletionRequest):

    response_data = None
    try:
        print(request.input)
        print(request.output)

        
        compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", question="", target_token=200)
        print(compressed_prompt)
    
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
