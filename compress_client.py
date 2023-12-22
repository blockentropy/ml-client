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
import llm_blender

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')

# Parse the arguments
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)

class CompletionRequest(BaseModel):
    model: Optional[str] = "PairRM"
    input: Union[str, List[str], List[List[str]]]
    output: Union[str, List[str], List[List[str]]]
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('PairRM', 'repo')
host = config.get('settings', 'host')
port = args.port if args.port is not None else config.getint('settings', 'port')

blender = llm_blender.Blender()
blender.loadranker(repo_id) # load ranker checkpoint

print("*** Loaded.. now Rank...:")

app = FastAPI(title="Blender-Rerank")

@app.post('/v1/rank')
async def main(request: CompletionRequest):

    response_data = None
    try:
        print(request.input)
        print(request.output)
        ranks = blender.rank(request.input, request.output, return_scores=False, batch_size=1)
        response_data = {
            "rank": ranks.tolist() if isinstance(ranks, np.ndarray) else ranks
        }
    
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
