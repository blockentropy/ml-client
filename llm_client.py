import asyncio
import json
import os
import logging
import time
import configparser
from typing import AsyncIterable, List, Generator, Union, Optional

import requests
import sseclient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread, BoundedSemaphore
from auto_gptq import exllama_set_max_input_length

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stop: Optional[str] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    suffix: Optional[str] = None
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('llama70BchatGPTQ', 'repo')
host = config.get('settings', 'host')
port = config.getint('settings', 'port')

# only allow one client at a time
busy = False
condition = asyncio.Condition()

model = AutoModelForCausalLM.from_pretrained(repo_id,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             use_flash_attention_2=True,)

## Only for Llama Models
model = exllama_set_max_input_length(model, 4096)

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

print("*** Loaded.. now Inference...:")

app = FastAPI(title="Llama70B")

#def model_generate(inputs, streamer, max_new_tokens):
#    global busy
#    busy = True
#    model.generate(input_ids=inputs['input_ids'], streamer=streamer, max_new_tokens=max_new_tokens)
#    busy = False


async def streaming_request(prompt: str, max_tokens: int = 100, tempmodel: str = 'Llama70'):
    """Generator for each chunk received from OpenAI as response
    :return: generator object for streaming response from OpenAI
    """
    global busy
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        busy = True
        generated_text += new_text
        reason = None
        if new_text == "</s>":
            reason = "stop"
            new_text = ''
        response_data = {
            "id": "cmpl-0",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "Llama70B",
            "choices": [
                {
                    "index": 0,
                    "text": new_text,  # Changed this line to match OpenAI's format
                    "logprobs": None,
                    "finish_reason": reason
                }
            ]
        }
        json_output = json.dumps(response_data)
        yield f"data: {json_output}\n\n"  # SSE format

    yield 'data: [DONE]'
    busy = False
    async with condition:
        condition.notify_all()

def non_streaming_request(prompt: str, max_tokens: int = 100, tempmodel: str = 'Llama70'):
    # Assume generated_text is the output text you want to return
    # and assume you have a way to calculate prompt_tokens and completion_tokens

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.0,
        #top_p=0.95,
        #top_k=40,
        repetition_penalty=1.1,
    )
    output = pipe(prompt, return_full_text=False)
    generated_text = output[0]['generated_text']
    completion_tokens = max_tokens
    prompt_tokens = max_tokens

    response_data = {
        "id": "cmpl-0",
        "object": "text_completion",
        "created": int(time.time()),
        "model": tempmodel,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"  # Assuming max length
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    return response_data

@app.post('/v1/completions')
async def main(request: CompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        prompt = ""
        if isinstance(request.prompt, list):
            # handle list of strings
            prompt = request.prompt[0]  # just grabbing the 0th index
        else:
            # handle single string
            prompt = request.prompt

        if request.stream:
            response = StreamingResponse(streaming_request(prompt, request.max_tokens), media_type="text/event-stream")
        else:
            response_data = non_streaming_request(prompt, request.max_tokens)
            response = response_data  # This will return a JSON response
    
    except Exception as e:
        # Handle exception...
        async with condition:
            if request.stream == True:
                busy = False
                await condition.notify_all()

    finally:
        async with condition:
            if request.stream != True:
                busy = False
                condition.notify_all()

    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
