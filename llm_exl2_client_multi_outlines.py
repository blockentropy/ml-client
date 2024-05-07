import asyncio
import json
import os
import logging
import time
import configparser
import argparse
import tiktoken
import torch
import random
from typing import AsyncIterable, List, Generator, Union, Optional
import traceback
from typing import Mapping
import requests
import sseclient
import subprocess
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread
import queue
import numpy as np

import sys, os
import outlines
from outlines.samplers import multinomial
from exl2_outlines import exl2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)
import uuid

def generate_unique_id():
    return uuid.uuid4()

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    suffix: Optional[str] = None
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None
    stop_at: Optional[str] = None
    outlines_type: Optional[str] = None
    choices: Optional[list[str]] = None
    regex: Optional[str] = None
    json: Optional[str] = None

#repo_str = 'theprofessor-exl2-speculative'

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')
parser.add_argument('--use_outlines', action='store_true', help='Use outlines.')
parser.add_argument('--gpu_split', type=str, default="17,19,19,19", help='GPU splits.')
parser.add_argument('--max_context', type=int, default=12288, help='Context length.')
parser.add_argument('--cache_8bit', action='store_true', help='Use 8 bit cache.')
parser.add_argument('--cache_q4', action='store_true', help='Use 4 bit cache.')
parser.add_argument('--repo_str', type=str, default='llama3-70b-instruct', help='The model repository name')
parser.add_argument('--outlines_device', type=int, default=2, help='The cuda device to which the outlines device is set')

# Parse the arguments
args = parser.parse_args()
repo_str = args.repo_str

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
host = config.get('settings', 'host')

port = args.port if args.port is not None else config.getint('settings', 'port')

# only allow one client at a time
busy = False
condition = asyncio.Condition()

config = ExLlamaV2Config()
config.model_dir = repo_id
config.prepare()

use_dynamic_rope_scaling = False 
dynamic_rope_mult = 1.5
dynamic_rope_offset = 0.0

ropescale = 1.0
max_context = args.max_context
config.scale_alpha_value = ropescale
config.max_seq_len = max_context
base_model_native_max = 8192
cache_8bit = args.cache_8bit
cache_q4 = args.cache_q4

if args.use_outlines:
    model = exl2(
        config.model_dir,
        f"cuda:{args.outlines_device}",
        max_seq_len = config.max_seq_len,
        scale_pos_emb = config.scale_pos_emb,
        scale_alpha_value = config.scale_alpha_value,
        no_flash_attn = config.no_flash_attn,
        num_experts_per_token = config.num_experts_per_token,
        cache_8bit = cache_8bit,
        cache_q4 = cache_q4,
        tokenizer_kwargs = {},
        gpu_split = args.gpu_split, # we might be able to make this auto
        low_mem = None,
        verbose = None
    )
else:
    model = ExLlamaV2(config)
print("Loading model: " + repo_id)
#cache = ExLlamaV2Cache(model, lazy=True, max_seq_len = 20480)
#model.load_autosplit(cache)
if not args.use_outlines:
    model.load([int(gpu_memory) for gpu_memory in args.gpu_split.split(",")])

tokenizer = ExLlamaV2Tokenizer(config)

# Cache mode


settings_proto = ExLlamaV2Sampler.Settings()
settings_proto.temperature = 0
settings_proto.top_k = 50
settings_proto.top_p = 0.8
settings_proto.top_a = 0.0
settings_proto.token_repetition_penalty = 1.1
#settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

# Active sequences and corresponding caches and settings
prompts = queue.Queue()
responses = {}

input_ids = []
prompt_length = []
prompt_ids = []
streamer = []
caches = []
input_prompts = []
generators = []
generations = []
settings = []
future_tokens = []
future_logits = []
sin_arr = []
cos_arr = []

# Global variable for storing partial responses
partial_responses = {}

max_parallel_seqs = 3 
num_of_gpus = len(args.gpu_split.split(","))

print("*** Loaded.. now Inference...:")

app = FastAPI(title="EXL2")

async def stream_response(prompt_id, timeout=180):
    global partial_responses
    while True:
        await asyncio.sleep(0.05)  # Sleep to yield control to the event loop

        # Check if prompt_id exists in partial_responses
        if prompt_id in partial_responses:
            # Stream partial responses
            while partial_responses[prompt_id]:
                response_chunk = partial_responses[prompt_id].pop(0)
                yield f"data: {json.dumps(response_chunk)}\n\n"

            # Check for final response or timeout
            if prompt_id in responses:
                final_response = responses.pop(prompt_id)
                yield f'data: {{"id":"chatcmpl-{prompt_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{repo_str}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n'
                break

# Worker thread function
def process_outline_prompts():
    global partial_responses
    assert args.use_outlines
    assert not use_dynamic_rope_scaling, "Currently ROPE scaling is not supported with outlines"
    base_model = model.model
    while True:
        while not prompts.empty() or len(prompt_ids):
            while len(prompt_ids) < max_parallel_seqs and not prompts.empty():
                prompt_id, prompt, max_tokens, stream, temperature, outlines_dict = prompts.get()
                print(f"got prompt with outlines dict {outlines_dict}")
                sampler = multinomial(top_k=50, top_p=1.0, temperature=temperature)
                ids = tokenizer.encode(prompt)
                prompt_tokens = ids.shape[-1]
                max_tokens=min(max_tokens, max_context-prompt_tokens)
                full_tokens = prompt_tokens + max_tokens
                print("Processing prompt: " + str(prompt_id) + "  Req tokens: " + str(full_tokens))
                # Truncate if new_tokens exceed max_context
                if full_tokens > max_context:
                    # Calculate how many tokens to truncate
                    ids = tokenizer.encode("Say, 'Prompt exceeds allowed length. Please try again.'")
                    # Update new_tokens after truncation
                    prompt_tokens = ids.shape[-1]
                    full_tokens = prompt_tokens + max_tokens
                    print("Truncating prompt: " + str(prompt_id) + "  Req tokens: " + str(full_tokens))
                if cache_8bit:
                    ncache = ExLlamaV2Cache_8bit(base_model, lazy=not base_model.loaded, max_seq_len = full_tokens)  # (max_seq_len could be different for each cache)
                elif cache_q4:
                    ncache = ExLlamaV2Cache_Q4(base_model, lazy=not base_model.loaded, max_seq_len = full_tokens)
                else:
                    ncache = ExLlamaV2Cache(base_model, lazy=not base_model.loaded, max_seq_len = full_tokens)  # (max_seq_len could be different for each cache)
                model.cache = ncache
                model.past_seq = None
                stop_at = outlines_dict.get("stop_at", None)
                if outlines_dict["type"] == "choices":
                    generator = outlines.generate.choice(model, outlines_dict["choices"], sampler=sampler)
                elif outlines_dict["type"] == "json":
                    generator = outlines.generate.json(model, outlines_dict["json"], sampler=sampler)
                elif outlines_dict["type"] == "regex":
                    generator = outlines.generate.regex(model, outlines_dict["regex"], sampler=sampler)
                else:
                    generator = outlines.generate.text(model, sampler=sampler)
                generators.append(generator.stream(prompt, stop_at=stop_at, max_tokens=max_tokens))
                prompt_ids.append(prompt_id)
                input_prompts.append(prompt)
                generations.append("")
                caches.append(ncache)
                streamer.append(stream)
            if(len(prompt_ids)):
                eos = []
                for i in range(len(prompt_ids)):
                    model.cache = caches[i]
                    is_finished = False
                    try:
                        decoded_response_token = next(generators[i])
                        generations[i] += decoded_response_token
                    except StopIteration:
                        is_finished = True
                    reason = None
                    if(streamer[i]):
                        outcontent = decoded_response_token
                        if is_finished:
                            outcontent = ""
                            reason = "stop"
                        partial_response_data = {
                            "id": f"chatcmpl-{prompt_ids[i]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": repo_str,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": outcontent
                                    },
                                    "finish_reason": reason
                                }
                            ]
                        }

                        # Initialize a list for new prompt_id or append to existing one
                        if prompt_ids[i] not in partial_responses:
                            partial_responses[prompt_ids[i]] = []
                        partial_responses[prompt_ids[i]].append(partial_response_data)

                    if is_finished:
                        eos.insert(0, i)

                # Generate and store response
                for i in eos:
                    output = generations[i].strip()
                    prompt = input_prompts[i]
                    #output = tokenizer.decode(input_ids[i])[0]
                    print("-----")
                    print(output)
                    generated_text = output
                    # Calculate token counts
                    completion_tokens = (tokenizer.encode(generated_text)).shape[-1]
                    prompt_tokens = (tokenizer.encode(prompt)).shape[-1]
                    full_tokens = completion_tokens + prompt_tokens
                    eos_prompt_id = prompt_ids.pop(i)
                    if(streamer[i]):
                        ## Generator, yield here..
                            partial_response_data = {
                                "finish_reason": "stop"
                            }

                            responses[eos_prompt_id] = partial_response_data
                    else:# Construct the response based on the format
                        response_data = {
                            "id": f"chatcmpl-{prompt_id}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": repo_str,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": generated_text,
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": full_tokens
                            }
                        }
                        responses[eos_prompt_id] = response_data
                    # Clean up
                    generators.pop(i)
                    input_prompts.pop(i)
                    generations.pop(i)
                    caches.pop(i)
                    streamer.pop(i)

        else:
            # Sleep for a short duration when there's no work
            time.sleep(0.1)  # Sleep for 100 milliseconds



# Start worker thread
worker = Thread(target=process_outline_prompts)
worker.start()


async def format_prompt(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"{message.content}\n\n"
        elif message.role == "user":
            formatted_prompt += f"### User:\n{message.content}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"### Assistant:\n{message.content}\n\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "### Assistant:\n"
    return formatted_prompt

async def format_prompt_llama3(messages):
    formatted_prompt = ""
    system_message_found = False

    # Check for a system message first
    for message in messages:
        if message.role == "system":
            system_message_found = True
            break

    # If no system message was found, prepend a default one
    if not system_message_found:
        formatted_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>"
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{message.content}<|eot_id|>"
        elif message.role == "user":
            formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"
        elif message.role == "assistant":
            formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_prompt

async def format_prompt_yi(messages):
    formatted_prompt = ""
    system_message_found = False
    
    # Check for a system message first
    for message in messages:
        if message.role == "system":
            system_message_found = True
            break
    
    # If no system message was found, prepend a default one
    if not system_message_found:
        formatted_prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            formatted_prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt

async def format_prompt_nous(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"{message.content}\n"
        elif message.role == "user":
            formatted_prompt += f"USER: {message.content}\n"
        elif message.role == "assistant":
            formatted_prompt += f"ASSISTANT: {message.content}\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "ASSISTANT: "
    return formatted_prompt

async def format_prompt_tess(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"SYSTEM: {message.content}\n"
        elif message.role == "user":
            formatted_prompt += f"USER: {message.content}\n"
        elif message.role == "assistant":
            formatted_prompt += f"ASSISTANT: {message.content}\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "ASSISTANT: "
    return formatted_prompt

async def format_prompt_code(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"### System Prompt\nYou are an intelligent programming assistant.\n\n"
        elif message.role == "user":
            formatted_prompt += f"### User Message\n{message.content}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"### Assistant\n{message.content}\n\n"
    # Add the final "### Assistant" with ellipsis to prompt for the next response
    formatted_prompt += "### Assistant\n..."
    return formatted_prompt

async def format_prompt_zephyr(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            formatted_prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|assistant|>\n{message.content}</s>\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|assistant|>\n"
    return formatted_prompt

async def format_prompt_starling(messages):
    formatted_prompt = ""
    system_message = ""
    for message in messages:
        if message.role == "system":
            # Save system message to prepend to the first user message
            system_message += f"{message.content}\n\n"
        elif message.role == "user":
            # Prepend system message if it exists
            if system_message:
                formatted_prompt += f"GPT4 Correct User: {system_message}{message.content}<|end_of_turn|>"
                system_message = ""  # Clear system message after prepending
            else:
                formatted_prompt += f"GPT4 Correct User: {message.content}<|end_of_turn|>"
        elif message.role == "assistant":
            formatted_prompt += f"GPT4 Correct Assistant: {message.content}<|end_of_turn|>"  # Prep for user follow-up
    formatted_prompt += "GPT4 Correct Assistant: \n\n"
    return formatted_prompt

async def format_prompt_mixtral(messages):
    formatted_prompt = "<s> "
    system_message = ""
    for message in messages:
        if message.role == "system":
            # Save system message to prepend to the first user message
            system_message += f"{message.content}\n\n"
        elif message.role == "user":
            # Prepend system message if it exists
            if system_message:
                formatted_prompt += f"[INST] {system_message}{message.content} [/INST] "
                system_message = ""  # Clear system message after prepending
            else:
                formatted_prompt += f"[INST] {message.content} [/INST] "
        elif message.role == "assistant":
            formatted_prompt += f" {message.content}</s> "  # Prep for user follow-up
    return formatted_prompt


async def format_prompt_commandr(messages):
    formatted_prompt = ""
    system_message_found = False
    
    # Check for a system message first
    for message in messages:
        if message.role == "system":
            system_message_found = True
            break
    
    # If no system message was found, prepend a default one
    if not system_message_found:
        formatted_prompt += f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
 
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
        elif message.role == "user":
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
        elif message.role == "assistant":
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    return formatted_prompt


@app.post('/v1/chat/completions')
async def mainchat(request: ChatCompletionRequest):
    try:
        prompt = ''
        if repo_str == 'Phind-CodeLlama-34B-v2':
            prompt = await format_prompt_code(request.messages)
        elif repo_str == 'zephyr-7b-beta':
            prompt = await format_prompt_zephyr(request.messages)
        elif repo_str == 'llama3-70b-instruct':
            prompt = await format_prompt_llama3(request.messages)
        elif repo_str == 'Starling-LM-7B-alpha':
            prompt = await format_prompt_starling(request.messages)
        elif repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ':
            prompt = await format_prompt_mixtral(request.messages)
        elif repo_str == 'Yi-34B-Chat-GPTQ' or repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ' or repo_str == 'theprofessor-exl2-speculative' or repo_str == 'dbrx-instruct-exl2':
            prompt = await format_prompt_yi(request.messages)
        elif repo_str == 'Nous-Capybara-34B-GPTQ' or repo_str == 'goliath-120b-GPTQ' or repo_str == 'goliath-120b-exl2' or repo_str == 'goliath-120b-exl2-rpcal':
            prompt = await format_prompt_nous(request.messages)
        elif repo_str == 'tess-xl-exl2' or repo_str == 'tess-xl-exl2-speculative':
            prompt = await format_prompt_tess(request.messages)
        elif repo_str == 'commandr-exl2' or repo_str == 'commandr-exl2-speculative':
            prompt = await format_prompt_commandr(request.messages)
        else:
            prompt = await format_prompt(request.messages)
        print(prompt)

        timeout = 180  # seconds
        start_time = time.time()
        prompt_id = generate_unique_id() # Replace with a function to generate unique IDs
        outlines_dict = {}
        if request.stop_at is not None:
            outlines_dict["stop_at"] = request.stop_at
        if request.outlines_type is not None:
            outlines_dict["type"] = request.outlines_type
        elif args.use_outlines:
            outlines_dict["type"] = "text"
        else:
            raise Exception("Enable outlines")
        if outlines_dict["type"] == "choices":
            assert request.choices is not None
            outlines_dict["choices"] = request.choices
        elif outlines_dict["type"] == "json":
            assert request.json is not None
            outlines_dict["json"] = request.json
        elif outlines_dict["type"] == "regex":
            assert request.regex is not None
            outlines_dict["regex"] = request.regex
        else:
            assert (outlines_dict["type"] == "text") or not args.outlines
        if not args.use_outlines:
            prompts.put((prompt_id, prompt, request.max_tokens, request.stream, request.temperature))
        else:
            prompts.put((prompt_id, prompt, request.max_tokens, request.stream, request.temperature, outlines_dict))


        if request.stream:
            #response = StreamingResponse(streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion'), media_type="text/event-stream")
            return StreamingResponse(stream_response(prompt_id), media_type="text/event-stream")
        else:
            #response_data = non_streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion')
            #response = response_data  # This will return a JSON response
            while prompt_id not in responses:
                await asyncio.sleep(0.1)  # Sleep to yield control to the event loop
                if time.time() - start_time > timeout:
                    return {"error": "Response timeout"} 

            return responses.pop(prompt_id)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return response




@app.get('/ping')
async def get_status():
    return {"ping": sum(prompt_length)}

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
