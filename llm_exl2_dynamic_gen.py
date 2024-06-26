import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from blessed import Terminal
import pprint

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

import requests
import sseclient
import subprocess
import textwrap


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import threading
from threading import Thread
import queue
import uvicorn
from io import StringIO

def generate_unique_id():
    return uuid.uuid4()

# This is a demo and small stress to showcase some of the features of the dynamic batching generator.
repo_str = 'llama3-70b-instruct-speculative'

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

class StatusArea:
    def __init__(self, num_lines):
        self.num_lines = num_lines
        self.messages = [""] * num_lines

    def update(self, message, line=None):
        if line is not None:
            # Update a specific line
            if 0 <= line < self.num_lines:
                self.messages[line] = message
        else:
            # Handle multi-line message
            lines = message.split('\n')
            if len(lines) > self.num_lines:
                # Truncate to last num_lines if exceeds num_lines
                lines = lines[-self.num_lines:]
            
            # Update messages, padding with empty strings if necessary
            self.messages = lines + [""] * (self.num_lines - len(lines))

        self.display()

    def display(self):
        for i, message in enumerate(self.messages):
            wrapped_message = textwrap.shorten(message, width=term.width, placeholder="...")
            print(term.move_xy(0, i) + term.clear_eol + wrapped_message)
        
        # Move cursor below the status area
        print(term.move_xy(0, self.num_lines), end='', flush=True)


class JobStatusDisplay:

    def __init__(self, job, status_lines):
        #self.console_line = console_line + status_lines
        self.console_line = None
        self.job = job
        self.prefill = 0
        self.max_prefill = 0
        self.collected_output = ""
        self.tokens = 0
        self.spaces = " " * 150
        self.status_lines = status_lines
        self.display_text = ""
        #text = term.black(f"{self.console_line:3}:")
        #text += term.blue("enqueued")
        #print(term.move_xy(0, self.console_line) + text)

    def update_position(self, index):
        self.console_line = self.status_lines + index
        self.init_display_text()

    def init_display_text(self):
        self.display_text = term.black(f"{self.console_line:3}:") + term.blue("enqueued")


    def update(self, r):
        if self.console_line is None:
            return  # Skip update if position hasn't been set yet
        stage = r["stage"]
        stage = r.get("eos_reason", stage)

        self.collected_output += r.get("text", "").replace("\n", "\\n")

        token_ids = r.get("token_ids", None)
        if token_ids is not None: self.tokens += token_ids.shape[-1]

        self.prefill = r.get("curr_progress", self.prefill)
        self.max_prefill = r.get("max_progress", self.max_prefill)

        text = term.black(f"{self.console_line:3}:")
        text += term.blue(f"{stage:16}")
        text += "prefill [ " + term.yellow(f"{self.prefill: 5} / {self.max_prefill: 5}")+" ]"
        text += "   "
        text += term.green(f"{self.tokens: 5} t")
        text += term.black(" -> ")
        text += (self.spaces + self.collected_output)[-150:].replace("\t", " ")

        if "accepted_draft_tokens" in r:
            acc = r["accepted_draft_tokens"]
            rej = r["rejected_draft_tokens"]
            eff = acc / (acc + rej) * 100.0
            text += term.bright_magenta(f"   SD eff.: {eff:6.2f}%")

        #print(term.move_xy(0, self.console_line) + text)
        self.display_text = text

    def display(self):
        if self.console_line is not None:
            print(term.move_xy(0, self.console_line) + self.display_text)


parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')

# Parse the arguments
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
specrepo_id = config.get(repo_str, 'specrepo')
host = config.get('settings', 'host')

port = args.port if args.port is not None else config.getint('settings', 'port')
# Display modes for this demo:
# 1: One line per job, updated continuously
# 2: Print completions as jobs finish
# 3: Step over output iteration by iteration
# 4: Space heater mode (no output)
display_mode = 1

# Whether to use paged mode or not. The generator is very handicapped in unpaged mode, does not support batching
# or CFG, but it will work without flash-attn 2.5.7+
paged = True

# Where to find our model
model_dir = repo_id

# Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but
# the total to distribute dynamically over however many jobs are active at once
total_context = 32768

# Max individual context
max_context = 8192

# N-gram or draft model speculative decoding. Largely detrimental to performance at higher batch sizes.
use_ngram = False
use_draft_model = False
if use_draft_model:
    model_dir = repo_id
    draft_model_dir = specrepo_id

# Max number of batches to run at once, assuming the sequences will fit within total_context.
max_batch_size = 6 if paged else 1

# Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a
# new job is started, but at the expense of overall prompt ingestion speed.
max_chunk_size = 2048

# Max new tokens per completion. For this example applies to all jobs.
max_new_tokens = 2048

# Use LMFE to constrain the output to JSON format. See schema and details below.
json_mode = False

# Demonstrate token healing
healing = True

# Ban some phrases maybe
ban_strings = None
# ban_strings = [
#     "person to person",
#     "one person to another"
# ]


term = Terminal()

if use_draft_model:

    draft_config = ExLlamaV2Config(draft_model_dir)
    draft_model = ExLlamaV2(draft_config)

    draft_cache = ExLlamaV2Cache(
        draft_model,
        max_seq_len = total_context,
        lazy = True
    )

    draft_model.load_autosplit(draft_cache, progress = True)

else:

    draft_model = None
    draft_cache = None

# Create config. We use the default max_batch_size of 1 for the model and the default max_input_len of
# 2048, which will also be the limit of the chunk size for prefill used by the dynamic generator.

config = ExLlamaV2Config(model_dir)
config.max_input_len = max_chunk_size
config.max_attention_size = max_chunk_size ** 2
model = ExLlamaV2(config)

# Configure the cache. The dynamic generator expects a batch size of 1 and a max_seq_len equal to
# the total number of cached tokens. The flat cache will be split dynamically

cache = ExLlamaV2Cache(
    model,
    max_seq_len = total_context,
    lazy = True
)

model.load_autosplit(cache, progress = True)

# Also, tokenizer

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    draft_model = draft_model,
    draft_cache = draft_cache,
    tokenizer = tokenizer,
    max_batch_size = max_batch_size,
    use_ngram_draft = use_ngram,
    max_chunk_size = max_chunk_size,
    paged = paged,
)


# Active sequences and corresponding caches and settings
prompts = queue.Queue()
responses = {}
input_ids = []
# Global variable for storing partial responses
partial_responses = {}

# Create jobs
STATUS_LINES = 40  # Number of lines to dedicate for status messages
LLM_LINES = max_batch_size
status_area = StatusArea(STATUS_LINES)
displays = {}


if json_mode:
    print("Creating jobs... (initializing JSON filters could take a moment.)")


def get_stop_conditions(prompt_format, tokenizer):
    if prompt_format == "llama":
        return [tokenizer.eos_token_id]
    elif prompt_format == "llama3":
        return [tokenizer.single_id("<|eot_id|>")]
    elif prompt_format == "granite":
        return [tokenizer.eos_token_id, "\n\nQuestion:"]


# Only import lmfe if json_mode is set

if json_mode:
    import json
    from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
    from lmformatenforcer import JsonSchemaParser
    from exllamav2.generator.filters import ExLlamaV2PrefixFilter
    from pydantic import BaseModel
    from typing import Literal

    class JSONResponse(BaseModel):
        response: str
        confidence: Literal["low", "medium", "high"]
        is_subjective: Literal["no", "yes", "possibly"]

    schema_parser = JsonSchemaParser(JSONResponse.schema())



print("*** Loaded.. now Inference...:")

app = FastAPI(title="EXL2")

async def stream_response(prompt_id, timeout=180):
    global partial_responses
    while True:
        await asyncio.sleep(0.001)  # Sleep to yield control to the event loop

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
def process_prompts():
    global partial_responses  

        # To see what's going on, mode 1
    while True:
        while not prompts.empty() or len(input_ids):
            while len(input_ids) < max_batch_size and not prompts.empty():
                prompt_id, prompt, max_tokens, stream, temperature = prompts.get()
                if json_mode:
                    prompt += "\n\n Answer in JSON syntax."
                    filters = [
                        ExLlamaV2TokenEnforcerFilter(schema_parser, tokenizer),
                        ExLlamaV2PrefixFilter(model, tokenizer, "{")
                    ]
                else:
                    filters = None
                ids = tokenizer.encode(prompt, encode_special_tokens = True)
                prompt_tokens = ids.shape[-1]
                new_tokens = prompt_tokens + max_tokens
                #print("Processing prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                status_area.update(f"Processing prompt: {prompt_id}  Req tokens: {new_tokens}", line=STATUS_LINES-1)
                # Truncate if new_tokens exceed max_context
                if new_tokens > max_context:
                    # Calculate how many tokens to truncate
                    ids = tokenizer.encode("Say, 'Prompt exceeds allowed length. Please try again.'")
                    # Update new_tokens after truncation
                    prompt_tokens = ids.shape[-1]
                    new_tokens = prompt_tokens + max_tokens
                    print("Truncating prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                #prompt_length.append(prompt_tokens)
                input_ids.append(ids)
                #streamer.append(stream)
                #prompt_ids.append(prompt_id)
                
                job = ExLlamaV2DynamicJob(
                    input_ids = ids,
                    max_new_tokens = max_tokens,
                    stop_conditions = get_stop_conditions('llama3', tokenizer),
                    gen_settings = ExLlamaV2Sampler.Settings(),
                    banned_strings = ban_strings,
                    filters = filters,
                    filter_prefer_eos = True,
                    token_healing = healing
                )
                
                job.prompt_length = prompt_tokens
                job.input_ids = ids
                job.streamer = stream
                job.prompt_ids = prompt_id

                generator.enqueue(job)
                #displays = { job: JobStatusDisplay(job, line, STATUS_LINES) for line, job in enumerate(jobs) }
                displays[job] = JobStatusDisplay(job, STATUS_LINES)

                for index, (job, display) in enumerate(list(displays.items())):
                    display.update_position(index%LLM_LINES)  # Set position before updating
                
   
            if(len(input_ids)):
                #inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
                #logits = model.forward(inputs, caches, input_mask = None).float().cpu()
                
                results = generator.iterate()
                for r in results:
                #for i in range(len(input_ids)):
                    #r = results[i]
                    job = r["job"]
                    displays[job].update(r)
                    displays[job].display()
                    stage = r["stage"]
                    stage = r.get("eos_reason", stage)
                    outcontent = r.get("text", "")
                    reason = None
                    if(job.streamer):
                        partial_response_data = {
                                "id": f"chatcmpl-{job.prompt_ids}",
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
                        if job.prompt_ids not in partial_responses:
                            partial_responses[job.prompt_ids] = []
                        partial_responses[job.prompt_ids].append(partial_response_data)

                    if r['eos'] == True:
                        total_time = r['time_generate']
                        total_tokens = r['new_tokens']
                        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                        status_area.update(f"EOS detected: {stage}, Generated Tokens: {total_tokens}, Tokens per second: {tokens_per_second}/s", line=STATUS_LINES-2)

                        #generated_part = job.input_ids[:, job.prompt_length:]
                        #output = tokenizer.decode(generated_part[0]).strip()
                        #output = tokenizer.decode(input_ids[i])[0]
                        generated_text = r['full_completion']

                        # Calculate token counts
                        completion_tokens_old = (tokenizer.encode(generated_text)).shape[-1]
                        prompt_tokens_old = (tokenizer.encode(prompt)).shape[-1]

                        completion_tokens = r['new_tokens']
                        prompt_tokens = r['prompt_tokens']

                        full_tokens = completion_tokens + prompt_tokens
                        status_area.update(f"Completion Tokens: {completion_tokens_old}, New Completion Tokens: {completion_tokens}", line=STATUS_LINES-3)


                        eos_prompt_id = job.prompt_ids
                        if(job.streamer):
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
                        input_ids.pop()
                        #prompt_length.pop(i)
                        #streamer.pop(i)

        else:
            # Sleep for a short duration when there's no work
            time.sleep(0.1)  # Sleep for 100 milliseconds

# Start worker thread
worker = Thread(target=process_prompts)
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

@app.post('/v1/chat/completions')
async def mainchat(request: ChatCompletionRequest):

    try:
        prompt = ''
        if repo_str == 'Phind-CodeLlama-34B-v2':
            prompt = await format_prompt_code(request.messages)
        elif repo_str == 'zephyr-7b-beta':
            prompt = await format_prompt_zephyr(request.messages)
        elif repo_str == 'llama3-70b-instruct' or repo_str == 'llama3-70b-instruct-speculative':
            prompt = await format_prompt_llama3(request.messages)
        elif repo_str == 'Starling-LM-7B-alpha':
            prompt = await format_prompt_starling(request.messages)
        elif repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ' or repo_str == 'miqu-exl2-speculative':
            prompt = await format_prompt_mixtral(request.messages)
        elif repo_str == 'Yi-34B-Chat-GPTQ' or repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ' or repo_str == 'theprofessor-exl2-speculative' or repo_str == 'Yi-34B-Chat':
            prompt = await format_prompt_yi(request.messages)
        elif repo_str == 'Nous-Capybara-34B-GPTQ' or repo_str == 'goliath-120b-GPTQ' or repo_str == 'goliath-120b-exl2' or repo_str == 'goliath-120b-exl2-rpcal':
            prompt = await format_prompt_nous(request.messages)
        elif repo_str == 'tess-xl-exl2' or repo_str == 'tess-xl-exl2-speculative' or repo_str == 'venus-exl2-speculative':
            prompt = await format_prompt_tess(request.messages)
        elif repo_str == 'tinyllama-exl2-speculative':
            prompt = await format_prompt_zephyr(request.messages)
        else:
            prompt = await format_prompt(request.messages)
        status_area.update(f"Prompt: {prompt}")

        timeout = 180  # seconds
        start_time = time.time()
        prompt_id = generate_unique_id() # Replace with a function to generate unique IDs
        prompts.put((prompt_id, prompt, request.max_tokens, request.stream, request.temperature))

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

    uvicorn.run(app, host=host, port=port, log_level="error")

    print(term.enter_fullscreen())
    

    