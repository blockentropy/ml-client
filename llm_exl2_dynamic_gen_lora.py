import asyncio
import json
import os
import time
import configparser
import argparse
from typing import AsyncIterable, List, Generator, Union, Optional
import traceback
import subprocess
import itertools

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread
import queue
import traceback
import re


import sys, os
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)

from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
import uuid
from blessed import Terminal
import textwrap
from outlines.integrations.exllamav2 import RegexFilter, TextFilter, JSONFilter, ChoiceFilter
from util import format_prompt_llama3, format_prompt, format_prompt_tess, format_prompt_commandr
from util_merge import ExLlamaV2MergePassthrough

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
    request_id: Optional[str] = None
    partial_generation: Optional[str] = None

#repo_str = 'theprofessor-exl2-speculative'

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')
parser.add_argument('--repo_str', type=str, default='llama3-70b-instruct', help='The model repository name')
parser.add_argument('--max_chunk_size', type=int, default=2048, help='Max chunk size.')
parser.add_argument('--max_new_tokens', type=int, default=2048, help='Max new tokens.')
parser.add_argument('--use_draft_model', action="store_true", help='Do speculative decoding')
parser.add_argument('--not_paged', action="store_true", help='Do not do paged attention')



# Parse the arguments
args = parser.parse_args()
repo_str = args.repo_str

term = Terminal()

class StatusArea:
    def __init__(self, num_lines):
        self.num_lines = min(num_lines, term.height - 8)  # Ensure we don't exceed terminal height
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
        self.spaces = " " * term.width
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
        output_length = term.width - len(text) +20
        text += (self.spaces + self.collected_output)[-output_length:].replace("\t", " ")

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

def get_stop_conditions(tokenizer):
    # get_stop_condition special case if model is llama3 
    if "llama3" in repo_str:
        return [tokenizer.single_id("<|eot_id|>"), tokenizer.eos_token_id, 198]
    # elif prompt_format == "granite":
    #     return [tokenizer.eos_token_id, "\n\nQuestion:"]
    else:
        return [tokenizer.eos_token_id]


configini = configparser.ConfigParser()
configini.read('config.ini')

repo_id = configini.get(repo_str, 'repo')
specrepo_id = configini.get(repo_str, 'specrepo')
host = configini.get('settings', 'host')
# Max individual context
max_context = int(configini.get(repo_str, 'max_context'))
# Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but
# the total to distribute dynamically over however many jobs are active at once
total_context = int(configini.get(repo_str, 'total_context'))

port = args.port if args.port is not None else configini.getint('settings', 'port')
display_mode = 1

# Whether to use paged mode or not. The generator is very handicapped in unpaged mode, does not support batching
# or CFG, but it will work without flash-attn 2.5.7+
paged = not args.not_paged

# Where to find our model
model_dir = repo_id

# N-gram or draft model speculative decoding. Largely detrimental to performance at higher batch sizes.
use_ngram = False
use_draft_model = args.use_draft_model
if use_draft_model:
    model_dir = repo_id
    draft_model_dir = specrepo_id

# Max number of batches to run at once, assuming the sequences will fit within total_context.
max_batch_size = 4 if paged else 1

# Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a
# new job is started, but at the expense of overall prompt ingestion speed.
max_chunk_size = args.max_chunk_size

# Max new tokens per completion. For this example applies to all jobs.
max_new_tokens = args.max_new_tokens

# Demonstrate token healing
healing = True



if use_draft_model:

    draft_config = ExLlamaV2Config(draft_model_dir)
    draft_config.scale_alpha_value = 6.0
    draft_config.max_seq_len = max_context
    draft_model = ExLlamaV2(draft_config)

    draft_cache = ExLlamaV2Cache_Q4(
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

#ropescale = 2.5
#config.scale_alpha_value = ropescale
config.max_seq_len = max_context
model = ExLlamaV2(config)

# Configure the cache. The dynamic generator expects a batch size of 1 and a max_seq_len equal to
# the total number of cached tokens. The flat cache will be split dynamically


#model.load_autosplit(cache, progress = True)
model.load([16,18,18], progress = True)
# Also, tokenizer

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)
hf_tokenizer_kwargs = {}
hf_tokenizer_kwargs.setdefault("padding_side", "left")
hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, **hf_tokenizer_kwargs)



# Model Merge
merge_model = configini.get(repo_str, 'merge_model')
if merge_model == 'True':
    model = ExLlamaV2MergePassthrough(model)


generators = {}
generator_name = configini.get(repo_str, 'string')
cache = ExLlamaV2Cache_Q4(
    model,
    max_seq_len = total_context,
    #lazy = True
)
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
generators[generator_name] = generator

for key in configini[repo_str]:
    if key.startswith('lora') and key.endswith('_name'):
        lora_index = key.split('_')[0]  # lora1, lora2, etc.
        lora_name = configini.get(repo_str, key)
        lora_repo_key = f'{lora_index}_repo'
        if lora_repo_key in configini[repo_str]:
            lora_repo = configini.get(repo_str, lora_repo_key)
            lora_model = ExLlamaV2Lora.from_directory(model, lora_repo)

            # Initialize a generator for each Lora
            lora_generator = ExLlamaV2DynamicGenerator(
                model=model,
                cache=ExLlamaV2Cache_Q4(model, max_seq_len=configini.getint(repo_str, 'total_context', fallback=32768)),
                draft_model = draft_model,
                draft_cache = draft_cache,
                tokenizer = tokenizer,
                max_batch_size = max_batch_size,
                use_ngram_draft = use_ngram,
                max_chunk_size = max_chunk_size,
                paged = paged,
            )
            lora_generator.set_loras(lora_model)
            generators[lora_name] = lora_generator
            print(f"Initialized {lora_name} Lora generator.")

#lora_directory = "../Documents/trained_llama3_lr2e4_r64/"
#lora = ExLlamaV2Lora.from_directory(model, lora_directory)
#lora = None



# Active sequences and corresponding caches and settings
prompts = queue.Queue()
responses = {}
prompt_length = {}
prompt_model = {}
# Global variable for storing partial responses
partial_responses = {}

# Create jobs
STATUS_LINES = term.height-8  # Number of lines to dedicate for status messages
LLM_LINES = max_batch_size
status_area = StatusArea(STATUS_LINES)
displays = {}
prompt_ids2jobs = {}
cancelled_request_ids = []

print("*** Loaded.. now Inference...:")

# take from https://github.com/tiangolo/fastapi/discussions/11360
class RequestCancelledMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global prompt_ids2jobs, prompt_length, cancelled_request_ids
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        # Let's make a shared queue for the request messages
        req_queue = asyncio.Queue()
        #cancelled_request_ids = []
        async def message_poller(sentinel, handler_task):
            nonlocal req_queue
            request_id = str(generate_unique_id())
            while True:
                message = await receive()
                #print(message)
                if "body" in message:
                    scope['extensions'] = {'request_id': request_id}

                if message["type"] == "http.disconnect":
                    cancelled_request_ids.append(request_id)
                    handler_task.cancel()
                    return sentinel # Break the loop

                # Puts the message in the queue
                await req_queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, req_queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            status_area.update(f"Cancelling request due to disconnect prompt", line=STATUS_LINES-1)
#             # TODO: FIgure out how to get prompt id that disconnected
#             while len(cancelled_request_ids) > 0:
#                 cancelled_id = cancelled_request_ids.pop()
#                 if cancelled_id in prompt_ids2jobs:
#                     generator.cancel(prompt_ids2jobs[cancelled_id])
#                     del prompt_ids2jobs[cancelled_id]
#                     del prompt_length[cancelled_id]
#                     status_area.update(f"Cancelling request due to disconnect prompt: {cancelled_id}", line=STATUS_LINES-1)
#                 else: 
#                     status_area.update(f"Cannot find job: {cancelled_id}", line=STATUS_LINES-1)


app = FastAPI(title="EXL2")
app.add_middleware(RequestCancelledMiddleware)

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


def process_prompts():
    global partial_responses
    global prompt_ids2jobs, prompt_length, prompt_model, cancelled_request_ids
    try:

        while True:
            while not prompts.empty() or len(prompt_length):
                while len(prompt_length) < max_batch_size and not prompts.empty():
                    prompt_id, prompt, max_tokens, stream, temperature, rmodel, outlines_dict = prompts.get()
                    stop_at = outlines_dict.get("stop_at", None)
                    if outlines_dict["type"] == "choices":
                        filters = [ChoiceFilter(outlines_dict["choices"], hf_tokenizer)]
                    elif outlines_dict["type"] == "json":
                        filters = [JSONFilter(outlines_dict["json"], hf_tokenizer)]
                    elif outlines_dict["type"] == "regex":
                        filters = [RegexFilter(outlines_dict["regex"], hf_tokenizer)]
                    else:
                        filters = []
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
                    prompt_length[prompt_id] = prompt_tokens
                    #streamer.append(stream)
                    #prompt_ids.append(prompt_id)

                    preferred_eos = get_stop_conditions(tokenizer)

                    if stop_at is not None:
                        preferred_eos.append(stop_at)

                    gen_settings = ExLlamaV2Sampler.Settings()
                    gen_settings.temperature = 1.0 if temperature>1 else temperature  # To make sure the temperature value does not exceed 1

                    job = ExLlamaV2DynamicJob(
                        input_ids = ids,
                        max_new_tokens = max_tokens,
                        stop_conditions = preferred_eos if stop_at is None else [tokenizer.eos_token_id, stop_at],
                        gen_settings = gen_settings,
                        filters = filters,
                        token_healing = healing
                    )

                    # Check if rmodel exists in the generators dictionary
                    if rmodel not in generators:
                        rmodel = generator_name  # Set rmodel to the default generator name if not found

                    # Now select the generator with the possibly updated rmodel
                    selected_generator = generators[rmodel]
                    status_area.update(f"Using generator: {rmodel}", line=STATUS_LINES-1)
                
                    job.prompt_length = prompt_tokens
                    job.input_ids = ids
                    job.streamer = stream
                    job.prompt_ids = prompt_id
                    job.stop_at = stop_at
                    job.model = rmodel
                    
                    selected_generator.enqueue(job)
                    
                    displays[job] = JobStatusDisplay(job, STATUS_LINES)

                    for index, (job, display) in enumerate(list(displays.items())):
                        display.update_position(index%LLM_LINES)  # Set position before updating
                    prompt_ids2jobs[prompt_id] = job
                    prompt_model[prompt_id] = rmodel

                if(len(prompt_length)):
                    # Collect all results iterables
                    all_results = (gen.iterate() for gen in generators.values())

                    # Chain all results into a single iterable
                    results = itertools.chain.from_iterable(all_results)
                    for r in results:
                        job = r["job"]
                        #displays[job].update(r)
                        #displays[job].display()
                        stage = r["stage"]
                        stage = r.get("eos_reason", stage)
                        outcontent = r.get("text", "")
                        print(outcontent)
                        print(r.get("token_ids", ""))
                        reason = None
                        if(job.streamer):
                            if r["eos"] and job.stop_at is not None:
                                outcontent += job.stop_at
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
                                if job.stop_at is not None:
                                    generated_text += job.stop_at
                                response_data = {
                                    "id": f"chatcmpl-{eos_prompt_id}",
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
                            del prompt_ids2jobs[eos_prompt_id]
                            del prompt_length[eos_prompt_id]
                            del prompt_model[eos_prompt_id]
                    if len(cancelled_request_ids):
                        cancelled_id = cancelled_request_ids.pop()
                        status_area.update(f"Cancelling request due to disconnect prompt: {cancelled_id}", line=STATUS_LINES-1)
                        if cancelled_id in prompt_ids2jobs:
                            which_model = prompt_model[cancelled_id]
                            generators[which_model].cancel(prompt_ids2jobs[cancelled_id])
                            del prompt_ids2jobs[cancelled_id]
                            del prompt_length[cancelled_id]
                            del prompt_model[cancelled_id]
                            status_area.update(f"Found and cancelling: {cancelled_id}", line=STATUS_LINES-1)
                        else: 
                            # Temporarily store items to check against cancelled_id
                            temp_storage = []

                            # Drain the queue and check each item
                            while not prompts.empty():
                                prompt_id, prompt, max_tokens, stream, temperature, rmodel, outlines_dict = prompts.get()
                                if prompt_id != cancelled_id:
                                    # Only requeue prompts that do not match the cancelled_id
                                    temp_storage.append((prompt_id, prompt, max_tokens, stream, temperature, rmodel, outlines_dict))

                            # Re-add the valid items back to the queue
                            for item in temp_storage:
                                prompts.put(item)



            else:
                # Sleep for a short duration when there's no work
                time.sleep(0.1)  # Sleep for 100 milliseconds
    except Exception as e:
        print("Reset server due to ", e)
        print(traceback.format_exc())
        for prompt_id in prompt_ids2jobs:
            job = prompt_ids2jobs[prompt_id]
            if(job.streamer):
                ## Generator, yield here..
                partial_response_data = {
                    "finish_reason": "stop"
                }

                responses[prompt_id] = partial_response_data
            else:
                print("Error handling for full generation current not implemented")
            generators[job.model].cancel(job)
            #generator.cancel(job)
        prompt_ids2jobs = {}
        prompt_length = {}
        prompt_model = {}

# Start worker thread
worker = Thread(target=process_prompts)
worker.start()


@app.post('/v1/chat/completions')
async def mainchat(requestid: Request, request: ChatCompletionRequest):
    try:
        prompt = ''
        if repo_str == 'Phind-CodeLlama-34B-v2':
            prompt = await format_prompt_code(request.messages)
        elif repo_str == 'zephyr-7b-beta':
            prompt = await format_prompt_zephyr(request.messages)
        elif repo_str == 'llama3-70b-instruct' or 'llama3-70b-instruct-speculative':
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
        if request.partial_generation is not None:
            prompt += request.partial_generation
        

        timeout = 180  # seconds
        start_time = time.time()
        prompt_id = requestid.scope.get("extensions", {}).get("request_id", "Unknown ID")
        #prompt_id = generate_unique_id()
        status_area.update(f"Prompt: {prompt}, Prompt ID: {prompt_id}")
        outlines_dict = {}
        
        # Adjust temperature if it is 0
        if request.temperature == 0:
            request.temperature = 0.001

        if request.stop_at is not None:
            outlines_dict["stop_at"] = request.stop_at
        if request.outlines_type is not None:
            outlines_dict["type"] = request.outlines_type
        else:
            outlines_dict["type"] = "text"
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
            assert outlines_dict["type"] == "text"
        prompts.put((prompt_id, prompt, request.max_tokens, request.stream, request.temperature, request.model, outlines_dict))

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





@app.get('/ping')
async def get_status():
    return {"ping": sum(prompt_length.values())}

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
