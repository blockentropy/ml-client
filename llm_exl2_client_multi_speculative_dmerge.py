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

from copy import copy, deepcopy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread
from auto_gptq import exllama_set_max_input_length
import queue
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)
import uuid
from exllamav2.ext import exllamav2_ext as ext_c
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2 import ext

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

repo_str = 'tinyllama-exl2-speculative'
#repo_str = 'theprofessor-exl2-speculative'

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

# only allow one client at a time
busy = False
condition = asyncio.Condition()

config = ExLlamaV2Config()
config.model_dir = repo_id
config.prepare()

use_dynamic_rope_scaling = True
dynamic_rope_mult = 1.5
dynamic_rope_offset = 0.0

ropescale = 1.0
max_context = 12288
config.scale_alpha_value = ropescale
config.max_seq_len = max_context
base_model_native_max = 4096

# DRAFT 
draft_config = ExLlamaV2Config()
draft_config.model_dir = specrepo_id
draft_config.prepare()

draft_ropescale = 3.0
num_speculative_tokens = 5
speculative_prob_threshold = 0.25
draft_config.scale_alpha_value = draft_ropescale
draft_config.max_seq_len = max_context
draft_model_native_max = 2048

model = ExLlamaV2(config)
print("Loading model: " + repo_id)
#cache = ExLlamaV2Cache(model, lazy=True, max_seq_len = 20480)
#model.load_autosplit(cache)
model.load([9,9])

draft = ExLlamaV2(draft_config)
print("Loading draft model: " + specrepo_id)
draft.load()
tokenizer = ExLlamaV2Tokenizer(config)

# Cache mode

cache_8bit = True

settings_proto = ExLlamaV2Sampler.Settings()
settings_proto.temperature = 0
settings_proto.top_k = 50
settings_proto.top_p = 0.8
settings_proto.top_a = 0.0
settings_proto.token_repetition_penalty = 1.10
#settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

# Active sequences and corresponding caches and settings
prompts = queue.Queue()
responses = {}

input_ids = []
prompt_length = []
prompt_ids = []
streamer = []
caches = []
draft_caches = []
settings = []
draft_settings = []
future_tokens = []
future_logits = []
sin_arr = []
cos_arr = []
draft_sin_arr = []
draft_cos_arr = []

# Global variable for storing partial responses
partial_responses = {}

max_parallel_seqs = 5 
num_of_gpus = 2

dynamic_merge = True
## Dynamic Merge Slicing
if dynamic_merge == True:
    #layer_arrangement = list(range(0,20)) + list(range(10,30))+ list(range(20,40)) + list(range(30,50))+list(range(40,60))
    #layer_arrangement = list(range(0,6)) + list(range(2,8))+ list(range(4,10)) + list(range(6,12))+list(range(8,14)) +\
    #                    list(range(10,16)) + list(range(12,18))+ list(range(14,20)) + list(range(16,22))
    layer_arrangement = [0,1] + [i for i in range(2, 20) for _ in range(5)] + [ 20, 21]
    #layer_arrangement = list(range(0,22))
    #layer_arrangement = list(range(0,94))
    # modules arangement: [embedding, [...layers], rms-norm, head]
    # where each layer is [attention, mlp]
    old_modules = model.modules
    model.modules = old_modules[:1]
    #old_modules_dict = model.modules_dict
    #model.modules_dict = old_modules_dict[:1]
    model.modules_dict = {}
    #print("Keys in new model.modules_dict:")
    #for key in model.modules_dict.keys():
    #    print(key)
    model.modules_dict[model.modules[-1].key] = model.modules[-1]
    for i, idx in enumerate(layer_arrangement):
        model.modules += [copy(old_modules[idx*2 + 1])]
        model.modules[-1].layer_idx = i # for duplicate layers to use a different cache
        model.modules[-1].key = model.modules[-1].key.split('.')[0] + "." + model.modules[-1].key.split('.')[1] + '.' + str(i)
        model.modules_dict[model.modules[-1].key] = model.modules[-1]

        updated_submodules = []
        for m in model.modules[-1].submodules:
            copied_submodule = copy(m)
            updated_submodules.append(copied_submodule)
            # Split the key into parts
            key_parts = copied_submodule.key.split('.')
            key_parts[2] = str(i)  # Convert i to string, since we're dealing with string manipulation
            updated_key = '.'.join(key_parts)
            print(f"layer {i} is {updated_key}")
            #copied_submodule.reload()
            copied_submodule.key = updated_key
            # Check if the index exists and perform the check
            if len(key_parts) > 4:  # Ensure the list is long enough
                #copied_submodule.layer_idx = i
                copied_submodule.lora_a_tensors = {}
                copied_submodule.lora_b_tensors = {}
                if key_parts[4] == 'q_proj':
                    model.modules[-1].q_proj = copied_submodule
                if key_parts[4] == 'k_proj':
                    model.modules[-1].k_proj = copied_submodule
                if key_parts[4] == 'v_proj':
                    model.modules[-1].v_proj = copied_submodule
                if key_parts[4] == 'o_proj':
                    model.modules[-1].o_proj = copied_submodule
            #model.modules[-1].load()
            model.modules_dict[updated_key] = copied_submodule
        model.modules[-1].submodules = updated_submodules
        model_self =  model.modules[-1]
        if model_self.q_proj.is_quant():
            device_tensors = model.get_device_tensors(model_self.device_idx)
            device_tensors.begin_scratch_alloc()
            model_self.temp_state = device_tensors.get_scratch_slice(model_self.temp_state_size())
            model_self.temp_dq = device_tensors.get_scratch_slice(model_self.temp_dq_size())
            model_self.q_handle = ext_c.make_q_attn(model_self.input_layernorm.weight,
                                        model_self.input_layernorm.bias if model_self.input_layernorm.bias is not None else ext.none_tensor,
                                        isinstance(model_self.input_layernorm, ExLlamaV2RMSNorm),
                                        model_self.input_layernorm.variance_epsilon,
                                        model_self.q_proj.q_handle,
                                        model_self.k_proj.q_handle,
                                        model_self.v_proj.q_handle,
                                        model_self.o_proj.q_handle,
                                        model_self.temp_state,
                                        model_self.temp_dq,
                                        model.config.max_input_len * model.config.max_batch_size,
                                        model.config.hidden_size,
                                        model.config.num_attention_heads,
                                        model.config.num_key_value_heads,
                                        model.config.head_dim,
                                        model.config.max_seq_len)

        print(f"layer {i} is {model.modules[-1].key}")
        model.modules += [copy(old_modules[idx*2 + 2])]
        model.modules[-1].layer_idx = i
        model.modules[-1].key = model.modules[-1].key.split('.')[0] + "." +  model.modules[-1].key.split('.')[1] + '.' + str(i)
        model.modules_dict[model.modules[-1].key] = model.modules[-1]
        updated_submodules_mlp = []
        for m in model.modules[-1].submodules:
            copied_submodule = copy(m)
            updated_submodules_mlp.append(copied_submodule)
            # Split the key into parts
            key_parts = copied_submodule.key.split('.')
            key_parts[2] = str(i)  # Convert i to string, since we're dealing with string manipulation
            updated_key = '.'.join(key_parts)
            print(f"layer {i} is {updated_key}")
            #copied_submodule.reload()
            copied_submodule.key = updated_key
            if len(key_parts) > 4:  # Ensure the list is long enough
                #copied_submodule.layer_idx = i
                copied_submodule.lora_a_tensors = {}
                copied_submodule.lora_b_tensors = {}
                if key_parts[4] == 'down_proj':
                    model.modules[-1].down_proj = copied_submodule
                if key_parts[4] == 'up_proj':
                    model.modules[-1].up_proj = copied_submodule
                if key_parts[4] == 'gate_proj':
                    model.modules[-1].gate_proj = copied_submodule
            # Update the dictionary entry with the new key
            #model.modules[-1].load()
            model.modules_dict[updated_key] = copied_submodule
        model.modules[-1].submodules = updated_submodules_mlp
        model_self =  model.modules[-1]
        if model_self.up_proj.is_quant():
            device_tensors = model.get_device_tensors(model_self.device_idx)
            device_tensors.begin_scratch_alloc()
            model_self.q_handle = ext_c.make_q_mlp(model_self.post_attention_layernorm.weight,
                                        model_self.post_attention_layernorm.bias if model_self.post_attention_layernorm.bias is not None else ext.none_tensor,
                                        isinstance(model_self.post_attention_layernorm, ExLlamaV2RMSNorm),
                                        model_self.post_attention_layernorm.variance_epsilon,
                                        model_self.gate_proj.q_handle,
                                        model_self.up_proj.q_handle,
                                        model_self.down_proj.q_handle,
                                        device_tensors.get_scratch_slice(model_self.temp_state_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_a_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_b_size()),
                                        device_tensors.get_scratch_slice(model_self.temp_dq_size()),
                                        model.config.max_input_len * model.config.max_batch_size,
                                        model.config.architecture == "Gemma")

    model.modules += old_modules[-2:]
    model.head_layer_idx = len(model.modules) -1
    model.config.num_hidden_layers = len(layer_arrangement)
    model.last_kv_layer_idx = len(model.modules) -4

    model.modules_dict[old_modules[-2].key] = model.modules[-2]
    model.modules_dict[old_modules[-1].key] = model.modules[-1]
    model.cache_map = {}
    model.set_cache_map()

    print("Keys in new model.modules_dict:")
    for key in model.modules_dict.keys():
        print(key)
    print("Num of hidden layers:" +str(model.config.num_hidden_layers))
# Load LoRA
lora_directory = "../exllamav2/openhermes_out/checkpoint-12500/"
#lora_directory = "../exllamav2/trained_base_model_tinyllama.base/"
lora = ExLlamaV2Lora.from_directory(model, lora_directory)
#lora = None
for key in model.config.tensor_file_map.keys():
    print(key)



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
def process_prompts():
    global partial_responses  

    while True:
        while not prompts.empty() or len(input_ids):
            while len(input_ids) < max_parallel_seqs and not prompts.empty():
                prompt_id, prompt, max_tokens, stream, temperature = prompts.get()
                ids = tokenizer.encode(prompt)
                prompt_tokens = ids.shape[-1]
                new_tokens = prompt_tokens + max_tokens
                print("Processing prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                # Truncate if new_tokens exceed max_context
                if new_tokens > max_context:
                    # Calculate how many tokens to truncate
                    ids = tokenizer.encode("Say, 'Prompt exceeds allowed length. Please try again.'")
                    # Update new_tokens after truncation
                    prompt_tokens = ids.shape[-1]
                    new_tokens = prompt_tokens + max_tokens
                    print("Truncating prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                prompt_length.append(prompt_tokens)
                if use_dynamic_rope_scaling:
                    # Dynamic Rope Scaling
                    head_dim = model.config.head_dim
                    model_base = model.config.rotary_embedding_base
                    draft_head_dim = draft.config.head_dim
                    draft_model_base = draft.config.rotary_embedding_base
                    ratio = new_tokens / base_model_native_max
                    draft_ratio = new_tokens / draft_model_native_max
                    alpha = 1.0
                    draft_alpha = 3.0
                    ropesin = [None] * num_of_gpus
                    ropecos = [None] * num_of_gpus
                    draft_ropesin = [None] * num_of_gpus
                    draft_ropecos = [None] * num_of_gpus
                    if ratio > 1.0:
                        alpha = ((0.2500*ratio**2) + (0.3500*ratio) + 0.4000)*dynamic_rope_mult + dynamic_rope_offset
                        draft_alpha = (-0.13436 + 0.80541 * draft_ratio + 0.28833 * draft_ratio ** 2)*dynamic_rope_mult + dynamic_rope_offset
                        print("DYNAMIC ROPE SCALE Alpha: " + str(alpha) + "  Ratio: " + str(ratio) + " Draft Alpha: " + str(draft_alpha) + "  Draft Ratio: " + str(draft_ratio))

                    for g in range(num_of_gpus):
                        base = model_base
                        draft_base = draft_model_base
                        try:
                            tensors = model.get_device_tensors(g)
                        except IndexError:
                            tensors = None

                        try:
                            draft_tensors = draft.get_device_tensors(g)
                        except IndexError:
                            draft_tensors = None

                        if tensors is not None:
                            if alpha != 1.0: base *= alpha ** (model.config.head_dim / (model.config.head_dim - 2))

                            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = "cuda:"+str(g)).float() / head_dim))
                            t = torch.arange(model.config.max_seq_len, device = "cuda:"+str(g), dtype = torch.float32)

                            freqs = torch.einsum("i,j->ij", t, inv_freq)
                            emb = torch.cat((freqs, freqs), dim=-1)

                            ropesin[g] = emb.sin()[None, None, :, :].half()
                            ropecos[g] = emb.cos()[None, None, :, :].half()

                            #if torch.equal(tensors.sin, ropesin[g]):
                            #    print("Same")
                            #else:
                            #    print("Not same")
                            #    diff = torch.norm(tensors.sin - ropesin[g], p=2)  # Calculate L2 distance
                            #    print(f"Different: tensors.sin and ropesin[g]. Difference: {diff.item()}")
                            #    print("tensors.sin:", tensors.sin[0, 0, :3, :3])
                            #    print("ropesin[g]:", ropesin[g][0, 0, :3, :3])
                            #    print("inv_freq:", inv_freq[:3])
                            #    print("t:", t[:3])
                            #    print("head_dim:", head_dim)
                            #    print("base:", base)
                            #    print("alpha:", alpha) 
                            tensors.sin = ropesin[g]
                            tensors.cos = ropecos[g]

                        if draft_tensors is not None:
                            if draft_alpha != 1.0: draft_base *= draft_alpha ** (draft.config.head_dim / (draft.config.head_dim - 2))
                            draft_inv_freq = 1.0 / (draft_base ** (torch.arange(0, draft_head_dim, 2, device = "cuda:"+str(g)).float() / draft_head_dim))
                            draft_t = torch.arange(draft.config.max_seq_len, device = "cuda:"+str(g), dtype = torch.float32)

                            draft_freqs = torch.einsum("i,j->ij", draft_t, draft_inv_freq)
                            draft_emb = torch.cat((draft_freqs, draft_freqs), dim=-1)
                            draft_ropesin[g] = draft_emb.sin()[None, None, :, :].half()
                            draft_ropecos[g] = draft_emb.cos()[None, None, :, :].half()
                            draft_tensors.sin = draft_ropesin[g]
                            draft_tensors.cos = draft_ropecos[g]

                if cache_8bit:
                    ncache = ExLlamaV2Cache_8bit(model, max_seq_len = new_tokens)  # (max_seq_len could be different for each cache)
                    ncache_draft = ExLlamaV2Cache_8bit(draft, max_seq_len = new_tokens)  # (max_seq_len could be different for each cache)
                else:
                    ncache = ExLlamaV2Cache(model, max_seq_len = new_tokens)  # (max_seq_len could be different for each cache)
                    ncache_draft = ExLlamaV2Cache(draft, max_seq_len = new_tokens)  # (max_seq_len could be different for each cache)

                #print("Setting up Cache: " + str(prompt_id))
                
                if use_dynamic_rope_scaling:
                    sin_arr.append(ropesin)
                    cos_arr.append(ropecos)
                    draft_sin_arr.append(draft_ropesin)
                    draft_cos_arr.append(draft_ropecos)

                model.forward(ids[:, :-1], ncache, preprocess_only = True, loras = [lora])
                draft.forward(ids[:1, :-1], ncache_draft, preprocess_only = True)
                print("Cache setup: " + str(np.shape(ids[:1, :-1])))
                input_ids.append(ids)
                prompt_ids.append(prompt_id)
                caches.append(ncache)
                draft_caches.append(ncache_draft)
                streamer.append(stream)
                settings_proto.temperature = temperature
                settings.append(settings_proto.clone())  # Need individual settings per prompt to support Mirostat
                draft_settings.append(settings_proto.clone())
                future_tokens.append(None)
                future_logits.append(None)
                #print("Prompt added to queue: " + str(prompt_id))

            # Create a batch tensor of the last token in each active sequence, forward through the model using the list of
            # active caches rather than a single, batched cache. Then sample for each token indidividually with some
            # arbitrary stop condition
            if(len(input_ids)):
                #inputs = torch.cat([x[:, -1:] for x in input_ids], dim = 0)
                #logits = model.forward(inputs, caches, input_mask = None).float().cpu()
                eos = []
                r = random.random()
                for i in range(len(input_ids)):
                    # if using dynamic rope
                    if use_dynamic_rope_scaling:
                        for g in range(num_of_gpus):
                            if draft_sin_arr[i][g] is not None and draft_cos_arr[i][g] is not None:
                                draft_tensors = draft.get_device_tensors(g)
                                draft_tensors.sin = draft_sin_arr[i][g]
                                draft_tensors.cos = draft_cos_arr[i][g]
                            if sin_arr[i][g] is not None and cos_arr[i][g] is not None:
                                tensors = model.get_device_tensors(g)
                                tensors.sin = sin_arr[i][g]
                                tensors.cos = cos_arr[i][g]

                    if future_tokens[i] is None:
                        draft_sequence_ids = input_ids[i]
                        num_drafted_tokens = 0
                        for k in range(num_speculative_tokens):
                            logits = draft.forward(draft_sequence_ids[:, -1:], draft_caches[i]).float().cpu()
                            token, _, _, prob, _ = ExLlamaV2Sampler.sample(logits, draft_settings[i], draft_sequence_ids, random.random(), tokenizer)

                            if prob < speculative_prob_threshold:
                                draft_caches[i].current_seq_len -= 1
                                break

                            draft_sequence_ids = torch.cat((draft_sequence_ids, token), dim = 1)
                            num_drafted_tokens += 1
                        


                        # Rewind draft cache

                        draft_caches[i].current_seq_len -= num_drafted_tokens

                        # Forward last sampled token plus draft through model

                        if input_ids[i].shape[0] > 1:
                            future_tokens[i] = draft_sequence_ids[:, -1 - num_drafted_tokens:].repeat(input_ids[i].shape[0], 1)
                        else:
                            future_tokens[i] = draft_sequence_ids[:, -1 - num_drafted_tokens:]
                        future_logits[i] = model.forward(future_tokens[i], caches[i], input_mask = None, loras = [lora] ).float().cpu()

                        # Rewind model cache

                        caches[i].current_seq_len -= num_drafted_tokens + 1


                    token, _, _, _, _ = ExLlamaV2Sampler.sample(future_logits[i][:, :1, :], settings[i], input_ids[i], r, tokenizer)
                    future_logits[i] = future_logits[i][:, 1:, :]
                    future_tokens[i] = future_tokens[i][:, 1:]
                    caches[i].current_seq_len += 1
                    draft_caches[i].current_seq_len += 1

                    # If sampled token doesn't match future token or no more future tokens

                    if future_tokens[i].shape[-1] == 0 or future_tokens[i][0, 0] != token[0, 0]:
                        future_tokens[i] = None
                        future_logits[i] = None

                    input_ids[i] = torch.cat([input_ids[i], token], dim = 1)

                    new_text = tokenizer.decode(input_ids[i][:, -2:-1], decode_special_tokens=False)[0]
                    new_text2 = tokenizer.decode(input_ids[i][:, -2:], decode_special_tokens=False)[0]
                    diff = new_text2[len(new_text):]

                    #print(diff)
                    reason = None
                    if(streamer[i]):
                        ## Generator, yield here..
                        partial_response_data = {
                            "id": f"chatcmpl-{prompt_ids[i]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": repo_str,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": diff
                                    },
                                    "finish_reason": reason
                                }
                            ]
                        }

                        # Initialize a list for new prompt_id or append to existing one
                        if prompt_ids[i] not in partial_responses:
                            partial_responses[prompt_ids[i]] = []
                        partial_responses[prompt_ids[i]].append(partial_response_data)

                    if token.item() == tokenizer.eos_token_id or caches[i].current_seq_len == caches[i].max_seq_len - num_speculative_tokens:
                        eos.insert(0, i)
                        
                # Generate and store response
                for i in eos:
                    generated_part = input_ids[i][:, prompt_length[i]:]
                    output = tokenizer.decode(generated_part[0]).strip()
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
                    input_ids.pop(i)
                    caches.pop(i)
                    settings.pop(i)
                    prompt_length.pop(i)
                    streamer.pop(i)
                    draft_caches.pop(i)
                    draft_settings.pop(i)
                    future_tokens.pop(i)
                    future_logits.pop(i)
                    if use_dynamic_rope_scaling:
                        cos_arr.pop(i)
                        sin_arr.pop(i)
                        draft_cos_arr.pop(i)
                        draft_sin_arr.pop(i)

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
        elif repo_str == 'Starling-LM-7B-alpha':
            prompt = await format_prompt_starling(request.messages)
        elif repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ':
            prompt = await format_prompt_mixtral(request.messages)
        elif repo_str == 'Yi-34B-Chat-GPTQ' or repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ' or repo_str == 'theprofessor-exl2-speculative':
            prompt = await format_prompt_yi(request.messages)
        elif repo_str == 'Nous-Capybara-34B-GPTQ' or repo_str == 'goliath-120b-GPTQ' or repo_str == 'goliath-120b-exl2' or repo_str == 'goliath-120b-exl2-rpcal':
            prompt = await format_prompt_nous(request.messages)
        elif repo_str == 'tess-xl-exl2' or repo_str == 'tess-xl-exl2-speculative':
            prompt = await format_prompt_tess(request.messages)
        elif repo_str == 'tinyllama-exl2-speculative':
            prompt = await format_prompt_zephyr(request.messages)
        else:
            prompt = await format_prompt(request.messages)
        print(prompt)

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
