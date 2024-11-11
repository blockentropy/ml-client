import asyncio
import json
import time
import configparser
import argparse
import outlines
import outlines.models
import outlines.models.vllm
import torch
from typing import AsyncIterable, List, Generator, Union, Optional


from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from threading import Thread, BoundedSemaphore


from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm import SamplingParams
from PIL import Image
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)



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
    content: str | list

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

parser = argparse.ArgumentParser(description='Run server with specified port.')

parser.add_argument('--port', type=int, help='Port to run the server on.')
parser.add_argument('--repo_str', type=str, default='llama3-70b-instruct', help='The model repository name')
parser.add_argument('--max_new_tokens', type=int, default=2048, help='Max new tokens.')
parser.add_argument('--use_outlines', action='store_true', help='Use outlines.')
parser.add_argument('--max_context', type=int, default=12288, help='Context length.')
parser.add_argument('--use_lora', action='store_true', help='Use lora.')
parser.add_argument('--lora_repo', type=str, default='llama3-70b-instruct', help='The lora model name')
parser.add_argument('--mm_limit', type=int, default=3, help='Number of multimodal inputs per prompt')


# Parse the arguments
args = parser.parse_args()


config = configparser.ConfigParser()
config.read('ml-client/config.ini')

repo_id = config.get(args.repo_str, 'repo')
host = config.get('settings', 'host')

port = args.port if args.port is not None else config.getint('settings', 'port')

# only allow one client at a time
busy = False
condition = asyncio.Condition()


torch_dtype = torch.float16  # Set a default dtype
if args.repo_str == 'zephyr-7b-beta' or args.repo_str == 'Starling-LM-7B-alpha':
    torch_dtype = torch.float16

revision = "main"
if args.repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ' or args.repo_str == 'Yi-34B-Chat-GPTQ':
    revision = 'gptq-4bit-32g-actorder_True'

remote_code = False
if args.repo_str == 'Nous-Capybara-34B-GPTQ':
    remote_code = True


max_input_length = 4096

if args.repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ' or args.repo_str == 'Nous-Capybara-34B-GPTQ':
    max_input_length = 8192

if args.use_lora:
    lora_repo = args.lora_repo
    lora_path = config.get(args.lora_repo, 'repo')


mm_limit = 3

# max_input_length = 32768

engine_args = AsyncEngineArgs(model=repo_id, max_seq_len_to_capture=max_input_length, tensor_parallel_size=1, device="cuda", revision=revision, trust_remote_code=remote_code, dtype=torch_dtype, enable_lora=args.use_lora, limit_mm_per_prompt={"image": mm_limit})

# Initialize model with vllm
if args.use_outlines:
    model = outlines.models.vllm(
        repo_id,
        enforce_eager=True, 
        dtype=torch_dtype,
        max_num_seqs=1,
        tensor_parallel_size=1,
        device="cuda",
        revision=revision,
        trust_remote_code=remote_code,
        max_seq_len_to_capture=max_input_length,
        enable_lora=args.use_lora,
        limit_mm_per_prompt={"image": mm_limit}
    )
else:
    model = AsyncLLMEngine.from_engine_args(engine_args)

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False, trust_remote_code=remote_code)

print("*** Loaded.. now Inference...:")

app = FastAPI(title="vLLM")


# Streaming case
async def streaming_request(prompt: str, image_urls: list, max_tokens: int = 1024, tempmodel: str = 'Llama70', response_format: str = 'completion'):

    global busy

    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens > max_input_length:
        print(f"Warning: over {max_input_length} tokens in context.")
        busy = False
        yield 'data: [DONE]'
        async with condition:
            condition.notify_all()
        return
    
    if args.use_outlines:
        print(f"Cannot use streaming with vLLM outlines integration.")
        busy = False
        yield 'data: [DONE]'
        async with condition:
            condition.notify_all()
        return

    inputs = {
        "prompt": prompt,
    }

    # For mutimodal queries
    if image_urls != []:
        urls = []

        for url in image_urls:
            img = fetch_image(url)
            urls.append(img)

        inputs["multi_modal_data"] = {"image": urls}
  

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None,
                                     repetition_penalty=1.1)
    
    prompt_id = random_uuid()

    if args.use_lora:
        # generator = model.generate(inputs, sampling_params=sampling_params, lora_request=LoRARequest(lora_repo, 1, lora_path), request_id=prompt_id)
        generator = await model.add_request(inputs=inputs, params=sampling_params, lora_request=LoRARequest(lora_repo, 1, lora_path), request_id=prompt_id)
    else:
        # generator = model.generate(inputs, sampling_params=sampling_params, request_id=prompt_id)
        generator = await model.add_request(inputs=inputs, params=sampling_params, request_id=prompt_id)

    try:
        previous_text = ""

        async for request_output in generator:
            if not request_output.outputs:
                continue

            output_text = request_output.outputs[-1].text
            print(f"Generated text: {output_text}")

            generated_text = output_text[len(previous_text):]

            previous_text = output_text

            if response_format == 'chat_completion':
                # Format the response as a proper SSE message
                response_data = {
                    "id": f"chatcmpl-{prompt_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": tempmodel,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": generated_text},
                        "finish_reason": None
                    }]
                }
            else:
                response_data = {
                "id": prompt_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": tempmodel,
                "choices": [
                    {
                        "index": 0,
                        "text": generated_text,
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }

            # Proper SSE format with data: prefix
            yield f"data: {json.dumps(response_data)}\n\n"

            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        error_response = {
            "error": str(e),
            "status": "error"
        }
        yield f"data: {json.dumps(error_response)}\n\n"

    # Send the final [DONE] message
    yield "data: [DONE]\n\n"

    busy = False
    async with condition:
        condition.notify_all()


async def non_streaming_request(prompt: str, image_urls: list,  max_tokens: int = 1024, outlines_dict: dict | None = None, tempmodel: str = 'Llama70',  response_format: str = 'completion'):

    # Assume generated_text is the output text you want to return
    # and assume you have a way to calculate prompt_tokens and completion_tokens
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print("Prompt Tokens: " + str(prompt_tokens))
    

    inputs = {
        "prompt": prompt,
    }

    # For mutimodal queries
    if image_urls != []:
        urls = []

        for url in image_urls:
            img = fetch_image(url)
            urls.append(img)

        inputs["multi_modal_data"] = {"image": urls}
  

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None,
                                     repetition_penalty=1.1)
    
    prompt_id = random_uuid()

    generated_text = ''
    try:
        
        if outlines_dict:
            stop_at = outlines_dict.get("stop_at", None)
            
            if outlines_dict["type"] == "choices":
                generator = outlines.generate.choice(model, outlines_dict["choices"])
            elif outlines_dict["type"] == "json":
                generator = outlines.generate.json(model, outlines_dict["json"])
            elif outlines_dict["type"] == "regex":
                generator = outlines.generate.regex(model, outlines_dict["regex"])
            else:
                generator = outlines.generate.text(model, sampling_params)

            if outlines_dict["type"] != "json":
                output = generator(prompt, stop_at=stop_at, max_tokens=max_tokens)
            else:
                output = generator(prompt, stop_at=stop_at)

            generated_text = output

           
        else:

            if args.use_lora:
                # output = model.generate(inputs, sampling_params=sampling_params, lora_request=LoRARequest(lora_repo, 1, lora_path), request_id=prompt_id)
                output = await model.add_request(inputs=inputs, params=sampling_params, lora_request=LoRARequest(lora_repo, 1, lora_path), request_id=prompt_id)
            else:
                # output = model.generate(inputs, sampling_params=sampling_params, request_id=prompt_id)
                output = await model.add_request(inputs=inputs, params=sampling_params, request_id=prompt_id)
                # output = await model.add_request(prompt_id, inputs, sampling_params)

            async for request_output in output:
                generated_text = request_output
            
            generated_text = generated_text.outputs[0].text

    except Exception as e:
        print(e)

    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory error caught. Attempting to free up memory.")
        torch.cuda.empty_cache()  # Free up unoccupied cached memory
        generated_text = "out of memory"

    completion_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
    full_tokens = completion_tokens + prompt_tokens


   # Prepare the response based on the format required
    if response_format == 'completion':
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
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": full_tokens
            }
        }
    elif response_format == 'chat_completion':
        response_data = {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": tempmodel,
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
    else:
        raise ValueError(f"Unsupported response_format: {response_format}")

    return response_data


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
            # formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"

            if isinstance(message.content, list):
                formatted_prompt += f"<|im_start|>user\n"
                for item in message.content:
                    if item["type"] == "image":
                        formatted_prompt += "<|image|><|eot_id|>"
                    elif item["type"] == "text":
                        text = item["text"]
                        formatted_prompt += f"<|begin_of_text|>{text}\n"
            else:
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"

        elif message.role == "assistant":
            formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"


    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_prompt


async def format_prompt_qwen(messages):
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
            if isinstance(message.content, list):
                formatted_prompt += f"<|im_start|>user\n"
                for item in message.content:
                    if item["type"] == "image":
                        formatted_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
                    elif item["type"] == "text":
                        text = item["text"]
                        formatted_prompt += f"{text}<|im_end|>\n"
            else:
                formatted_prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|im_start|>assistant\n"
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
            formatted_prompt += f"{message.content}\n\n"
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

@app.post('/v1/completions')
async def main(request: CompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        prompt = ""
        image_urls = []

        if isinstance(request.prompt, list):
            # handle list of strings
            prompt = request.prompt[0]  # just grabbing the 0th index
        else:
            # handle single string
            prompt = request.prompt

        if request.stream:
            response = StreamingResponse(streaming_request(prompt, image_urls, request.max_tokens, tempmodel=args.repo_str), media_type="text/event-stream")
        else:
            response_data = non_streaming_request(prompt, image_urls, request.max_tokens, tempmodel=args.repo_str)
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

@app.post('/v1/chat/completions')
async def mainchat(request: ChatCompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        # t = await num_tokens_from_messages(request.messages)
        # print(t)
 
        prompt = ''
        if args.repo_str == 'Phind-CodeLlama-34B-v2':
            prompt = await format_prompt_code(request.messages)
        elif args.repo_str == 'zephyr-7b-beta':
            prompt = await format_prompt_zephyr(request.messages)
        elif args.repo_str == 'Starling-LM-7B-alpha':
            prompt = await format_prompt_starling(request.messages)
        elif args.repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ':
            prompt = await format_prompt_mixtral(request.messages)
        elif args.repo_str == 'Yi-34B-Chat-GPTQ' or args.repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ':
            prompt = await format_prompt_yi(request.messages)
        elif args.repo_str == 'Nous-Capybara-34B-GPTQ' or args.repo_str == 'goliath-120b-GPTQ':
            prompt = await format_prompt_nous(request.messages)
        elif args.repo_str == 'Qwen2-VL-7B-Instruct':
            prompt = await format_prompt_qwen(request.messages)
        elif 'llama3' in args.repo_str.lower():
            prompt = await format_prompt_llama3(request.messages)
        else:
            prompt = await format_prompt(request.messages)

        print(prompt)

        # For multimodal requests
        image_urls = []
    
        for message in request.messages:
            if isinstance(message.content, list):
                    for item in message.content:
                        if item["type"] == "image":
                            image_urls.append(item["image"])


        # Structured outputs 
        outlines_dict = {}

        if args.use_outlines:
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
                assert (outlines_dict["type"] == "text") or not args.outlines
    
        if request.stream:
            response = StreamingResponse(streaming_request(prompt, image_urls, request.max_tokens, tempmodel=args.repo_str, response_format='chat_completion'), media_type="text/event-stream")
        else:
            response_data = await non_streaming_request(prompt, image_urls, request.max_tokens, outlines_dict = outlines_dict, tempmodel=args.repo_str, response_format='chat_completion')
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




@app.get('/ping')
async def get_status():
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
