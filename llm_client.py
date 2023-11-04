import asyncio
import json
import os
import logging
import time
import configparser
import tiktoken
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

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stop: Optional[str] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None



config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get('Genz-70b-GPTQ', 'repo')
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

##Use tiktoken for token counts
async def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        message_attributes = vars(message)

        # Iterate over the key-value pairs of the attributes
        for key, value in message_attributes.items():
            num_tokens += len(encoding.encode(str(value)))  # Make sure to convert values to string if they are not already
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with assistant
    return num_tokens


#def model_generate(inputs, streamer, max_new_tokens):
#    global busy
#    busy = True
#    model.generate(input_ids=inputs['input_ids'], streamer=streamer, max_new_tokens=max_new_tokens)
#    busy = False

async def streaming_request(prompt: str, max_tokens: int = 100, tempmodel: str = 'Llama70', response_format: str = 'completion'):
    """Generator for each chunk received from OpenAI as response
    :param response_format: 'text_completion' or 'chat_completion' to set the output format
    :return: generator object for streaming response from OpenAI
    """
    global busy
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_tokens)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    completion_id = f"chatcmpl-{int(time.time() * 1000)}"  # Unique ID for the completion

    if response_format == 'chat_completion':
       yield f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{tempmodel}","choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":"null"}}]}}\n\n'

    for new_text in streamer:
        busy = True
        generated_text += new_text
        reason = None
        if "</s>" in new_text:
            reason = "stop"
            # Strip the </s> from the new_text
            new_text = new_text.replace("</s>", "")
        
        if response_format == 'chat_completion':
            response_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": tempmodel,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": new_text
                        },
                        "finish_reason": reason
                    }
                ]
            }
        else:  # default to 'completion'
            response_data = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": tempmodel,
                "choices": [
                    {
                        "index": 0,
                        "text": new_text,
                        "logprobs": None,
                        "finish_reason": reason
                    }
                ]
            }
            
        json_output = json.dumps(response_data)
        yield f"data: {json_output}\n\n"  # SSE format

    if response_format == 'chat_completion':
        yield f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{tempmodel}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n'
    else:
        yield 'data: [DONE]'

    busy = False
    async with condition:
        condition.notify_all()

def non_streaming_request(prompt: str, max_tokens: int = 100, tempmodel: str = 'Llama70', response_format: str = 'completion'):

    # Assume generated_text is the output text you want to return
    # and assume you have a way to calculate prompt_tokens and completion_tokens
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

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

@app.post('/v1/chat/completions')
async def mainchat(request: ChatCompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        t = await num_tokens_from_messages(request.messages)
        print(t)
 
        prompt = await format_prompt(request.messages)
        if request.stream:
            response = StreamingResponse(streaming_request(prompt, request.max_tokens, response_format='chat_completion'), media_type="text/event-stream")
        else:
            response_data = non_streaming_request(prompt, request.max_tokens, response_format='chat_completion')
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
