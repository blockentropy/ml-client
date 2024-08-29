import json

def tools_to_string(tools):
    if not tools:
        return None
    
    tool_strings = []
    for tool in tools:
        tool_string = json.dumps(tool, indent=2)
        tool_strings.append(tool_string)
    
    return "\n\n".join(tool_strings)

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


async def format_prompt_llama3(messages, tool_string=None):
    formatted_prompt = ""
    system_message = None

    # Check for an existing system message
    for message in messages:
        if message.role == "system":
            system_message = message.content
            break

    if tool_string:
        tool_string = tools_to_string(tool_string)
        system_message = "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question."
    elif not system_message:
        system_message = "You are a helpful AI assistant."

    tool_preface = 'Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n\
    Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.'

    # Add the system message
    formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"

    first_user_message = True
    for message in messages:
        if message.role == "user":
            user_content = message.content
            if tool_string:
                if first_user_message:
                    user_content = f"{tool_preface}\n\n{tool_string}\n\nQuestion: {user_content}"
                    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                    first_user_message = False
                else:
                    formatted_prompt += f"<|start_header_id|>ipython<|end_header_id|>\n\n{user_content}<|eot_id|>"
            else:
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
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


