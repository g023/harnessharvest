# 2 main functions:
# llm_nonstream(conv=[], thinking=True, options=G_OPTIONS)
# llm_stream(conv=[], thinking=True, options=G_OPTIONS)
# Any thinking outputs will detect and load a reasoning field on a return and content for the answer

import requests
import json
from typing import List, Dict, Optional, Generator, Union
import os
import time
import random

G_LOOP_SIZE = 75  # Size of the substring to check for repeats

G_APPEND_PROMPT = "" # can set to "no_think" to disable thinking for models that don't support it, or "think:" to use a custom think prefix for models that do support it.

G_HOST = "http://localhost:11434"  # Default host for Ollama server

# # if you want to use Qwen3.5 use these settings:
G_MODEL = "qwen3:4b-thinking-2507-q4_K_M" # if
G_THINKING = True # modern LLMs want this switch to enable the "thinking" process. Even when disabled, if there is a </think> tag in the output it will still split reasoning and content
# G_APPEND_PROMPT = "" # can set to "no_think" to disable thinking for models that don't support it, or "think:" to use a custom think prefix for models that do support it.

# OLLAMA PULL AND THEN ROCK AND ROLL
# ollama pull hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0
# so this is a model I created by duplicating a layer in the orig qwen3 1.7B . Its fast. 

G_MODEL = "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0" 
# G_TURBO_TIME = True # for hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0 to run without thinking at full clip (very very very fast; still smart)
# if G_TURBO_TIME:
#     # does not support thinking param so have to set thinking=False but you can disable thinking with 'no_think' in the prompt you send
#     G_THINKING = False # old school LLMs like Qwen3 original (can deactivate think with no_think in prompt)
#     G_APPEND_PROMPT = "\n<ignore_token:no_think>\n" # can set to "no_think" to disable thinking for models that don't support it, or "think:" to use a custom think prefix for models that do support it.
# else:
#     G_MODEL = "hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0" 
#     # does not support thinking param so have to set thinking=False but you can disable thinking with 'no_think' in the prompt you send
#     G_THINKING = False # old school LLMs like Qwen3 original (can deactivate think with no_think in prompt)
#     G_APPEND_PROMPT = "" # can set to "no_think" to disable thinking for models that don't support it, or "think:" to use a custom think prefix for models that do support it.

G_CONTEXT_WINDOW = 40000 # num_ctx in ollama api
G_MAX_OUTPUT_TOKENS = 16384  # -1 for no limit, otherwise set to desired max output tokens (e.g., 2048) # controls how many tokens are output
G_TEMP = 1.0

#   "temperature": 0.65,
#   "top_p": 0.9,
#   "top_k": 20,
#   "min_p": 0.0,
#   "repeat_penalty": 1.05,
#   "presence_penalty": 0.1,
#   "frequency_penalty": 0.1

G_OPTIONS = {
    # "num_keep": 5, # Keep last 5 messages for context
    # "seed": 42,
    "num_predict": G_MAX_OUTPUT_TOKENS,
    "top_k": 95,
    "top_p": 0.95,
    "min_p": 0.35,
    "typical_p": 0.3,
    "repeat_last_n": 16384,
    "temperature": G_TEMP,
    "repeat_penalty": 15.2,
    "presence_penalty": 0.5,
    "frequency_penalty": 1.0,
    "mirostat": 2,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": True,
    # "stop": ["\n", "user:"],
    "numa": False,
    "num_ctx": G_CONTEXT_WINDOW,
    "num_batch": 2,
    # "num_gpu": 1,
    # "main_gpu": 0,
    "low_vram": False,
    "vocab_only": False,
    "use_mmap": True,
    "use_mlock": True,
    "num_thread": 1        
}

_STREAM_DONE = object()

# response from URL
# Available models: {'models': [{'name': 'qwen3.5:9b', 'model': 'qwen3.5:9b', 'modified_at': '2026-03-19T10:44:54.916232506-06:00', 'size': 6594474711, 'digest': '6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35', 'families': ['qwen35'], 'parameter_size': '9.7B', 'quantization_level': 'Q4_K_M'}}, {'name': 'hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0', 'model': 'hf.co/g023/Qwen3-1.77B-g023-GGUF:Q8_0', 'modified_at': '2026-03-17T23:48:10.772799698-06:00', 'size': 1887924365, 'digest': 'aa6a780c31fe01e26decd483945cbc53c6cfeccef6a2788599638f310e64ae81', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen3', 'families': ['qwen3'], 'parameter_size': '1.77B', 'quantization_level': 'unknown'}}, {'name': 'qwen3.5:0.8b', 'model': 'qwen3.5:0.8b', 'modified_at': '2026-03-15T21:41:52.569383136-06:00', 'size': 1036046583, 'digest': 'f3817196d142eaf72ce79dfebe53dcb20bd21da87ce13e138a8f8e10a866b3a4', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35', 'families': ['qwen35'], 'parameter_size': '873.44M', 'quantization_level': 'Q8_0'}}, {'name': 'qwen3.5:2b', 'model': 'qwen3.5:2b', 'modified_at': '2026-03-15T21:02:58.062661704-06:00', 'size': 2741192820, 'digest': '324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35', 'families': ['qwen35'], 'parameter_size': '2.3B', 'quantization_level': 'Q8_0'}}, {'name': 'qwen3:1.7b', 'model': 'qwen3:1.7b', 'modified_at': '2026-03-15T13:12:10.148438465-06:00', 'size': 1359293444, 'digest': '8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen3', 'families': ['qwen3'], 'parameter_size': '2.0B', 'quantization_level': 'Q4_K_M'}}, {'name': 'qwen3.5:4b-q4_K_M', 'model': 'qwen3.5:4b-q4_K_M', 'modified_at': '2026-03-11T15:00:46.612888456-06:00', 'size': 3389983735, 'digest': '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35', 'families': ['qwen35'], 'parameter_size': '4.7B', 'quantization_level': 'Q4_K_M'}}, {'name': 'dolphin-mistral:7b', 'model': 'dolphin-mistral:7b', 'modified_at': '2026-03-06T11:12:45.453109351-07:00', 'size': 4108940323, 'digest': '5dc8c5a2be6510dcb2afbcffdedc73acbd5868d2c25d9402f6044beade3d5f70', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'llama', 'families': ['llama'], 'parameter_size': '7B', 'quantization_level': 'Q4_0'}}, {'name': 'hf.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive:Q4_K_M', 'model': 'hf.co/HauhauCS/Qwen3-4B-2507-Instruct-Uncensored-HauhauCS-Aggressive:Q4_K_M', 'modified_at': '2026-03-06T07:57:05.893841398-07:00', 'size': 2497281068, 'digest': 'a3568d632e3582f93787b6d9057fe4bfa02b9e3991e9bc1fd9ab69578cd2381c', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen3', 'families': ['qwen3'], 'parameter_size': '4.02B', 'quantization_level': 'unknown'}}, {'name': 'qwen3.5:35b-a3b-q4_K_M', 'model': 'qwen3.5:35b-a3b-q4_K_M', 'modified_at': '2026-03-04T10:22:39.34820127-07:00', 'size': 23869191742, 'digest': '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen35moe', 'families': ['qwen35moe'], 'parameter_size': '36.0B', 'quantization_level': 'Q4_K_M'}}, {'name': 'qwen3:4b-thinking-2507-q4_K_M', 'model': 'qwen3:4b-thinking-2507-q4_K_M', 'modified_at': '2026-02-06T00:04:44.154256714-07:00', 'size': 2497293931, 'digest': '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen3', 'families': ['qwen3'], 'parameter_size': '4.0B', 'quantization_level': 'Q4_K_M'}}, {'name': 'hf.co/mradermacher/gemma-3-4b-it-heretic-uncensored-abliterated-Extreme-GGUF:Q4_K_M', 'model': 'hf.co/mradermacher/gemma-3-4b-it-heretic-uncensored-abliterated-Extreme-GGUF:Q4_K_M', 'modified_at': '2026-01-20T22:51:34.688744432-07:00', 'size': 3081273571, 'digest': '4323b3915fa66500cbab3dfb3079ac895379906f385cad667dd725f16dd8e8cd', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'gemma3', 'families': ['gemma3'], 'parameter_size': '3.88B', 'quantization_level': 'unknown'}}, {'name': 'fluffy/l3-8b-stheno-v3.2:latest', 'model': 'fluffy/l3-8b-stheno-v3.2:latest', 'modified_at': '2025-12-13T22:21:58.263738496-07:00', 'size': 4921247877, 'digest': 'f1afe09480f356ebb81f43d96f82220b5adc541b36421434628a758f0493095a', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'llama', 'families': ['llama'], 'parameter_size': '8.0B', 'quantization_level': 'Q4_K_M'}}, {'name': 'goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest', 'model': 'goonsai/qwen2.5-3B-goonsai-nsfw-100k:latest', 'modified_at': '2025-12-13T22:19:25.577567739-07:00', 'size': 6178922220, 'digest': 'cf62f31c7147c87e67f2bab1859ce482335862f3a67d94c53ad286c57b88351e', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen2', 'families': ['qwen2'], 'parameter_size': '3.1B', 'quantization_level': 'F16'}}, {'name': 'qwen3:4b-instruct-2507-q4_K_M', 'model': 'qwen3:4b-instruct-2507-q4_K_M', 'modified_at': '2025-10-17T00:33:23.277692624-06:00', 'size': 2497293803, 'digest': '0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'qwen3', 'families': ['qwen3'], 'parameter_size': '4.0B', 'quantization_level': 'Q4_K_M'}}, {'name': 'brxce/stable-diffusion-prompt-generator:latest', 'model': 'brxce/stable-diffusion-prompt-generator:latest', 'modified_at': '2024-09-04T08:44:17.814551443-06:00', 'size': 4108917578, 'digest': '474a09318a2e9e88320ecd6792b1142ac7bb9f78fbf991e1856d75ccaea4378a', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'llama', 'families': ['llama'], 'parameter_size': '7B', 'quantization_level': 'Q4_0'}}]}

def ollama_get_models(host: str = None) -> List[str]:
    """Fetch the list of available models from the Ollama server."""
    effective_host = _resolve_host(host)
    try:
        response = requests.get(f"{effective_host}/api/tags", timeout=10)
        response.raise_for_status()

        models_info = response.json()
        models = []

        for model in models_info.get("models", []):
            name = model.get("name")
            if name:
                models.append(name)

        return models

    except requests.exceptions.RequestException as e:
        print(f"Error fetching models from Ollama server at {effective_host}: {e}")
        return []
    

def _resolve_host(host: Optional[str]) -> str:
    """Resolve the Ollama host with explicit override, environment fallback, then default."""
    return (host or os.getenv("OLLAMA_HOST") or G_HOST).rstrip('/')


def _parse_stream_line(raw_line: Union[bytes, str]) -> Optional[Union[Dict, object]]:
    """Parse one SSE line from an Ollama streaming response."""
    if isinstance(raw_line, bytes):
        line = raw_line.decode('utf-8', errors='replace')
    else:
        line = raw_line

    line = line.strip()
    if not line or line.startswith(':'):
        return None

    if line.startswith('data:'):
        payload = line[5:].strip()
        if not payload or payload == '[DONE]':
            return _STREAM_DONE
    else:
        payload = line

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Failed to decode chunk: {raw_line!r}, error: {exc}")
        return None


def _is_reasoning_effort_unsupported_error(status_code: int, body: str) -> bool:
    if status_code != 400 or not isinstance(body, str):
        return False

    lowered = body.lower()
    return (
        ("think value" in lowered and "not supported" in lowered)
        or "does not support thinking" in lowered
        or "does not support think" in lowered
        or "does not support 'think'" in lowered
    )

def chat_with_ollama(
    messages: List[Dict[str, str]],
    model: str = G_MODEL,
    host: str = None,
    stream: bool = False,
    reasoning_effort: Optional[str] = None,
    options: Dict = G_OPTIONS,  # default options with num_ctx set to max
    thinking: bool = G_THINKING,  #
    **kwargs
) -> Union[Dict, Generator[Dict, None, None]]:
    """
    Sends a conversation to Ollama and returns the model's response.
    This function is designed to be beautiful, robust, and leverage the latest Ollama features.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries.
                                         Each dict should have 'role' (e.g., 'user', 'assistant', 'system')
                                         and 'content' (the message text).
        model (str): The name of the model to use (e.g., "gemma3", "llama3.2", "nemotron-3-nano").
                     Defaults to "gemma3". You can also use the ':cloud' suffix for cloud models.
        host (str, optional): The base URL of the Ollama server. If None, it tries to get it from the
                              OLLAMA_HOST environment variable, otherwise defaults to "http://localhost:11434".
        stream (bool): If True, returns a generator yielding response chunks. If False, returns the full response.
                       Defaults to False.
        reasoning_effort (str, optional): For reasoning models, controls the effort ("low", "medium", "high").
        options (Dict, optional): Additional options for the model (e.g., {"num_ctx": 32767}).
        thinking (bool, optional): Whether to include the "thinking" process in the response. Defaults to True.
        **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens, top_p).

    Returns:
        Union[Dict, Generator[Dict, None, None]]: If stream=False, returns a dictionary with the full response.
                                                   If stream=True, returns a generator yielding chunks.

    Raises:
        ConnectionError: If the Ollama server is not reachable.
        ValueError: If the input messages are invalid.
        requests.exceptions.RequestException: For other API-related errors.
    """

    # append G_APPEND_PROMPT to the last user message if set
    if G_APPEND_PROMPT and messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += f"{G_APPEND_PROMPT}"


    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": thinking,
        "options": options,
    }

    # force num_ctx and output token sizes from the globals if not set on request
    if "num_ctx" not in payload["options"]:
        payload["options"]["num_ctx"] = G_CONTEXT_WINDOW
    if "num_predict" not in payload["options"]:
        payload["options"]["num_predict"] = G_MAX_OUTPUT_TOKENS
    # force temp
    if "temperature" not in payload["options"]:
        payload["options"]["temperature"] = G_TEMP

    headers = {"Content-Type": "application/json"}

    # --- Configuration & Validation ---
    if not messages:
        raise ValueError("The 'messages' list cannot be empty.")

    # Determine host with environment variable fallback
    effective_host = _resolve_host(host)
    
    # --- Select Endpoint ---
    endpoint = f"{effective_host}/api/chat"
    
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Add any additional parameters from kwargs (e.g., temperature, max_tokens)
    payload.update(kwargs)
    

    # print model, temp, num_ctx for debugging
    print(f"Using model: {model}")
    print(f"Temperature: {payload['options']['temperature']}")
    print(f"Context window (num_ctx): {payload['options']['num_ctx']}")

    # --- Make the Request ---
    try:
        if stream:
            # For streaming, we return a generator
            return _stream_response(endpoint, headers, payload)
        else:
            # For non-streaming, make a single request
            response = requests.post(endpoint, headers=headers, json=payload, timeout=600)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                body = getattr(getattr(e, "response", None), "text", "")
                if reasoning_effort and _is_reasoning_effort_unsupported_error(response.status_code, body):
                    retry_payload = payload.copy()
                    retry_payload.pop("reasoning_effort", None)
                    response = requests.post(
                        endpoint, headers=headers, json=retry_payload, timeout=600)
                    response.raise_for_status()
                    return response.json()
                raise
            return response.json()
        
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Could not connect to Ollama server at {effective_host}. "
                              f"Please ensure it's running (ollama serve).") from e
    except requests.exceptions.Timeout as e:
        raise requests.exceptions.Timeout("Request to Ollama timed out. Consider increasing the timeout.") from e
    except requests.exceptions.RequestException as e:
        # Re-raise other request exceptions
        raise e


def _stream_response(endpoint: str, headers: Dict, payload: Dict) -> Generator[Dict, None, None]:
    """
    Internal generator to handle streaming responses.
    Yields parsed JSON chunks from the server-sent events stream.
    """
    payload["stream"] = True  # Ensure stream is enabled
    with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            chunk = _parse_stream_line(line)
            if chunk is None:
                continue
            if chunk is _STREAM_DONE:
                break
            yield chunk

            if isinstance(chunk, dict):
                if chunk.get("done"):
                    break
                choices = chunk.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason in {"stop", "length", "content_filter"}:
                        break


def llm_nonstream(conv=[], thinking=True, options=G_OPTIONS, the_model=None):

    ret_dict = {
        "reasoning": "",
        "content": "",
        "usage": {},
        "time_taken": 0,
    }

    effective_model = the_model or G_MODEL

    print("\n--- (Non-Streaming) ---") # 98.14 tokens/second # ~ 25-30% faster than streaming ;)
    try:
        time_start = time.time()    

        response = chat_with_ollama(
            messages=conv,
            model=effective_model, 
            temperature=G_TEMP,
            reasoning_effort="medium",  # New parameter for reasoning models, # don't really care as the models that use it don't really work for me *wink* *wink* *nudge* *nudge*
            thinking=thinking,
            options=options
        )

        ### XXXX
        message = response['message']

        ret_dict["time_taken"] = time.time() - time_start
        ret_dict["reasoning"] = message.get('thinking')
        ret_dict["content"] = message.get('content')

        if "</think>" in ret_dict["content"]:
            # break it in two at the LAST occurrence of </think>
            parts = ret_dict["content"].rsplit("</think>", 1)
            r = parts[0].strip()
            c = parts[1].strip()

            ret_dict["reasoning"] = (ret_dict["reasoning"] or "") + r
            ret_dict["content"] = c


        # tokens are roughly 3.245 characters so calculate estimated reason/output/total tokens based on that.
        reasoning_tokens = len(ret_dict.get('reasoning', '')) / 3.245
        content_tokens = len(ret_dict.get('content', '')) / 3.245
        total_tokens = reasoning_tokens + content_tokens
        # round to nearest whole number
        reasoning_tokens = round(reasoning_tokens)
        content_tokens = round(content_tokens)
        total_tokens = round(total_tokens)

        ret_dict["usage"] = {
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": content_tokens,
            "total_tokens": total_tokens,
        }


    except Exception as e:
        print(f"Error: {e}")

    return ret_dict

def llm_stream(conv=[], thinking=True, options=G_OPTIONS, retry_on_repeat=False,the_model=G_MODEL): # loop detection sucks
    print("\n--- Streaming Example ---") # 80.61 tokens/second

    retry_on_repeat = False # force for now

    options = options.copy()  # Work on a copy to avoid modifying the original

    print(f"Using model: {the_model}")
    print(f"Temperature: {options.get('temperature', G_TEMP)}")
    # print(f"Context window (num_ctx): {options.get('num_ctx', G_CONTEXT_WINDOW)}")

    # Use num_ctx from options if provided, otherwise fall back to global
    if 'num_ctx' not in options:
        options['num_ctx'] = G_CONTEXT_WINDOW

    while True:
        ret_dict = {
            "reasoning": "",
            "content": "",
            "usage": {},
            "time_taken": 0,
        }
        
        try:
            stream = chat_with_ollama(
                messages=conv,
                model=the_model,
                stream=True,
                thinking=thinking,
                options=options,
            )

            print("Streaming response: ", end="")
            time_start = time.time()
            token_count = 0
            token_count_reasoning = 0
            token_count_content = 0
            reason_str = ""
            response_str = ""
            in_reasoning = True
            for chunk in stream:
                # print(f"\n--\nChunk received: {chunk}\n--\n")  # Debug print for each chunk

                token_count += 1

                if chunk.get('choices'):

                    delta = chunk['choices'][0].get('delta', {})

                    if delta.get('reasoning'):
                        print(delta['reasoning'], end="", flush=True)
                        reason_str += delta['reasoning']
                    if delta.get('content'):
                        if in_reasoning and delta.get('content'):
                            print("\n--- End of Reasoning, Start of Content ---")
                            in_reasoning = False
                        print(delta['content'], end="", flush=True)
                        response_str += delta['content']

                # handle Ollama style
                elif chunk.get('message'):
                    message = chunk['message']
                    if message.get('thinking'):
                        print(message['thinking'], end="", flush=True)
                        reason_str += message['thinking']
                    if message.get('content'):
                        if in_reasoning and message.get('content'):
                            print("\n--- End of Reasoning, Start of Content ---")
                            in_reasoning = False

                        print(message['content'], end="", flush=True)
                        response_str += message['content']
                    # update token counts based on whether we're in reasoning or content
                    if in_reasoning:
                        token_count_reasoning += 1
                    else:
                        token_count_content += 1

            print()  # Newline after stream

            # handle </think>
            if "</think>" in response_str:
                parts = response_str.rsplit("</think>", 1)
                reason_str += parts[0].strip()
                response_str = parts[1].strip()

                # recalculate token counts based on the split
                token_count_reasoning = round(len(reason_str) / 3.245)
                token_count_content = round(len(response_str) / 3.245)
                token_count = token_count_reasoning + token_count_content

                # update ret_dict usage
                ret_dict["usage"] = {
                    "reasoning_tokens": token_count_reasoning,
                    "content_tokens": token_count_content,
                    "total_tokens": token_count,
                }


            # update ret_dict with final values
            ret_dict["reasoning"] = reason_str
            ret_dict["content"] = response_str
            ret_dict["time_taken"] = time.time() - time_start
            ret_dict["generation_speed"] = token_count / (time.time() - time_start) if time.time() - time_start > 0 else 0

        except Exception as e:
            print(f"Error: {e}")
            return ret_dict  # Return on error to avoid infinite loop

        # Check for repeat if retry_on_repeat is enabled
        if not retry_on_repeat:
            break

        repeat_found = False
        
        # 1. Line-level loop detection (for conversational/action loops)
        lines = [line.strip() for line in response_str.split('\n') if line.strip()]
        for block_size in range(1, 6):  # Check repeating blocks of 1 to 5 lines
            if len(lines) < block_size * 3:
                continue
            for i in range(len(lines) - block_size * 3 + 1):
                block = lines[i:i+block_size]
                # Count occurrences of this block
                matches = sum(1 for j in range(len(lines) - block_size + 1) if lines[j:j+block_size] == block)
                if matches >= 4:  # If the same line block appears 4+ times, it's a loop
                    repeat_found = True
                    break
            if repeat_found:
                break
                
        # 2. Substring level loop detection (for strict character repetitions)
        if not repeat_found and len(response_str) >= 100:
            for chunk_size in [75, 100, 150]: 
                if len(response_str) < chunk_size * 4:
                    continue
                # Sample the response string with a sliding window
                for i in range(0, len(response_str) - chunk_size, chunk_size // 2):
                    chunk = response_str[i:i+chunk_size]
                    if response_str.count(chunk) >= 4:
                        repeat_found = True
                        break
                if repeat_found:
                    break

        if not repeat_found:
            break

        # Retry with varied temperature
        options['temperature'] = round(random.uniform(0.6, 1.5), 2)
        print(f"\n[!] Repeat detected, retrying with temperature {options['temperature']}...")

    return ret_dict


# --- Example Usage ---
if __name__ == "__main__":
    # Basic example
    conversation = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Say hello in an alien language:"}
    ]

    # OUTPUT MODELS
    models = ollama_get_models()
    print(f"Available models: {models}")
    
    # ret_dict = llm_nonstream(conversation, thinking=G_THINKING) # NON-STREAMING
    ret_dict = llm_stream(conversation, thinking=G_THINKING) # STREAMING

    # output reasoning and content separately for clarity
    print(f"\n--- Reasoning ---\n")
    print(ret_dict["reasoning"])
    print(f"\n--- Content ---\n")
    print(ret_dict["content"])

    # output token counts and timing info
    print(f"\n--- Token Counts and Timing Info ---\n")
    print(f"Estimated Reasoning Tokens: {ret_dict['usage'].get('reasoning_tokens', 'N/A')}")
    print(f"Estimated Content Tokens: {ret_dict['usage'].get('content_tokens', 'N/A')}")
    print(f"Estimated Total Tokens: {ret_dict['usage'].get('total_tokens', 'N/A')}")
    print(f"Total Time: {ret_dict.get('time_taken', 'N/A'):.2f} seconds")
    if ret_dict.get('time_taken', 0) > 0:
        print(f"Average Speed: {ret_dict['usage'].get('total_tokens', 0) / ret_dict['time_taken']:.2f} tokens/second")

    print("\n\n\n")
