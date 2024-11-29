import os
import copy
import aiohttp
import aiobotocore
import base64
import struct
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Any, Literal
import ollama

from groq import (
    AsyncGroq,
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from openai import (
    AsyncOpenAI, 
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from common.base import BaseKVStorage
from common.utils import compute_args_has


os.environ["TOKENIZERS_PARALLELISM"] = "false"

@retry(
    stop=stop_after_attempt(3), # dừng sau 3 lần thử
    wait=wait_exponential(multiplier=1, min=4, max=10), # thời gian chờ giữa các lần thửtăng theo cấp số nhân
    retry=retry_if_exception_type(RateLimitError | APIConnectionError | Timeout) # thử lại nếu gặp lỗi RateLimitError, APIConnectionError, Timeout
)
async def openai_complete_if_cache(
    prompt: str,
    model: Literal["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"] = "gpt-4o-mini",
    system_prompt: str = None,
    history_messages: List =  [],
    base_url: str = None,
    api_key: str = None,
    **kwargs
) -> str:
    
    """
    Hàm này thực hiện yêu cầu response từ OpenAI API với khả năng kiểm tra và sử dụng cache.
    Args:
        model: Tên của mô hình OpenAI để sử dụng.
        prompt: Văn bản yêu cầu từ người dùng.
        system_prompt (str, optional): Văn bản hệ thống để thiết lập ngữ cảnh. Mặc định là None.
        history_messages (List, optional): Danh sách các tin nhắn lịch sử. Mặc định là danh sách rỗng.
        base_url (str, optional): URL cơ sở cho OpenAI API. Mặc định là None.
        api_key (str, optional): Khóa API để xác thực với OpenAI. Mặc định là None.
        **kwargs: Các tham số bổ sung khác.
    Returns:
        str: Văn bản hoàn thành từ OpenAI API hoặc từ cache nếu có.
    """
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    openai_async_client = (
        AsyncOpenAI() if not base_url else AsyncOpenAI(base_url=base_url)
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", 
                         "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user",
                    "content": prompt})
    

    hashing_kv: BaseKVStorage = kwargs.get("hashing_kv", None)
    if hashing_kv is not None:
        id_hash = compute_args_has(model, messages)
        if_cache_return  = await hashing_kv.get_by_id(id_hash)
        if if_cache_return is not None:
            return if_cache_return['response']
        
    response = await openai_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {id_hash: {"response": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content

@retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min = 4, max = 10),
      retry=retry_if_exception_type(RateLimitError | APIConnectionError | Timeout)  
)
async def groq_complete_if_cache(
    prompt: str,
    system_prompt: str = None,
    history_message: List = [],
    base_url: str = None,
    api_key: str = None,
    model: Literal["llama3-70b-8192", 
                   "llama-3.1-70b-versatile", 
                   "llava-v1.5-7b-4096-preview", 
                   "gemma2-9b-it", 
                   "mixtral-8x7b-32768",] = "llama-3.1-70b-versatile",
    **kwargs,
) -> str:
    if api_key: 
        os.environ["GROQ_API_KEY"] = api_key
    
    groq_async_client = AsyncGroq() if not base_url else AsyncGroq(base_url=base_url)

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.extend(history_message)
    if prompt:
        messages.append({
            "role": "user",
            "content": prompt
        })
    hashing_kv: BaseKVStorage = kwargs.get("hasing_kv", None)
    if hashing_kv is not None:
        id_hash = compute_args_has(model, messages)
        if_cache_return = await hashing_kv.get_by_id(id=id_hash)
        if if_cache_return is not None:
            return if_cache_return['response']
    
    response = await groq_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            data={id_hash: {"response": response.choices[0].message.content, "model": model}}
        )

    return response.choices[0].message.content
    

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError | APIConnectionError | Timeout)
)
async def aruze_openai_complete_if_cache(
    model,
    prompt,
    system_prompt: str = None,
    history_messages: List = [],
    base_url: str = None,
    api_key: str = None,
    **kwargs,
) -> str:
    
    if api_key:
        os.getenv["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["ARUZE_OPENAI_ENDPOINT"] = base_url

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("ARUZE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    messages = []
    messages.append({"role": "system", 
                     "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user",
                     "content": prompt})
    
    hashing_kv: BaseKVStorage = kwargs.get("hashing_kv", None)
    if hashing_kv is not None:
        id_hash = compute_args_has(model, messages)
        if_return_cache = hashing_kv.get_by_id(id_hash)
        if if_return_cache is not None:
            return if_return_cache['response']
    
    response = openai_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {id_hash: {"response": response.choices[0].message.content, "model": model}}
        )
    
    return response.choices[0].message.content


@lru_cache(maxsize=1)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt: str = None,
    history_message: List = [],
    **kwargs
) -> str:
    
    hf_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model,
        device_map = "auto",
        trust_remote_code = True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model,
        device_map = "auto",
        trust_remote_code = True
    )

    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    messages =[]
    messages.append({"role": "system",
                     "content": system_prompt})
    messages.extend(history_message)
    messages.append({"role": "user",
                     "content": prompt})
    
    hashing_kv: BaseKVStorage = kwargs.get("hashing_kv", None)
    if hashing_kv is not None:
        id_hash = compute_args_has(model, messages)
        if_cache_return = hashing_kv.get_by_id(id_hash)
        if if_cache_return is not None:
            return if_cache_return['response']
    
    input_prompt = ""
    try: 
        input_prompt = hf_tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]['role'] == 'system':
                messages[1]['content'] = (
                    "<system>" + messages[0]['content'] + 
                    "</system>\n" + messages[1]['content']
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False, 
                    add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for mgid in range(len_message):
                input_prompt = (
                    input_prompt 
                    + "<" + ori_message[mgid]['role'] + ">" 
                    + "<" + ori_message[mgid]['content'] + ">\n"
                )
    
    input_ids = hf_tokenizer(
        text=input_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(**inputs, max_new_tokens=512, num_return_sequences=1, early_stopping=True)

    response_text = hf_tokenizer.decode(
        output[0][len(inputs['input_ids'][0]): ], skip_special_tokens=True
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {id_hash: {"response": response_text, "model": model}}
        )
    return response_text


async def ollama_model_if_cache(
    model, 
    prompt, 
    system_prompt: str = None,
    history_messages: List = [],
    **kwargs
) -> str:
    kwargs.pop("max_token", None)
    kwargs.pop("resposne_format", None)
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout)

    messages = []
    if system_prompt:
        messages.append({"role": "system", 
                         "content": system_prompt})
    
    messages.extend(history_messages)
    messages.append({"role": "user", 
                     "content": prompt})
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None: 
        id_has = compute_args_has(model, messages)
        if_return_cache = hashing_kv.get_by_id(id_has)
        if if_return_cache is not None:
            return if_return_cache['response']
    
    response = ollama_client.chat(model=model, messages=messages, **kwargs)['message']['content']

    if hashing_kv is not None:
        await hashing_kv.upsert({
            {id_has: {"response": response, "model": model}}
        })
    return response
