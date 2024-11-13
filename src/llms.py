import os
import copy
import aiohttp
import aiobotocore
import base64
import struct
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Any
import numpy as np
import ollama

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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.base import BaseKVStorage
from utils.utilities import compute_args_has, wrap_embedding_func_with_attrs


os.environ["TOKENIZERS_PARALLELISM"] = "false"

@retry(
    stop=stop_after_attempt(3), # dừng sau 3 lần thử
    wait=wait_exponential(multiplier=1, min=2, max=10), # thời gian chờ giữa các lần thửtăng theo cấp số nhân
    retry=retry_if_exception_type(RateLimitError, APIConnectionError, Timeout) # thử lại nếu gặp lỗi RateLimitError, APIConnectionError, Timeout
)
async def openai_complete_if_cache(
    model,
    prompt,
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

    hashing_kv: BaseKVStorage = kwargs.get("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", 
                         "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user",
                    "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_has(model, messages)
        if_cache_return  = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return['return']
        
    response = await openai_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content