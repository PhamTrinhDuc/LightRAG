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
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    openai_async_client = (
        AsyncOpenAI() if not base_url else AsyncOpenAI(base_url=base_url)
    )

