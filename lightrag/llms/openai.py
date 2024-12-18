import os
from typing import List, Literal

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
from lightrag.base import BaseKVStorage
from lightrag.utils import compute_args_has


os.environ["TOKENIZERS_PARALLELISM"] = "false"

@retry(
    stop=stop_after_attempt(3), # dừng sau 3 lần thử
    wait=wait_exponential(multiplier=1, min=4, max=10), # thời gian chờ giữa các lần thửtăng theo cấp số nhân
    retry=retry_if_exception_type(RateLimitError | APIConnectionError | Timeout) # thử lại nếu gặp lỗi RateLimitError, APIConnectionError, Timeout
)
async def openai_complete_if_cache(
    prompt: str,
    model: Literal["gpt-4o-mini", 
                   "gpt-4o", 
                   "gpt-3.5-turbo", 
                   "gpt-4-turbo"] = "gpt-4o-mini",
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
