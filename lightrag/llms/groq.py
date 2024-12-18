import os
from typing import List, Literal

from groq import (
    AsyncGroq,
    APIConnectionError,
    RateLimitError,
    Timeout,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from lightrag.base import BaseKVStorage
from lightrag.utils import compute_args_hash



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
        id_hash = compute_args_hash(model, messages)
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
    