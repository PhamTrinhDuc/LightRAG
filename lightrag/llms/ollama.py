from typing import List
import ollama
from lightrag.base import BaseKVStorage
from lightrag.utils import compute_args_hash



async def ollama_model_if_cache(
    model: str, 
    prompt: str, 
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
        id_has = compute_args_hash(model, messages)
        if_return_cache = hashing_kv.get_by_id(id_has)
        if if_return_cache is not None:
            return if_return_cache['response']
    
    response = ollama_client.chat(model=model, messages=messages, **kwargs)['message']['content']

    if hashing_kv is not None:
        await hashing_kv.upsert({
            {id_has: {"response": response, "model": model}}
        })
    return response
