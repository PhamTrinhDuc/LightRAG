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
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import torch
from utils.utilities import wrap_embedding_func_with_attrs


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@wrap_embedding_func_with_attrs(embedding_dim = 1536, max_token=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError | APIConnectionError |Timeout)
)
async def openai_embedding(
    texts: List[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None 
) -> np.ndarray:
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    openai_async_client = (
        AsyncOpenAI() if not base_url else AsyncOpenAI(base_url=base_url)
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )

    return np.array([dp.embedding for dp in response.data])


async def hf_embedding(texts: List[str], tokenizer, embed_model) -> np.ndarray: 
    device = next(embed_model.parameters()).device  
    input_ids = tokenizer(texts, return_tensors = "pt", padding = True, trucation = True).input_ids.to(device)
    with torch.no_grad():
        outputs  = embed_model(input_ids)
        embeddings = outputs.last_hiddent_State.mean(dim=1)
    if embeddings.dtype == torch.float16: 
        return embeddings.detach().to(torch.float32).cpu().numpy()
    return embeddings.detach().cpu().numpy()


async def ollama_embedding(texts: List[str], embed_moded, **kwargs) -> np.ndarray:
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        embeddings = ollama_client.embeddings(model=embed_moded, prompt=text) 
        embed_text.append(embeddings['embedding'])
    
    return embed_text