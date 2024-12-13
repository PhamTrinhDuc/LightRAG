import os
from typing import List, Union
import numpy as np

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
from lightrag.utils import wrap_embedding_func_with_attrs


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@wrap_embedding_func_with_attrs(embedding_dim = 1536, max_token=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError | APIConnectionError |Timeout)
)
async def openai_embedding(
    texts: Union[str, List[str]],
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
