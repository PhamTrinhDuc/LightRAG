
import os
import numpy as np
import ollama
from typing import List


os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def ollama_embedding(texts: List[str], embed_moded, **kwargs) -> np.ndarray:
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        embeddings = ollama_client.embeddings(model=embed_moded, prompt=text) 
        embed_text.append(embeddings['embedding'])
    
    return embed_text