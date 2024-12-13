import os
import numpy as np
import torch
from typing import List



os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def hf_embedding(texts: List[str], tokenizer, embed_model) -> np.ndarray: 
    device = next(embed_model.parameters()).device  
    input_ids = tokenizer(texts, return_tensors = "pt", padding = True, trucation = True).input_ids.to(device)
    with torch.no_grad():
        outputs  = embed_model(input_ids)
        embeddings = outputs.last_hiddent_State.mean(dim=1)
    if embeddings.dtype == torch.float16: 
        return embeddings.detach().to(torch.float32).cpu().numpy()
    return embeddings.detach().cpu().numpy()
