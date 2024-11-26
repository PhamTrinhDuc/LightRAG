
from utils.utilities import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
)



def chunking_by_token_size(content: str, 
                           max_token_size: int = 1024, 
                           overlap_token_size: int = 128, 
                           tiktoken_model_name: str = "gpt-4o-mini"):
    tokens = encode_string_by_tiktoken(content=content, model_name=tiktoken_model_name)
    results = []
    for idx, start in enumerate(range(0, len(tokens), max_token_size - - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(
            tokens=tokens[start: start + max_token_size],
            model_name=tiktoken_model_name
        )

        results.append({
            "tokens": chunk_content, 
            "content": chunk_content.strip(),
            "chunk_order_index": idx
        })
    return results


def extract_entities():
    pass
def local_query():
    pass
def global_query():
    pass
def hybrid_query():
    pass
def naive_query():
    pass
