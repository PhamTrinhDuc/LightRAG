from lightrag.graphstore import NetworkXStorage
from lightrag.kvstore import JsonKVStorage
from lightrag.vectorstore import NanoVectorStorage
from lightrag.base import (
    QueryParam,
    BaseGraphStorage,
    BaseVectorStorage,
    BaseKVStorage
)
from lightrag.schemas import TextChunkSchema
from lightrag.utils import (
    logger,
    handle_cache,
    compute_args_hash
)
from lightrag.prompt import PROMPTS



async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relation_vdb: BaseVectorStorage,
    text_chunk_vdb: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: JsonKVStorage 
) -> str:
    
    # handle cache response
    llm_func: callable =  global_config['llm_model_func']
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )

    if cached_response is not None:
        return cached_response

    # num of examples prompt to extract (high level keywords, low level keywords) 
    example_numbers = global_config['addon_params']
    if example_numbers and example_numbers < len(PROMPTS['keywords_extraction_examples']):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][:int(example_numbers)]
        )
    else:
        examples = "\n".join(PROMPTS['keywords_extraction_examples'])
    
    language = PROMPTS['addon_params'].get(
        "language", PROMPTS['DEFAULT_LANGUAGE']
    )

    if query_param.mode not in ["local", "global", "hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS['fail_response']
    

    