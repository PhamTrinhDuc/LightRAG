from dataclasses import dataclass, field
from datetime import datetime
from lightrag.utils import EmbeddingFunc, convert_response_to_json, logger
from lightrag.storage import JsonKVStorage, NanoVectorStorage, NetworkXStorage
from lightrag.llms import openai_complete_if_cache
from lightrag.embeddings import openai_embedding


@dataclass
class ConfigParams:
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    
    storage_classes: dict = field(
        default_factory=lambda: {
            "JsonKVStorage": JsonKVStorage,
            "NanoVectorDBStorage": NanoVectorStorage,
            "NetworkXStorage": NetworkXStorage
        }
    )

    # directory
    working_dir: str = field(
        default_factory= lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    #text chunk 
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 200
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_token: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimentions": 1536,
            "num_walks": 10,
            "walk_length": 10,
            "windown_size": 2,
            "iterations": 3,
            "random_seed": 3
        }
    )
    # embedding model
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM API
    llm_model_func: callable = openai_complete_if_cache
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True
    
    # extention
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json: callable = convert_response_to_json