import asyncio
import os
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, asdict, field
from functools import partial
from typing import Union, List, Dict, Any

from .llms import (
    openai_complete_if_cache,
    openai_embedding
)

from .operate import (
    chunking_by_token_size, 
    extract_entities,
    local_query, 
    global_query, 
    hybrid_query, 
    naive_query
)

from.storage import (
    JsonKVStorage,
    NanoVectorStorage,
    NetworkXStorage
)

from utils.schema import (
    BaseKVStorage, 
    BaseVectorStorage,
    StorageNameSpace,
    BaseGraphStorage,
    EmbeddingFunc
)

from utils.utilities import (
    set_logger, 
    logger, 
    compute_mdhash_id,
)


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
    tiktoken_model_name: str = "gpt-40-mini"

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


@dataclass
class LightRAG:

    config = ConfigParams()

    def __post_init__(self):
        # log_file: str = os.path.join("log", self.config.working_dir, "lightrag.log")
        # set_logger(log_file)
        # logger.setLevel(self.log_level)
        # logger.info("Logger initialized for working directory: %s", self.config.working_dir)
        
        # mappping storage name to storage class
        self.json_kv_storage_cls: Type[BaseKVStorage] = self.config.storage_classes['JsonKVStorage']
        self.vector_storage_cls: Type[BaseVectorStorage] = self.config.storage_classes['NanoVectorDBStorage']
        self.graph_storage_cls: Type[BaseGraphStorage] = self.config.storage_classes['NetworkXStorage'] 

        self.full_docs = self.json_kv_storage_cls(
            namespace="full_docs",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
        ).filter_keys(data=['a', 'b'])

        self.text_chunks = self.json_kv_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
        )


    # def _get_storage_class(self) -> Type[BaseGraphStorage]:
    #     return {
    #         "JsonKVStorage": JsonKVStorage,
    #         "NanoVectorDBStorage": NanoVectorStorage,
    #         "NetworkXStorage": NetworkXStorage,
    #     }

    async def ainsert(self, data: Union[str, List[str]]):
        update_storage = False
        try:
            if isinstance(data, str):
                data = [data]
            new_docs = {
                compute_mdhash_id(content=text.strip(), prefix="doc-"): {"content": text.strip()}
                for text in data
            } # dict[str, dict[str, str]]
            new_key_in_docs = await self.full_docs.filter_keys()


        except Exception as e:
            pass

