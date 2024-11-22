import asyncio
import os
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, asdict, field
from functools import partial

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
    BaseGraphStorage
)

from utils.utilities import (
    set_logger, 
    logger, 
    compute_mdhash_id,
)

from config import Settings


@dataclass
class LightRAG:

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorBDStorage")
    graph_storage: str = field(default="NetworkXStorage")

    
    def __post_init__(self):
        log_file: str = os.path.join("Log", Settings.working_dir, "lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info("Logger initialized for working directory: %s", Settings.working_dir)
        
        self.kv_storage_cls: Type[BaseKVStorage] = self._get_storage_class()[self.kv_storage]
        self.vector_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[self.vector_storage]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[self.graph_storage] 



    def _get_storage_class(self) -> Type[None]:
        return {
            "JsonKVStorage": JsonKVStorage,
            "NanoVectorDBStorage": NanoVectorStorage,
            "NetworkXStorage": NetworkXStorage,
        }