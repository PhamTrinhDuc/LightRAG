import asyncio
import os
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, asdict, field
from tqdm.asyncio import tqdm as tqdm_async
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


@dataclass
class LightRAG:

    config = ConfigParams()

    def __post_init__(self):
        log_file: str = os.path.join("log", self.config.working_dir, "lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info("Logger initialized for working directory: %s", self.config.working_dir)
        
        # mappping storage name to storage class
        self.json_kv_storage_cls: Type[BaseKVStorage] = self.config.storage_classes['JsonKVStorage']
        self.vector_storage_cls: Type[BaseVectorStorage] = self.config.storage_classes['NanoVectorDBStorage']
        self.graph_storage_cls: Type[BaseGraphStorage] = self.config.storage_classes['NetworkXStorage'] 

        self.full_docs_kv = self.json_kv_storage_cls(
            namespace="full_docs",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
        )        
        self.text_chunks_kv = self.json_kv_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
        )

        self.entities_vdb = self.vector_storage_cls(
            namespace="entities",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
            meta_fields={"entity_name"}
        )

        self.relationships_vdb = self.vector_storage_cls(
            namespace="relationships",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
            meta_fields={"src_id", "tgt_id"}
        )

        self.chunks_vdb = self.vector_storage_cls(
            namespace="chunks",
            global_config=asdict(self.config),
            embedding_func=self.config.embedding_func,
        )

    async def ainsert(self, data: Union[str, List[str]]):
        update_storage = False
        try:
            if isinstance(data, str):
                data = [data]
            # 1. Lưu trữ toàn bộ văn bản trong data vào full_docs_kv - class: JsonKVStorage
            new_docs = {
                compute_mdhash_id(content=text.strip(), prefix="doc-"): {"content": text.strip()}
                for text in data
            } 
            # lọc ra những key chưa có trong new docs dựa vào data đã có
            new_key_in_docs = await self.full_docs_kv.filter_keys(data=new_docs)
            # lấy ra những data chưa có trong new docs dựa vào key đã lọc
            new_docs = {k: v for k, v in new_docs.items() if k in new_key_in_docs}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return 
            await self.full_docs_kv.upsert(data=new_docs)

            update_storage = True
            logger.info(f"[New docs] inserting {len(new_docs)} docs")

            # 2. Tách văn bản thành các phần nhỏ hơn và lưu trữ vào text_chunks_storagekv - class: JsonKVStorage
            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                iterable=new_docs.items(), 
                desc="Chunking docs", 
                unit="doc"):
                chunks = {
                    compute_mdhash_id(content=doc['content'], prefix="chunk-"): {
                        **dp, "full_doc_id": doc_key
                    }
                    for dp in chunking_by_token_size(
                        content=doc['content'],
                        max_token_size=self.config.chunk_token_size,
                        overlap_token_size=self.config.chunk_overlap_token_size,
                        tiktoken_model_name=self.config.tiktoken_model_name
                    )
                }
                inserting_chunks.update(chunks)
            new_key_in_chunks = await self.text_chunks_kv.filter_keys(data=inserting_chunks)
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k not in new_key_in_chunks}
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            await self.text_chunks_kv.upsert(data=inserting_chunks)
            # Thêm các chunk vào vector database
            await self.chunks_vdb.upsert(data=inserting_chunks)
            logger.info(f"[New chunks] inserting {len(inserting_chunks)} chunks")

            # 3. Trích xuất các thực thể từ các chunks và lưu trữ vào entities_vdb - class: NanoVectorDBStorage



        except Exception as e:
            pass

