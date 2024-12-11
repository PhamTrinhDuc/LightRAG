import asyncio
import os
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, asdict, field
from tqdm.asyncio import tqdm as tqdm_async
from functools import partial
from typing import Union, List, Dict, Any

from config import ConfigParams

from operate.query import (
    local_query,
    naive_query,
    hybrid_query,
    global_query
)

from source.operate.graph_extractor import (
    extract_entities, 
    chunking_by_token_size
)

from common.base import (
    BaseKVStorage, 
    BaseVectorStorage,
    BaseGraphStorage,
    StorageNameSpace
)

from common.utils import (
    logger,
    set_logger, 
    compute_mdhash_id,
)

@dataclass
class LightRAG:

    config = ConfigParams()

    def __post_init__(self):
        log_file: str = os.path.join("log", self.config.working_dir, "lightrag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        logger.info("Logger initialized for working directory: %s", self.config.working_dir)
        
        if not os.path.exists(self.config.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.config.working_dir)
            
        # mappping storage name to storage class
        self.json_kv_storage_cls: Type[BaseKVStorage] = self.config.storage_classes['JsonKVStorage']
        self.vector_storage_cls: Type[BaseVectorStorage] = self.config.storage_classes['NanoVectorDBStorage']
        self.graph_storage_cls: Type[BaseGraphStorage] = self.config.storage_classes['NetworkXStorage'] 

        # JSONKVStorage
        self.llm_response_cache = (
            self.json_kv_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self.config),
                embedding_func=None
            ) if self.enable_llm_cache else None
        )
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

        # Graph Storage
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self.config)
        )

        # Vectordb Storage
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
            
            # Insert chunk to vector database
            await self.chunks_vdb.upsert(data=inserting_chunks)
            logger.info(f"[New chunks] inserting {len(inserting_chunks)} chunks")

            # 3. Trích xuất các thực thể từ các chunks và lưu trữ vào entities_vdb - class: NanoVectorDBStorage
            maybe_new_kg = await extract_entities(
                chunks=inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationship_vdb=self.relationships_vdb,
                global_config=asdict(self.config),
            )
            if maybe_new_kg is not None:
                logger.info("")
                return
            
            self.chunk_entity_relation_graph = maybe_new_kg
            await self.full_docs_kv.upsert(data=new_docs)
            await self.text_chunks_kv.upsert(data=inserting_chunks)
        finally:
            if update_storage:
                self._insert_callback_done()


    async def _insert_callback_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs_kv,
            self.text_chunks_kv,
            self.llm_response_cache,
            self.chunks_vdb,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph
        ]:
            if storage_inst is not None:
                tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

            
