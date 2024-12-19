import asyncio
import os
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from nano_vectordb import NanoVectorDB
from typing import Dict, Any
from lightrag.utils import logger
from lightrag.base import BaseVectorStorage
from lightrag.schemas import EntitySchema, RelationSchema
from lightrag.utils import compute_mdhash_id

@dataclass
class NanoVectorStorage(BaseVectorStorage):
    # working_dir: str
    cosine_threshold: float = 0.2
    max_batch_size: int = 8

    def __post_init__(self):
        self._vector_storage_path = os.path.join(
            self.global_config['working_dir'], f"vdb_{self.namespace}.db"
        )
        self._max_batch_size = self.max_batch_size or 8
        self._client = NanoVectorDB(
            embedding_dim=self.embedding_func.embedding_dim, 
            storage_file=self._vector_storage_path
        )
        self.cosine_threshold = self.cosine_threshold or 0.2

    async def upsert(self, data: Dict[str, EntitySchema | RelationSchema]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            }
            for k, v in data.items()
        ]

        contents = [v['content'] for v in data.values()]
        batches = [contents[i: i + self._max_batch_size]
                        for i in range(0, len(data), self._max_batch_size)]
        
        async def wrapped_task(batch: list):
            result = self.embedding_func(batch)
            progess_bar.update(n=1)
            return result
        
        embedding_tasks = [wrapped_task(batch) for batch in batches] # no run
        progess_bar = tqdm_async(
            total=len(embedding_tasks),
            desc="Generating vector embeddings",
            unit="batch"
        )

        embedding_lists = asyncio.gather(*embedding_tasks) # run tasks

        embeddings = np.concatenate(embedding_lists)
        if len(embedding_lists) == len(list_data):
            for idx, data in enumerate(list_data):
                data["__vector__"] = embeddings[idx]
            results = self._client.upsert(datas=list_data)
            # return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        embedding_vector = self.embedding_func([query])[0]
        results = self._client.query(query=embedding_vector,
                                     top_k=top_k,
                                     better_than_threshold=self.cosine_threshold)
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"] } for dp in results
        ]
        return results
    
    @property
    def client_storage(self):
        return getattr(self._client, "NanoVectorDB__storage")
    
    async def delete_entity(self, entity_name: str):
        try:
            id_entity = [compute_mdhash_id(content=entity_name, prefix="ent-")]
            if self._client.get(ids=id_entity):
                self._client.delete(ids=id_entity)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, src_entity: str, tgt_entity: str):
        try:
            relation_id = [compute_mdhash_id(content=src_entity + tgt_entity, prefix="rel-")]
            if self._client.get(ids=relation_id):
                self._client.delete(ids=relation_id)
                logger.info(f"Relation has src_entity: {src_entity} and tgt_entity: {tgt_entity} have been deleted.")
            else:
                logger.info(f"No relation found with src_entity: {src_entity} and tgt_entity: {tgt_entity}")
        except Exception as e:
            logger.error(f"Error while deleting relation with src_entity: {src_entity} and tgt_entity: {tgt_entity}")

    def index_done_callback(self):
        self._client.save()