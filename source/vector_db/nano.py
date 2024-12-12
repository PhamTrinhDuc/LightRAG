import asyncio
import os
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from nano_vectordb import NanoVectorDB
from typing import Dict, Any
from common.utils import logger
from common.base import BaseVectorStorage


@dataclass
class NanoVectorStorage(BaseVectorStorage):
    cosine_threshold: float = 0.2

    def __post_init__(self):
        self._vector_storage_path = os.path.join(
            self.global_config['working_dir'], f"vdb_{self.namespace}.db"
        )
        self._max_batch_size = self.global_config.get('max_batch_size', 8)
        self._client = NanoVectorDB(
            embedding_dim=self.embedding_func.embedding_dim, 
            storage_file=self._vector_storage_path
        )
        self.cosine_threshold = self.global_config.get('cosine_threshold', self.cosine_threshold)

    async def upsert(self, data: Dict[str, Dict[str, Any]]):
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
        
        embeddings_task = [self.embedding_func(batch) for batch in batches]
        embedding_list = []
        for task in tqdm_async(
            asyncio.as_completed(embeddings_task),
            total=len(embeddings_task),
            desc = "Genrerating embeddings",
            unit="batch"
        ):
            embedding_list.append(await task)
        embeddings = np.concatenate(embedding_list)
        for idx, value in enumerate(embedding_list):
            value['__vector__'] = embeddings[idx]
        results = await self._client.upsert(datas=list_data)
        return results
    
    async def query():
        pass

    async def delete_entity():
        pass

    async def delete_relation():
        pass

    def index_done_callback(self):
        pass