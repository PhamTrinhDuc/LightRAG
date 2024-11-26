import asyncio
import html
import os
import networkx as nx
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from nano_vectordb import NanoVectorDB
from typing import List, Dict, Any
from config import Settings
from utils.utilities import load_json, write_json, logger
from utils.schema import BaseKVStorage, BaseVectorStorage, BaseGraphStorage

@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    Lớp JsonKVStorage kế thừa từ BaseKVStorage, sử dụng để lưu trữ dữ liệu dưới dạng JSON.
    Methods:
    - __post_init__(): Khởi tạo đối tượng, thiết lập đường dẫn file và tải dữ liệu từ file JSON.
    - index_done_callback(): Ghi dữ liệu hiện tại vào file JSON.
    - get_by_id(id: str) -> dict: Lấy dữ liệu theo ID.
    - get_by_ids(ids: List[str], fields: List[str] = None) -> List[dict]: Lấy dữ liệu theo danh sách ID, có thể chỉ định các trường cần lấy.
    - upsert(data: Dict[str, Dict[str, Any]]): Cập nhật hoặc thêm mới dữ liệu.
    - drop(): Xóa toàn bộ dữ liệu.
    """

    def __post_init__(self):
        working_dir: str = self.global_config['working_dir']
        self._file_name_data: str = os.path.join(working_dir, f"{self.namespace}.json")
        self._data_json: Dict[str, Dict[str, Any]]  = load_json(file_name=self._file_name_data)
        logger.info(f"Load KV {self.namespace} with {len(self._data_json)} data")
    
    async def index_done_callback(self):
        """Write data to json file after indexing"""
        write_json(json_obj=self._data_json, file_name=self._file_name_data)

    async def get_by_id(self, id: str) -> dict:
        """Get json data by id from current data"""
        return self._data_json.get(id, None)
    
    async def filter_keys(self, data: list[str]) -> set[str]:
        """Get keys in data that not in current data"""
        return set([key for key in data if key not in self._data_json])
    
    async def get_by_ids(self, ids: List[str], fields: List[str] = None) -> List[dict]:
        if fields is None:
            return [self.get_by_id(id) for id in ids]
        
        return [
            (
                {k: v for k, v in self._data_json[id].items() if k in fields}
                if self._data_json.get(id, None) else None
            )
            for id in ids
        ]

    async def upsert(self, data: Dict[str, Dict[str, Any]]):
        left_data = {k: v for k, v in data.items() if k not in self._data_json}
        self._data_json.update(left_data)
        return left_data
    
    async def drop(self):
        self._data_json = {}
    

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


class NetworkXStorage():
    pass

