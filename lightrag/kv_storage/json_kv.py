import os
from dataclasses import dataclass
from typing import List, Dict, Any
from lightrag.utils import load_json, write_json, logger
from lightrag.base import BaseKVStorage

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

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        left_data = {k: v for k, v in data.items() if k not in self._data_json}
        self._data_json.update(left_data)
    
    async def drop(self):
        self._data_json = {}
    