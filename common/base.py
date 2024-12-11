from dataclasses import dataclass, field
from typing import Union, Generic, TypeVar, List, Dict, Tuple
from common.utils import EmbeddingFunc

T = TypeVar("T")

@dataclass
class StorageNameSpace:
    """
    StorageNameSpace là một lớp đại diện cho không gian lưu trữ với các cấu hình toàn cục.
    Attributes:
        namespace (str): Tên không gian lưu trữ.
        global_config (dict): Cấu hình toàn cục cho không gian lưu trữ.
    Methods:
        index_done_callback(): Hàm bất đồng bộ được gọi sau khi hoàn tất việc lập chỉ mục.
        query_done_callback(): Hàm bất đồng bộ được gọi sau khi hoàn tất việc truy vấn.
    """
    
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        "commit the storage operations after indexing"
        pass

    async def query_done_callback(self):
        "commit the storage operations after querying"
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError
    
    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError
    
    
@dataclass 
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        """Get all keys in current data"""
        raise NotImplementedError
    
    async def get_by_id(self, id: str) -> Union[T, None]:
        """Get data by id from current data"""
        raise NotImplementedError
    
    async def filter_keys(self, data: list[str]) -> set[str]:
        """Get keys in data that not in current data"""
        raise NotImplementedError
    
    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError
    

@dataclass
class BaseGraphStorage(StorageNameSpace):
    
    async def has_node(self, node_id: str) -> bool:
        """Check if node exsits from node_id"""
        raise NotImplementedError
    
    async def has_edge(self, src_node_id: str, tgt_node_id: str) -> bool:
        """Check if edge exists from src_node and tgt_node"""
        raise NotImplementedError
        
    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Get node from graph through node_id"""
        raise NotImplementedError
     
    async def get_degree_node(self, node_id: str) -> int:
        """Get degree node from Graph through node_id"""
        raise NotImplementedError
        
    async def get_degree_between_nodes(self, src_node_id: str, tgt_node_id) -> int:
        """Get degree between 2 nodes from Graph through src_node_id and tgt_node_id"""
        raise NotImplementedError
    
    async def get_edge(self, src_node_id: str, tgt_node_id: str)-> Union[dict, None]:
        """Get edge from graph through src_node_id and tgt_node_id"""
        raise NotImplementedError
    
    async def get_node_edges(self, node_id: str) -> List[Tuple[str, str]]:
        """Get edges of node_id from graph"""
        raise NotImplementedError
        
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Update a node to graph based on the specificed node_id"""
        raise NotImplementedError
        
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from graph based on the specified node_id"""
        raise NotImplementedError
    
    async def upsert_edge(self, src_node_id: str, tgt_node_id: str, edge_data: dict[str, str]) -> None:
        """Update a edge to graph based on the specificed src_node_id and tgt_node_id"""
        raise NotImplementedError
    
    async def delete_edge(self, src_node_id: str, tgt_node_id: str) -> None:
        """Delete a edge from graph based on the specified src_node_id and tgt_node_id"""
        raise NotImplementedError    

