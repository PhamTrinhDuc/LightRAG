from typing import TypedDict, TypeVar


T = TypeVar("T")

class TextChunkSchema(TypedDict): # schema for data chunk to save in KVJson and VectorDB
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int

class EntitySchema(TypedDict): # schem for entity to save in VectoDB
    entity_name: str
    content: str

class RelationSchema(TypedDict): # schem for relation to save in VectoDB
    src_id: str
    tgt_id: str
    content: str

class NodeSchema(TypedDict): # schema for node to save in Network Graph
    entity_name: str
    entity_type: str
    entity_desc: str
    entity_source_id: str

class EdgeSchema(TypedDict): # schema for relationship save in Network Graph 
    source_node: str
    target_node: str
    edge_desc: str
    edge_keyword: str
    edge_weight: int
    edge_source_id: str