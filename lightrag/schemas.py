from typing import TypedDict


TextChunkSchema = TypedDict(  # schema for data chunk to save in KVJson and VectorDB
    "TextChunkSchema",
    {
        "tokens": int,
        "content": str, 
        "full_doc_id": str, 
        "chunk_order_index": int
    },
)

EntitySchema = TypedDict( # schema for entity to save in VectoDB
    "EntitySchema",
    {
        "entity_name": str,
        "content": str
    }
)

RelationSchema = TypedDict(
    "RelationSchema",
    {
        "source_node": str,
        "target_node": str,
        "content": str
    }
)

NodeSchema = TypedDict( # schema for node to save in Network Graph
    "NodeSchema",
    {
        "entity_name": str,
        "entity_type": str,
        "entity_desc": str,
        "entity_source_id": str
    }
)
EdgeSchema = TypedDict( # schema for relationship save in Network Graph 
    "EdgeSchema",
    {
        "source_node": str,
        "target_node": str,
        "edge_desc": str,
        "edge_keyword": str,
        "edge_weight": str,
        "edge_source_id": str
    }
)