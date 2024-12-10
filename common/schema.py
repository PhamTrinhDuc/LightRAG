from pydantic import BaseModel, Field

class TextChunkSchema(BaseModel): # schema for data chunk to save in KVJson and VectorDB
    tokens: int = Field(description="token form of chunk")
    content: str = Field(description="chunk of document")
    full_doc_id: str = Field(description="")
    chunk_order_index: int = Field(description="")

class EntitySchema(BaseModel): # schema for entity to save in VectoDB
    entity_name: str = Field(description="entity name is extracted")
    content: str = Field(description="chunk contains the entity name")

class RelationSchema(BaseModel): # schem for relation to save in VectoDB
    src_id: str = Field(description="id of the source node entity")
    tgt_id: str = Field(description="id of the target node entity")
    content: str = Field(description="chunk contains the src and tgt node")

class NodeSchema(BaseModel): # schema for node to save in Network Graph
    entity_name: str = Field(description="")
    entity_type: str = Field(description="")
    entity_desc: str = Field(description="")
    entity_source_id: str = Field(description="")

class EdgeSchema(BaseModel): # schema for relationship save in Network Graph 
    source_node: str
    target_node: str
    edge_desc: str
    edge_keyword: str
    edge_weight: int
    edge_source_id: str