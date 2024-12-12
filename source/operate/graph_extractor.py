import re
import asyncio
from typing import List
from collections import Counter, defaultdict
from tqdm.asyncio import tqdm as tqdm_async
from typing import Dict, Tuple
from source.prompt import PROMPTS, GRAPH_FIELD_SEP
from common.base import (
    BaseGraphStorage, 
    BaseVectorStorage, 
    BaseKVStorage,
)

from common.schema import (
    TextChunkSchema, 
    NodeSchema, 
    EdgeSchema,
    EntitySchema,
    RelationSchema,
)

from common.utils import (
    logger,
    clean_str,
    is_float_regex,
    compute_mdhash_id,
    format_to_openai_message,
    split_string_by_multi_markers
)

async def _handle_single_entity_extraction(
        record_attribute: list[str], # ['entity', 'Alex', 'person', 'description ...'] 
        chunk_key: str
    ) -> NodeSchema:
    

    if len(record_attribute) < 4 or '"entity"' not in record_attribute[0].lower():
        return None
    
    entity_name = clean_str(record_attribute[1])
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attribute[2].upper())
    entity_desc = clean_str(record_attribute[3])
    entity_source_id = chunk_key
    return dict(
        entity_name = entity_name,
        entity_type = entity_type,
        entity_desc = entity_desc,
        entity_source_id = entity_source_id
    )


async def _handle_single_relation_extraction(
        record_attribute: list[str], # ['relationship', 'Alex', 'Taylor', 'desctiption', 'keyword', 'weight edge']
        chunk_key: str
    ) -> EdgeSchema:
    
    if len(record_attribute) < 6 or '"relationship"' not in record_attribute[0].lower():
        return None
    
    source_node = clean_str(record_attribute[1].upper())
    target_node = clean_str(record_attribute[2].upper())
    if not source_node or not target_node: 
        return None
    edge_desc = clean_str(record_attribute[3])
    edge_keyword = clean_str(record_attribute[4])
    weight_edge = (
        float(record_attribute[-1] if is_float_regex(record_attribute[-1]) else 1.0)
    )
    edge_source_id = chunk_key

    return dict(
        source_node = source_node,
        target_node = target_node,
        edge_desc = edge_desc,
        edge_weight = weight_edge,
        edge_keyword = edge_keyword,
        edge_source_id = edge_source_id
    )


async def _hanle_entity_relation_summary():
    pass 

async def _merge_nodes_then_upsert(
    entity_name: str,
    list_entity: List[NodeSchema],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict
) -> NodeSchema:
    """
    Get current node from graph based on the specific entity_name
    Get list of available nodes from graph after merge current node with list nodes
    Args: 

    Return:
        list nodes after merged
    """
    already_entity_types = []
    already_entity_ids = []
    already_entity_descs = []

    already_node: dict = await knowledge_graph_inst.get_node(node_id=entity_name)
    # get infor from current node
    if already_node is not None:
        already_entity_types.append(already_node['entity_type'])
        already_entity_descs.append(already_node['entity_desc'])
        already_entity_ids.extend(
            split_string_by_multi_markers(
                content=already_node['entity_source_id'],
                markers=[GRAPH_FIELD_SEP]
            )
        )
    
    # start merge
    entity_type = sorted(
        Counter(
            [dp['entity_type'] for dp in list_entity] + already_entity_types 
        ).items(),
        key=lambda x: x[1],
        reverse=True
    )[0][0]
    entity_desc = GRAPH_FIELD_SEP.join(set([dp['entity_dsc'] for dp in list_entity] + already_entity_descs))
    entity_id = GRAPH_FIELD_SEP.join(set([dp['entity_source_id'] for dp in list_entity] + already_entity_ids))
    entity_desc_summary = await _hanle_entity_relation_summary()

    # update node
    node_data = dict(
        entity_type = entity_type,
        entity_desc = entity_desc_summary,
        entity_source_id = entity_id
    ) 

    # update to graph
    await knowledge_graph_inst.upsert_node(node_id=entity_name, node_data=node_data)

    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        source_node_id: str,
        target_node_id: str,
        edges_data: List[dict],
        knowledge_graph_inst: BaseGraphStorage,
        global_config: dict):
    pass


async def _process_single_content(
    chunk_key_dp: Tuple[str, TextChunkSchema],
    global_config: Dict
) -> Tuple[Dict, Dict]:

    use_llm_func: callable = global_config['llm_model_func'] # llm
    entity_extract_max_gleaning: int = global_config['entity_extract_max_gleaning'] # num of iterations to extract entity
    entity_extract_prompt = PROMPTS['entity_extraction']
    context_base = dict(
        tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER'], # <|>
        record_delimiter = PROMPTS['DEFAULT_RECORD_DELIMITER'], # ##
        completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DEMILITER'],
        entity_types = PROMPTS['DEFAULT_ENTITY_TYPES'],
    )
    continue_prompt = PROMPTS['entity_continue_extraction']
    if_loop_prompt = PROMPTS['entity_if_loop_extraction'] 

    already_processed = 0
    already_entities = 0
    already_relations = 0


    chunk_key, chunk_dp = chunk_key_dp[0], chunk_key_dp[1]
    content = chunk_dp['content']
    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content) # prompt to extract entity from content
    final_response = await use_llm_func(hint_prompt) # use llm to extract entities provided based prompt 
    
    history: list[dict] = format_to_openai_message(prompt=hint_prompt, response=final_response) # convert to openai message
    for now_glean_index  in range(entity_extract_max_gleaning): # loop to glean entities
        response_glening = await use_llm_func(continue_prompt, history_message = history) # continues glean, after return the missing entities 
        history += format_to_openai_message(prompt=continue_prompt, response=response_glening) 
        final_response += response_glening

        if now_glean_index == entity_extract_max_gleaning - 1:
            break

        if_loop_result: str = await use_llm_func(if_loop_prompt, message_history = history)
        if "yes" not in if_loop_result.lower(): # no need to extract entity anymore if llm return 'no'
            break

    records: list[str] = split_string_by_multi_markers(
        content=final_response, 
        markers=[context_base['complete_delimiter'], context_base['record_delimiter']]
    )

    maybe_nodes: dict[str, list[NodeSchema]] = defaultdict(list)
    maybe_edges: dict[tuple[str, str], list[EdgeSchema]] = defaultdict(list)

    for record in records:
        record = re.search(r"\((.*)\)", record) # see in 'entity_extraction' PROMPT to understand 
        if record is None:
            continue

        record = record.group(1)
        record_attribute: list[str] = split_string_by_multi_markers(
            content=record,
            markers=[context_base['tuple_delimiter']]
        ) # see in 'entity_extraction' PROMPT to understand 

        if_entities: NodeSchema = await _handle_single_entity_extraction(
            record_attribute=record_attribute,
            chunk_key=chunk_key)
        
        if if_entities is not None:
            maybe_nodes[if_entities['entity_name']].append(if_entities)
            continue
        
        if_relations: EdgeSchema = await _handle_single_relation_extraction(
            record_attribute=record_attribute,
            chunk_key=chunk_key)

        if if_relations is not None:
            maybe_edges[(if_entities['source_node'], if_entities['target_node'])].append(if_entities)
    
    already_processed += 1
    already_entities += len(maybe_nodes)
    already_relations += len(maybe_edges)
    now_ticks = PROMPTS["process_tickers"][
        already_processed % len(PROMPTS["process_tickers"])
    ]
    print(
        f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
        end="",
        flush=True,
    )
    return dict(maybe_nodes), dict(maybe_edges)


async def extract_entities(
    global_config: dict,
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationship_vdb: BaseVectorStorage,
) -> BaseGraphStorage:
    

    # get all node and relations from chunks ==============================
    results = []
    ordered_chunks = list(chunks.items())

    for result in tqdm_async(
        iterable=asyncio.as_completed([_process_single_content(chunk_key_dp=chunk, 
                                                               global_config=global_config) 
                                                               for chunk in ordered_chunks]),
        desc="Extracting entities from chunks",
        unit="chunk",
        total=len(ordered_chunks)
    ):
        results.append(await result)

    # a entity name can correspond to multiple nodes
    maybe_nodes: dict[str, list[NodeSchema]]= defaultdict(list)
    # a (src_node, tgt_node) can correspond to multiple egdes
    maybe_edges: dict[str, list[EdgeSchema]] = defaultdict(list)

    for node, edge in results:
        for k, v in node.items():
            maybe_nodes[k].extend(v)
        
        for k, v in edge.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    
    
    # merge entities and relationships ====================================================
    logger.info('Inserting entities into storage...') 
    all_entities_data: list[NodeSchema] = [] # for entities
    for result in tqdm_async(
        iterable=asyncio.as_completed(
            _merge_nodes_then_upsert(entity_name=k, list_entity=v, 
                                     knowledge_graph_inst=knowledge_graph_inst, 
                                     global_config=global_config) # merge nodes after upsert nodes to graph
            for k, v in maybe_nodes
        ),
        desc="Inserting entities",
        unit="entity",
        total=len(maybe_nodes)
    ):
        all_entities_data.append(await result)
    
    if not len(all_entities_data):
        logger.warning(
            "Didn't extract any entities, maybe your LLM is not working"
        )
        return None

    logger.info('Inserting relationships into storage...')
    all_relations_data: list[EdgeSchema] = [] # for relationships
    for result in tqdm_async(
        iterable=asyncio.as_completed(
            _merge_edges_then_upsert(source_node_id=k[0], target_node_id=k[1], edges_data=v, 
                                     knowledge_graph_inst=knowledge_graph_inst,
                                     global_config=global_config) # merge edges after upsert edges to graph
            for k, v in maybe_edges
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relation"
    ):
        all_relations_data.append(await result)
    
    if not len(all_relations_data):
        logger.warning(
            "Didn't extract relationships, mayby your LLM is not working"
        )
        return None

    # store entities and relationships to vector database ======================================
    if entity_vdb is not None: # for entity
        data_for_vdb: EntitySchema = {
            compute_mdhash_id(content=dp['entity_name'], prefix="ent-"): {
                "entity_name": dp['entity_name'],
                "content": dp['entity_name'] + dp['entity_desc']
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data=data_for_vdb)

    if relationship_vdb is not None: # for relationship
        data_for_vdb: RelationSchema = {
            compute_mdhash_id(content=dp['target_node'] + dp['source_node'], prefix="rel-"): {
                "content": dp['edge_keyword'] + dp['source_node'] + dp['target_node'] + dp['edge_desc'],
                "source_node": dp['source_node'],
                "target_node": dp['target_node'],
            }
            for dp in all_relations_data
        }
        await relationship_vdb.upsert(data=data_for_vdb)

    return knowledge_graph_inst

