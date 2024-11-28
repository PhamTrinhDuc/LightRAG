import re
import asyncio
from collections import Counter, defaultdict
from tqdm.asyncio import tqdm as tqdm_async
from utils.schema import TextChunkSchema
from src.prompt import PROMPTS
from src.storage import BaseGraphStorage, BaseVectorStorage, BaseKVStorage
from utils.utilities import (
    clean_str,
    is_float_regex,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    format_to_openai_message,
    split_string_by_multi_markers
)



def chunking_by_token_size(content: str, 
                           max_token_size: int = 1024, 
                           overlap_token_size: int = 128, 
                           tiktoken_model_name: str = "gpt-4o-mini"):
    tokens = encode_string_by_tiktoken(content=content, model_name=tiktoken_model_name)
    # print("Tokens: ", tokens) # list numerical
    results = []
    for idx, start in enumerate(range(0, len(tokens), max_token_size - - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(
            tokens=tokens[start: start + max_token_size],
            model_name=tiktoken_model_name
        )
        # print("Chunk content: ", chunk_content) # list string
        results.append({
            "tokens": tokens, 
            "content": chunk_content.strip(),
            "chunk_order_index": idx
        })
    return results

async def _handle_single_entity_extraction(
        record_attribute: list[str], # ['entity', 'Alex', 'person', 'description ...'] 
        chunk_key: str):
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
        source_id = entity_source_id
    )

async def _handle_single_relation_extraction(
        record_attribute: list[str], # ['relationship', 'Alex', 'Taylor', 'desctiption', 'keyword', 'weight edge']
        chunk_key: str
):
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
        weight_edge = weight_edge,
        edge_keyword = edge_keyword,
        source_id = edge_source_id
    )

async def _merge_node_then_upsert():
    pass

async def _merge_edges_then_upsert():
    pass


async def extract_entities(
    global_config: dict,
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationship_vdb: BaseVectorStorage,
):
    
    use_llm_func: callable = global_config['llm_model_func'] # llm
    entity_extract_max_gleaning: int = global_config['entity_extract_max_gleaning'] # num of iterations to extract entity

    entity_extract_prompt = PROMPTS['entity_extraction']
    context_base = dict(
        tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter = PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter = PROMPTS['DEFAULT_COMPLETION_DEMILITER'],
        entity_types = PROMPTS['DEFAULT_ENTITY_TYPES'],
    )

    continue_prompt = PROMPTS['entity_continue_extraction']
    if_loop_prompt = PROMPTS['entity_if_loop_extraction'] 

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
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

        maybe_nodes: dict[str, list[dict]] = defaultdict(list)
        maybe_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for record in records:
            record = re.search(r"\((.*)\)", record) # see in 'entity_extraction' PROMPT to understand 
            if record is None:
                continue

            record = record.group(1)
            record_attribute: list[str] = split_string_by_multi_markers(
                content=record,
                markers=[context_base['tuple_delimiter']]
            ) # see in 'entity_extraction' PROMPT to understand 

            if_entities = await _handle_single_entity_extraction(record_attribute=record_attribute,
                                                                 chunk_key=chunk_key)
            
            if if_entities is not None:
                maybe_nodes[if_entities['entity_name']].append(if_entities)
                continue
            
            if_relations = await _handle_single_relation_extraction(
                record_attribute=record_attribute,
                chunk_key=chunk_key
            )

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
    
    results = []
    ordered_chunks = list(chunks.items())
    for result in tqdm_async(
        iterable=asyncio.as_completed([_process_single_content(chunk) for chunk in ordered_chunks]),
        desc="Extracting entities from chunks",
        unit="chunk",
        total=len(ordered_chunks)
    ):
        results.append(await result)
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    for node, edge in results:
        for k, v in node.items():
            maybe_nodes[k].extend(edge)
        
        for k, v in edge.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    
    




def local_query():
    pass
def global_query():
    pass
def hybrid_query():
    pass
def naive_query():
    pass
