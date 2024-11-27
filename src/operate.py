
from utils.schema import TextChunkSchema
from src.prompt import PROMPTS
from src.storage import BaseGraphStorage, BaseVectorStorage, BaseKVStorage
from utils.utilities import (
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


def extract_entities(
        chunks: dict[str, TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationship_vdb: BaseVectorStorage,
        global_config: dict
):
    use_llm_func: callable = global_config['llm_model_func'] # llm
    entity_extract_max_gleaning: int = global_config['entity_extract_max_gleaning'] # num of iterations to extract entity

    entity_extract_prompt = PROMPTS['entity_extraction']
    context_base = dict(
        tuple_delimiter = PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimier = PROMPTS['DEFAULT_RECORD_DELIMITER'],
        complete_delimiter = PROMPTS['DEFAULT_COMPLETION_DEMILITER'],
        enity_type = PROMPTS['DEFAULT_ENTITY_TYPES'],
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








def local_query():
    pass
def global_query():
    pass
def hybrid_query():
    pass
def naive_query():
    pass
