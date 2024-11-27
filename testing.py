from src.llms import openai_complete_if_cache, openai_embedding
from src.operate import chunking_by_token_size
from utils.utilities import split_string_by_multi_markers
from src.lightrag import LightRAG
import os
import asyncio


async def main():
    # test async function openai gpt4o-mini
    # response = await openai_complete_if_cache(
    #     model='gpt-4o-mini',
    #     prompt='What is the capital of France?',
    # )
    # print(response) => text response
    #=====================================================================================
    # test async function openai embedding  
    # response = await openai_embedding(texts = ["hello", "hi"])
    # print(response.shape) # => shape = (2, 1536)
    # ====================================================================================
    # test lightrag
    # rag = LightRAG()
    # res = await rag.full_docs.filter_keys(data=['a', 'b', 'c'])
    # print(res)
    # ====================================================================================
    # test tokenizer 
    content = """while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. 
    It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.
    Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." 
    Taylor said, their voice quieter, "It could change the game for us. For all of us."""

    results = chunking_by_token_size(content=content)
    print(results)
    # ====================================================================================

if __name__ == "__main__":
    asyncio.run(main())
    # main()


