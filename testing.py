from src.llms import openai_complete_if_cache, openai_embedding
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
    response = await openai_embedding(texts = ["hello", "hi"])
    print(response.shape) # => shape = (2, 1536)


if __name__ == "__main__":
    asyncio.run(main())


