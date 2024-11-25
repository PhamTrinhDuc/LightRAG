from src.llms import openai_complete_if_cache
import os
import asyncio


async def main():
    response = await openai_complete_if_cache(
        model='gpt-4o-mini',
        prompt='What is the capital of France?',
    )
    return response

res = asyncio.run(main())
print(res)
