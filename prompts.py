from api_key import API_KEY
import os
import asyncio
from dask.distributed import Client, as_completed, get_client, wait
from mistralai import Mistral

async def query_mistral(prompt: str):
    model = "mistral-tiny"
    client = Mistral(api_key=API_KEY)

    response = await client.chat.stream_async(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = ""
    async for chunk in response:
        delta = chunk.data.choices[0].delta.content
        if delta is not None:
            output += delta
    return output


def run_async_query(prompt):
    return asyncio.run(query_mistral(prompt))