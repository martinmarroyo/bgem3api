"""A sample asynchronous client for the bgem3api"""
import json
import asyncio
import aiohttp
import time
from functools import partial

async def get_colbert_embedding(text: str, session):
    """Requests colBERT embeddings from the server."""
    url = "http://localhost:8000/embedding"
    data = {"text": text, "colbert": "true"}
    response = await session.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    payload = await response.json()
    await asyncio.sleep(1)
    return payload


async def main():
    texts = ["This is a test", 
             "This is also a test", 
             "This is a test within a test", 
             "Vector embeddings are cool!", 
             "I hope this works"]
    async with aiohttp.ClientSession() as session:
        embed = partial(get_colbert_embedding, session=session)
        tasks = list(map(embed, texts))
        results = await asyncio.gather(*tasks)
        results = list(map(json.loads, results))
        print(f"Retrieved {len(results)} results")
        for result in results:
            print(result["text"], result["colbert"][0][:5])
            

if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.2f} seconds")