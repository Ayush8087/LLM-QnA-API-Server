# api_server/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
import asyncio

app = FastAPI(title="API Server")

# set this to where your model server will run
MODEL_HOST = "http://127.0.0.1:8001"

class SingleQuery(BaseModel):
    chat_id: str | None = None
    system_prompt: str | None = ""
    user_prompt: str

class BatchQueryItem(SingleQuery):
    pass

class BatchRequest(BaseModel):
    queries: List[BatchQueryItem]

@app.post("/chat")
async def chat(q: SingleQuery):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{MODEL_HOST}/generate", json=q.dict())
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Model host error")
        return resp.json()

@app.post("/chat/batched")
async def chat_batched(batch: BatchRequest):
    # create an async client and send all requests concurrently
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for q in batch.queries:
            tasks.append(client.post(f"{MODEL_HOST}/generate", json=q.dict()))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for item in results:
            if isinstance(item, Exception):
                responses.append({"response": None, "error": str(item)})
            else:
                if item.status_code == 200:
                    responses.append(item.json())
                else:
                    responses.append({"response": None, "error": f"status {item.status_code}"})
        return {"results": responses}
