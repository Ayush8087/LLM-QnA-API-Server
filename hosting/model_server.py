# hosting/model_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
from typing import List
import time


app = FastAPI(title="Model Host")

# -------------------- MODEL LOADING --------------------
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("Loading model... (this may take a while)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully")
 
# -------------------- REQUEST MODELS --------------------
class GenerateRequest(BaseModel):
    chat_id: str 
    system_prompt: str 
    user_prompt: str

class BatchGenerateRequest(BaseModel):
    queries: List[GenerateRequest]

class GenerateResponse(BaseModel):
    chat_id: str | None = None
    response: str

# -------------------- ASYNC INFERENCE QUEUE --------------------
inference_queue = asyncio.Queue()
results = {}

async def inference_worker():
    """Background task that handles queued prompts asynchronously."""
    while True:
        job_id, prompt = await inference_queue.get()
        try:
            with torch.inference_mode():
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(**inputs, max_new_tokens=150)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results[job_id] = text
        except Exception as e:
            results[job_id] = f"Error during inference: {e}"
        finally:
            inference_queue.task_done()

@app.on_event("startup")
async def startup_event():
    """Start the async inference worker when server launches."""
    asyncio.create_task(inference_worker())
    print("✅ Async inference worker started")

# -------------------- SINGLE INFERENCE ENDPOINT --------------------
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.user_prompt:
        raise HTTPException(status_code=400, detail="user_prompt required")

    full_prompt = f"{req.system_prompt}\nUser: {req.user_prompt}\nAssistant:"
    job_id = req.chat_id or str(id(req))

    # Add to queue
    await inference_queue.put((job_id, full_prompt))

    # Wait until result is ready
    while job_id not in results:
        await asyncio.sleep(0.1)

    response_text = results.pop(job_id)
    return GenerateResponse(chat_id=req.chat_id, response=response_text)

# -------------------- BATCH INFERENCE ENDPOINT --------------------
@app.post("/generate/batch")
async def generate_batch(req: BatchGenerateRequest):
    """Dynamic batching: process multiple prompts together in one model.generate() call."""
    if not req.queries:
        raise HTTPException(status_code=400, detail="queries list cannot be empty")

    # Build all prompts
    prompts = [
        f"{q.system_prompt}\nUser: {q.user_prompt}\nAssistant:"
        for q in req.queries
    ]

    loop = asyncio.get_running_loop()

    def _sync_generate_batch():
        start_time = time.time()
        with torch.inference_mode():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=150)
            decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        end_time = time.time()
        print(f"[BATCH INFERENCE] Processed {len(prompts)} prompts in {end_time - start_time:.2f} seconds")
        return decoded

    results_texts = await loop.run_in_executor(None, _sync_generate_batch)

    return [
        {"chat_id": q.chat_id, "response": results_texts[i]}
        for i, q in enumerate(req.queries)
    ]


