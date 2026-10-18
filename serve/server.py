"""
OpenAI-Compatible FastAPI Model Serving Endpoint.
Exposes standard /v1/chat/completions, /v1/completions, and /v1/models endpoints
backed by our high-throughput InferenceEngine.
"""

import os
import sys
import time
import uuid
from typing import List, Dict, Any, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train.model import ModelArgs
from serve.inference_engine import InferenceEngine, SamplingParams
from pipelines.tokenization.custom_bpe_tokenizer import UniversalBPETokenizer


app = FastAPI(title="Foundation Model Inference Service", version="1.0.0")

# Initialize global engine and tokenizer
DEFAULT_MODEL_NAME = "universal-foundation-8b"
_engine: Optional[InferenceEngine] = None
_tokenizer: Any = None


def get_inference_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        ckpt_path = "data/checkpoints/checkpoint_final.pt"
        mini_args = ModelArgs(
            vocab_size=128000,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=2048
        )
        _engine = InferenceEngine(checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None, model_args=mini_args)
    return _engine


def get_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        candidate_paths = [
            "data/tokenizer_verified/tokenizer.json",
            os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer_verified", "tokenizer.json")
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                _tokenizer = UniversalBPETokenizer.load(path)
                return _tokenizer
        # Fallback if no pre-trained tokenizer on disk
        class FallbackTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                return [b % 1000 for b in text.encode("utf-8")]
            def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
                b = bytes([t % 256 for t in tokens])
                return b.decode("utf-8", errors="replace")
        _tokenizer = FallbackTokenizer()
    return _tokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_NAME
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 64
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_NAME
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 64


@app.get("/v1/models")
async def list_models():
    """Lists available foundation model checkpoints."""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepmind-agentic",
                "permission": [],
                "root": DEFAULT_MODEL_NAME
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completion endpoint.
    Formats chat messages into template, tokenizes, samples continuation, and returns JSON.
    """
    engine = get_inference_engine()
    tok = get_tokenizer()

    # Format messages into conversation prompt
    prompt_str = ""
    for msg in request.messages:
        prompt_str += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt_str += "<|im_start|>assistant\n"

    prompt_tokens = tok.encode(prompt_str)
    # Ensure at least 1 token
    if not prompt_tokens:
        prompt_tokens = [10]

    sampling = SamplingParams(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 50,
        max_tokens=request.max_tokens or 64
    )

    res = engine.generate(prompt_tokens, sampling_params=sampling)
    generated_text = tok.decode(res["generated_tokens"])

    comp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return {
        "id": comp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": res["tokens_generated"],
            "total_tokens": res["total_tokens"]
        },
        "telemetry": {
            "ttft_ms": res["ttft_ms"],
            "tokens_per_sec": res["tokens_per_sec"],
            "total_latency_sec": res["total_latency_sec"]
        }
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible raw prompt completion endpoint."""
    engine = get_inference_engine()
    tok = get_tokenizer()

    prompt_tokens = tok.encode(request.prompt)
    if not prompt_tokens:
        prompt_tokens = [10]

    sampling = SamplingParams(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        top_k=request.top_k or 50,
        max_tokens=request.max_tokens or 64
    )

    res = engine.generate(prompt_tokens, sampling_params=sampling)
    text = tok.decode(res["generated_tokens"])

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": res["tokens_generated"],
            "total_tokens": res["total_tokens"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve.server:app", host="0.0.0.0", port=8000, reload=False)
