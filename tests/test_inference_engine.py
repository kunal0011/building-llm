"""
Unit and integration tests for InferenceEngine, KV Cache, and OpenAI-compatible serving server.
"""

import pytest
import torch
from fastapi.testclient import TestClient

from train.model import ModelArgs
from serve.kv_cache import DynamicKVCache
from serve.inference_engine import InferenceEngine, SamplingParams
from serve.server import app


@pytest.fixture
def mini_model_args():
    return ModelArgs(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_dynamic_kv_cache():
    cache = DynamicKVCache(num_layers=2)
    k1 = torch.randn(1, 2, 4, 16)
    v1 = torch.randn(1, 2, 4, 16)
    k_out, v_out = cache.update(k1, v1, layer_idx=0)
    assert k_out.shape == (1, 2, 4, 16)

    # Next token step
    k2 = torch.randn(1, 2, 1, 16)
    v2 = torch.randn(1, 2, 1, 16)
    k_out2, v_out2 = cache.update(k2, v2, layer_idx=0)
    assert k_out2.shape == (1, 2, 5, 16)

    cache.reset()
    assert cache.k_cache[0] is None


def test_inference_engine_generation(mini_model_args):
    engine = InferenceEngine(model_args=mini_model_args, device="cpu")
    prompt = [10, 20, 30]
    sampling = SamplingParams(max_tokens=8, temperature=0.7, top_k=20, top_p=0.9)

    result = engine.generate(prompt, sampling_params=sampling)
    assert "generated_tokens" in result
    assert len(result["generated_tokens"]) > 0
    assert result["tokens_generated"] <= 8
    assert result["total_tokens"] == len(prompt) + result["tokens_generated"]
    assert "ttft_ms" in result
    assert "tokens_per_sec" in result


def test_openai_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_openai_chat_completions_endpoint(client):
    payload = {
        "model": "universal-foundation-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a python function to add two numbers."}
        ],
        "temperature": 0.5,
        "max_tokens": 16
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in data
    assert data["usage"]["completion_tokens"] > 0


def test_openai_completions_endpoint(client):
    payload = {
        "model": "universal-foundation-8b",
        "prompt": "def fibonacci(n):",
        "temperature": 0.0,
        "max_tokens": 12
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]
