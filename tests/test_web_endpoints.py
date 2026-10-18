"""
Integration tests for FastAPI dashboard and REST endpoints.
Verifies service status, shard listing, multimodal, and SWE trajectory synthesis endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from web.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Data Forge" in response.text
    assert "SWE Agent Synthesizer" in response.text
    assert "swe_agent" in response.text


def test_service_status_endpoint(client):
    response = client.get("/api/service-status")
    assert response.status_code == 200
    data = response.json()
    assert "docker_compose" in data
    assert "s3_connected" in data


def test_shards_endpoint(client):
    response = client.get("/api/shards")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_swe_trajectories_get(client):
    response = client.get("/api/swe/trajectories")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_swe_synthesize_trajectories_post(client):
    response = client.post("/api/swe/synthesize-trajectories")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    if data["mode"] == "distributed_collector":
        assert data["summary"]["verified_passed"] >= 1
        assert data["summary"]["pass_rate"] == 1.0
        assert "sample_trajectory" in data
        assert data["sample_trajectory"]["sandbox_passed"] is True


def test_swe_bench_eval_post(client):
    response = client.post("/api/eval/swe-bench")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["summary"]["total_benchmark_tasks"] >= 4
    assert data["summary"]["resolution_rate"] == 1.0
    assert "report_paths" in data


def test_eval_reports_get(client):
    response = client.get("/api/eval/reports")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
