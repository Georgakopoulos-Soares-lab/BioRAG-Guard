"""Service mode, skipped cleanly when the optional extra is absent."""

from __future__ import annotations

import pytest

from biorag_guard import Guard, demo

fastapi = pytest.importorskip("fastapi", reason="needs biorag-guard[service]")
from fastapi.testclient import TestClient  # noqa: E402

from biorag_guard.service import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=30)
    guard = Guard.auto(corpus.path)
    guard.calibrate(budget=0.05)
    return TestClient(create_app(guard, token="secret")), guard, corpus


def test_health_reports_the_corpus(client):
    api, _, _ = client
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["calibrated"] is True
    assert body["corpus"]["n_records"] == 120


def test_layers_endpoint_states_misses(client):
    api, _, _ = client
    layers = api.get("/layers").json()
    assert len(layers) >= 15
    assert all(layer["misses"] for layer in layers)


def test_check_requires_the_token(client):
    api, _, corpus = client
    vecs, rows = demo.poison(corpus, "duplicate", n=2)
    payload = {"records": [
        {"id": r["id"], "vector": vecs[i].tolist(), "label": r["label"],
         "donor": r["donor"], "counts": r["counts"]}
        for i, r in enumerate(rows)
    ]}
    assert api.post("/check", json=payload).status_code == 401
    ok = api.post("/check", json=payload, headers={"x-bioguard-token": "secret"})
    assert ok.status_code == 200
    assert ok.json()["verdict"] in ("quarantine", "reject", "flag")


def test_scan_returns_a_job_then_the_report(client):
    api, _, _ = client
    job = api.post("/scan", json={"sample": 30},
                   headers={"x-bioguard-token": "secret"}).json()
    report = api.get(f"/scan/{job['job_id']}").json()
    assert report["command"] == "scan"
    assert report["schema_version"] == "1.0"


def test_vote_returns_both_answers(client):
    api, guard, _ = client
    import numpy as np

    q = np.asarray(guard.adapter.vectors)[0].tolist()
    body = api.post("/vote", json={"vector": q},
                    headers={"x-bioguard-token": "secret"}).json()
    assert "label" in body and "undefended_label" in body
    assert body["n_neighbours"] > 0
