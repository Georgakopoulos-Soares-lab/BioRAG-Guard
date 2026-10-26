"""``bioguard serve`` -- the same two modes over HTTP.

The schema is generated from the handlers rather than written by hand, so
it cannot drift from the code. Authentication is a shared-token header,
off by default, with a loud warning when the service is bound to anything
other than localhost: a guard that anyone can reconfigure is not a guard.
"""

# No `from __future__ import annotations` here. FastAPI resolves handler
# annotations with get_type_hints against module globals; with postponed
# evaluation the request models -- which are defined inside create_app, so
# they can depend on the optional pydantic import -- are unresolvable strings
# and every body parameter silently degrades to a query parameter.

import uuid
from datetime import datetime
from typing import Any

import numpy as np

from ..core.record import Record
from ..core.registry import all_layers
from ..guard import Guard


def _record_from(payload: dict[str, Any]) -> Record:
    vector = payload.get("vector")
    counts = payload.get("counts")
    added = payload.get("added_at")
    return Record(
        id=str(payload.get("id", uuid.uuid4().hex)),
        vector=None if vector is None else np.asarray(vector, dtype=np.float32),
        label=payload.get("label"),
        text=payload.get("text"),
        payload=payload.get("payload"),
        counts=None if counts is None else np.asarray(counts, dtype=np.float32),
        donor=payload.get("donor"),
        added_at=None if added is None else datetime.fromisoformat(added),
        features=payload.get("features", {}),
        metadata=payload.get("metadata", {}),
        source=payload.get("source"),
    )


def create_app(guard: Guard, token: str | None = None) -> Any:
    """Build the FastAPI application around an open corpus."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "service mode needs fastapi and uvicorn.\n"
            "  pip install 'biorag-guard[service]'"
        ) from exc

    class RecordIn(BaseModel):
        id: str | None = None
        vector: list[float] | None = None
        label: str | None = None
        text: str | None = None
        payload: str | None = None
        counts: list[float] | None = None
        donor: str | None = None
        added_at: str | None = None
        features: dict[str, Any] = Field(default_factory=dict)
        metadata: dict[str, Any] = Field(default_factory=dict)
        source: str | None = None

    class CheckIn(BaseModel):
        records: list[RecordIn]

    class ScanIn(BaseModel):
        sample: int | None = None
        since: str | None = None

    class VoteIn(BaseModel):
        vector: list[float]
        k: int | None = None

    def auth(x_bioguard_token: str | None = Header(default=None)) -> None:
        if token is not None and x_bioguard_token != token:
            raise HTTPException(status_code=401, detail="bad or missing token")

    app = FastAPI(
        title="BioRAG-Guard",
        version="1.0",
        description="Check at ingest, scan a stored corpus.",
    )
    jobs: dict[str, dict[str, Any]] = {}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "corpus": guard.adapter.describe(),
                "calibrated": guard.baseline is not None}

    @app.get("/policy")
    def policy() -> dict[str, Any]:
        return guard.config.to_dict()

    @app.get("/layers")
    def layers() -> list[dict[str, Any]]:
        return [guard.explain(name) for name in all_layers()]

    @app.post("/check", dependencies=[Depends(auth)])
    def check(body: CheckIn) -> dict[str, Any]:
        records = [_record_from(r.model_dump()) for r in body.records]
        return guard.check_many(records).to_dict()

    @app.post("/scan", dependencies=[Depends(auth)])
    def scan(body: ScanIn) -> dict[str, str]:
        job_id = uuid.uuid4().hex
        report = guard.scan(sample=body.sample if body.sample is not None else -1,
                            since=body.since)
        jobs[job_id] = report.to_dict()
        return {"job_id": job_id}

    @app.get("/scan/{job_id}")
    def scan_result(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="no such job")
        return jobs[job_id]

    @app.post("/vote", dependencies=[Depends(auth)])
    def vote(body: VoteIn) -> dict[str, Any]:
        out = guard.vote(np.asarray(body.vector, dtype=np.float32), k=body.k)
        return {
            "label": out.label,
            "undefended_label": out.undefended_label,
            "abstained": out.abstained,
            "changed": out.changed,
            "n_neighbours": out.n_neighbours,
            "n_survivors": out.n_survivors,
            "policy": out.policy,
        }

    return app
