"""HTTP service mode, for ingest pipelines that are not written in Python."""

from .app import create_app

__all__ = ["create_app"]
