"""Entry point for `make api`.

Runs the FastAPI service via uvicorn, applying the configurable memory cap
(configs/data.yaml's resource_limits.max_memory_gb) before the app is
imported. This is deliberately NOT done inside src/api/main.py's lifespan
handler: that module is imported directly by the test suite via FastAPI's
TestClient, which really executes lifespan - applying a memory cap there
would cap the test runner's own process, not a real deployment's. This
wrapper is never imported by tests, so it's the safe place for the call.
"""

from __future__ import annotations

import uvicorn

from src.logging_config import configure_logging
from src.resource_limits import apply_memory_limit_from_config


def main() -> None:
    configure_logging()
    apply_memory_limit_from_config()
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
