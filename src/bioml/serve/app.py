"""FastAPI service: wearable session in, per-second metabolic predictions out.

Run locally::

    python -m bioml serve --model models/v2 --port 8000
    curl -s localhost:8000/health

The model is loaded once at startup (lifespan) from ``BIOML_MODEL_PATH`` or the
``--model`` flag. ``create_app`` is a factory so tests can point it at a
throwaway model directory.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from bioml import __version__
from bioml.serve.pipeline import predict_request
from bioml.serve.schemas import HealthResponse, PredictRequest, PredictResponse
from bioml.train import FeatureContractError, TrainedModel

DEFAULT_MODEL_PATH = Path(os.environ.get("BIOML_MODEL_PATH", "models/v2"))


def create_app(model_path: Path | str = DEFAULT_MODEL_PATH) -> FastAPI:
    model_path = Path(model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = TrainedModel.load(model_path)
        app.state.model_path = model_path
        yield

    app = FastAPI(
        title="bioML inference API",
        version=__version__,
        description=(
            "Predict VO2, RER, and Frayn-gated substrate oxidation from consumer "
            "wearable signals. Lab signals are never inputs."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model: TrainedModel = request.app.state.model
        return HealthResponse(
            status="ok",
            model_path=str(request.app.state.model_path),
            model_version=str(model.metadata.get("model_version", model.metadata.get("git_sha"))),
            bioml_version=__version__,
            n_features=len(model.feature_names),
        )

    @app.get("/model/info")
    def model_info(request: Request) -> dict[str, Any]:
        model: TrainedModel = request.app.state.model
        return model.metadata

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, body: PredictRequest) -> PredictResponse:
        model: TrainedModel = request.app.state.model
        try:
            return predict_request(model, body)
        except FeatureContractError as e:  # pragma: no cover - indicates a deploy mismatch
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="bioml.serve")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    os.environ["BIOML_MODEL_PATH"] = str(args.model)
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
