from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

RouteType = Literal["xgboost_interpolation", "ridge_extrapolation"]

FEATURE_COLUMNS = [
    "n_input_tokens",
    "n_output_tokens",
    "model_size_b",
    "latency_per_input_token_ms",
    "latency_per_output_token_ms",
    "gpu_encoded",
    "mmlu_pro_score",
    "bbh_score",
]


@dataclass(frozen=True)
class EstimationRequest:
    n_input_tokens: int
    n_output_tokens: int
    model_size_b: float
    latency_per_input_token_ms: float
    latency_per_output_token_ms: float
    gpu_encoded: int
    mmlu_pro_score: float
    bbh_score: float


@dataclass(frozen=True)
class EstimationResponse:
    prefill_joules: float
    decode_joules: float
    total_joules: float
    prefill_route: RouteType
    decode_route: RouteType


class DualModeRegressorEngine:
    def __init__(self, artifacts_dir: str | Path, interpolation_max_model_size_b: float = 111.0):
        self.artifacts_dir = Path(artifacts_dir)
        self.interpolation_max_model_size_b = interpolation_max_model_size_b

        self.xgb_prefill = self._load_xgb("xgb_prefill_interpolation.json")
        self.xgb_decode = self._load_xgb("xgb_decode_interpolation.json")
        self.ridge_prefill = self._load_pickle("ridge_prefill_extrapolation.pkl")
        self.ridge_decode = self._load_pickle("ridge_decode_extrapolation.pkl")

    def _load_xgb(self, filename: str):
        if XGBRegressor is None:
            return None

        path = self.artifacts_dir / filename
        if not path.exists():
            return None

        model = XGBRegressor()
        model.load_model(path)
        return model

    def _load_pickle(self, filename: str):
        if joblib is None:
            return None

        path = self.artifacts_dir / filename
        if not path.exists():
            return None

        return joblib.load(path)

    def is_ready(self) -> bool:
        return all(
            model is not None
            for model in [
                self.xgb_prefill,
                self.xgb_decode,
                self.ridge_prefill,
                self.ridge_decode,
            ]
        )

    def _route(self, model_size_b: float) -> RouteType:
        if model_size_b <= self.interpolation_max_model_size_b:
            return "xgboost_interpolation"
        return "ridge_extrapolation"

    def _as_frame(self, request: EstimationRequest) -> pd.DataFrame:
        row = {
            "n_input_tokens": request.n_input_tokens,
            "n_output_tokens": request.n_output_tokens,
            "model_size_b": request.model_size_b,
            "latency_per_input_token_ms": request.latency_per_input_token_ms,
            "latency_per_output_token_ms": request.latency_per_output_token_ms,
            "gpu_encoded": request.gpu_encoded,
            "mmlu_pro_score": request.mmlu_pro_score,
            "bbh_score": request.bbh_score,
        }
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict(self, request: EstimationRequest) -> EstimationResponse:
        features = self._as_frame(request)
        route = self._route(request.model_size_b)

        if route == "xgboost_interpolation":
            if self.xgb_prefill is None or self.xgb_decode is None:
                raise RuntimeError("XGBoost interpolation models are missing")
            prefill = float(self.xgb_prefill.predict(features)[0])
            decode = float(self.xgb_decode.predict(features)[0])
            prefill_route: RouteType = "xgboost_interpolation"
            decode_route: RouteType = "xgboost_interpolation"
        else:
            if self.ridge_prefill is None or self.ridge_decode is None:
                raise RuntimeError("Ridge extrapolation models are missing")
            prefill = float(np.asarray(self.ridge_prefill.predict(features))[0])
            decode = float(np.asarray(self.ridge_decode.predict(features))[0])
            prefill_route = "ridge_extrapolation"
            decode_route = "ridge_extrapolation"

        return EstimationResponse(
            prefill_joules=max(0.0, prefill),
            decode_joules=max(0.0, decode),
            total_joules=max(0.0, prefill + decode),
            prefill_route=prefill_route,
            decode_route=decode_route,
        )
