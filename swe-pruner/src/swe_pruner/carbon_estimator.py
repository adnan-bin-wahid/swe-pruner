from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from .carbon_model_engine import (
    DualModeRegressorEngine,
    EstimationRequest,
)


class CarbonEstimateRequest(BaseModel):
    input_tokens: int
    output_tokens: int = 256
    model_name: str = "gpt-4o"
    model_size_b: float | None = None
    gpu_type: str = "nvidia-a100-80gb"
    latency_per_input_token_ms: float = 0.8
    latency_per_output_token_ms: float = 2.2
    mmlu_pro_score: float | None = None
    bbh_score: float | None = None
    carbon_intensity_g_per_kwh: float = 475.0


class CarbonEstimateResponse(BaseModel):
    prefill_joules: float
    decode_joules: float
    total_joules: float
    co2_grams: float
    carbon_intensity_g_per_kwh: float
    model_name: str
    prefill_route: str
    decode_route: str
    features_source: str


MODEL_CONSTANTS = {
    "gpt-4o": {
        "prefill_joules_per_token": 0.003,
        "decode_joules_per_token": 0.012,
    },
    "claude-3-5-sonnet": {
        "prefill_joules_per_token": 0.0028,
        "decode_joules_per_token": 0.0102,
    },
    "llama-3-70b": {
        "prefill_joules_per_token": 0.0042,
        "decode_joules_per_token": 0.0144,
    },
    "default": {
        "prefill_joules_per_token": 0.002,
        "decode_joules_per_token": 0.008,
    },
}

DEFAULT_MODEL_REGISTRY = {
    "gpt-4o": {
        "model_size_b": 200.0,
        "mmlu_pro_score": 70.0,
        "bbh_score": 83.0,
        "gpu_type": "nvidia-a100-80gb",
    },
    "claude-3-5-sonnet": {
        "model_size_b": 70.0,
        "mmlu_pro_score": 74.0,
        "bbh_score": 86.0,
        "gpu_type": "nvidia-a100-80gb",
    },
    "llama-3-70b": {
        "model_size_b": 70.0,
        "mmlu_pro_score": 52.0,
        "bbh_score": 67.0,
        "gpu_type": "nvidia-a100-80gb",
    },
}


class CarbonEstimator:
    """SEAL-style prompt-level estimator with artifact-based dual-regressor routing."""

    def __init__(self) -> None:
        artifacts_dir = Path(
            os.getenv("SWEPRUNER_CARBON_ARTIFACTS_DIR", "./carbon_artifacts")
        )
        self.engine = DualModeRegressorEngine(artifacts_dir=artifacts_dir)
        self.gpu_encoder = self._load_gpu_encoder(artifacts_dir)
        self.model_registry = self._load_model_registry(artifacts_dir)

    def _load_gpu_encoder(self, artifacts_dir: Path) -> dict[str, int]:
        path = artifacts_dir / "feature_artifacts.json"
        if not path.exists():
            return {"nvidia-a100-80gb": 0}

        payload = json.loads(path.read_text(encoding="utf-8"))
        encoder = payload.get("gpu_encoder", {})
        if not isinstance(encoder, dict) or not encoder:
            return {"nvidia-a100-80gb": 0}
        return {str(k).strip().lower(): int(v) for k, v in encoder.items()}

    def _load_model_registry(self, artifacts_dir: Path) -> dict[str, dict[str, float | str]]:
        path = artifacts_dir / "model_registry.json"
        if not path.exists():
            return DEFAULT_MODEL_REGISTRY

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            return DEFAULT_MODEL_REGISTRY
        return payload

    def _resolve_model_features(self, request: CarbonEstimateRequest) -> tuple[float, float, float, str]:
        key = request.model_name.strip().lower()
        defaults = self.model_registry.get(key, DEFAULT_MODEL_REGISTRY.get(key, {}))

        model_size_b = (
            request.model_size_b
            if request.model_size_b is not None
            else float(defaults.get("model_size_b", 200.0))
        )
        mmlu = (
            request.mmlu_pro_score
            if request.mmlu_pro_score is not None
            else float(defaults.get("mmlu_pro_score", 50.0))
        )
        bbh = (
            request.bbh_score
            if request.bbh_score is not None
            else float(defaults.get("bbh_score", 60.0))
        )
        gpu_type = request.gpu_type or str(defaults.get("gpu_type", "nvidia-a100-80gb"))

        return float(model_size_b), float(mmlu), float(bbh), gpu_type

    def _encode_gpu(self, gpu_type: str) -> int:
        normalized = gpu_type.strip().lower()
        if normalized in self.gpu_encoder:
            return self.gpu_encoder[normalized]
        return next(iter(self.gpu_encoder.values()))

    def estimate(self, request: CarbonEstimateRequest) -> CarbonEstimateResponse:
        model_size_b, mmlu, bbh, gpu_type = self._resolve_model_features(request)
        gpu_encoded = self._encode_gpu(gpu_type)

        features_source = "artifact_models"
        try:
            prediction = self.engine.predict(
                EstimationRequest(
                    n_input_tokens=request.input_tokens,
                    n_output_tokens=request.output_tokens,
                    model_size_b=model_size_b,
                    latency_per_input_token_ms=request.latency_per_input_token_ms,
                    latency_per_output_token_ms=request.latency_per_output_token_ms,
                    gpu_encoded=gpu_encoded,
                    mmlu_pro_score=mmlu,
                    bbh_score=bbh,
                )
            )
            prefill_joules = prediction.prefill_joules
            decode_joules = prediction.decode_joules
            prefill_route = prediction.prefill_route
            decode_route = prediction.decode_route
        except Exception:
            # Fallback keeps endpoint available when artifacts are not present yet.
            constants = MODEL_CONSTANTS.get(request.model_name, MODEL_CONSTANTS["default"])
            prefill_scale = max(0.2, request.latency_per_input_token_ms / 1.0)
            decode_scale = max(0.2, request.latency_per_output_token_ms / 1.0)
            prefill_joules = (
                request.input_tokens
                * constants["prefill_joules_per_token"]
                * prefill_scale
            )
            decode_joules = (
                request.output_tokens
                * constants["decode_joules_per_token"]
                * decode_scale
            )
            prefill_route = (
                "ridge_extrapolation"
                if model_size_b > 111.0
                else "xgboost_interpolation"
            )
            decode_route = prefill_route
            features_source = "fallback_constants"

        total_joules = max(0.0, prefill_joules + decode_joules)
        co2_grams = (total_joules / 3_600_000) * max(
            1.0, request.carbon_intensity_g_per_kwh
        )

        return CarbonEstimateResponse(
            prefill_joules=max(0.0, prefill_joules),
            decode_joules=max(0.0, decode_joules),
            total_joules=total_joules,
            co2_grams=co2_grams,
            carbon_intensity_g_per_kwh=request.carbon_intensity_g_per_kwh,
            model_name=request.model_name,
            prefill_route=prefill_route,
            decode_route=decode_route,
            features_source=features_source,
        )
