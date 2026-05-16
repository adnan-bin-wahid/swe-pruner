from __future__ import annotations

from pydantic import BaseModel


class CarbonEstimateRequest(BaseModel):
    input_tokens: int
    output_tokens: int = 256
    model_name: str = "gpt-4o"
    model_size_b: float = 200.0
    latency_per_input_token_ms: float = 0.8
    latency_per_output_token_ms: float = 2.2
    carbon_intensity_g_per_kwh: float = 475.0


class CarbonEstimateResponse(BaseModel):
    prefill_joules: float
    decode_joules: float
    total_joules: float
    co2_grams: float
    carbon_intensity_g_per_kwh: float
    model_name: str
    route: str


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


class CarbonEstimator:
    """SEAL-style deterministic estimator scaffold for prompt-level carbon reporting."""

    def estimate(self, request: CarbonEstimateRequest) -> CarbonEstimateResponse:
        constants = MODEL_CONSTANTS.get(request.model_name, MODEL_CONSTANTS["default"])
        route = (
            "ridge_extrapolation"
            if request.model_size_b > 111.0
            else "xgboost_interpolation"
        )

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

        total_joules = max(0.0, prefill_joules + decode_joules)
        co2_grams = (total_joules / 3_600_000) * max(1.0, request.carbon_intensity_g_per_kwh)

        return CarbonEstimateResponse(
            prefill_joules=max(0.0, prefill_joules),
            decode_joules=max(0.0, decode_joules),
            total_joules=total_joules,
            co2_grams=co2_grams,
            carbon_intensity_g_per_kwh=request.carbon_intensity_g_per_kwh,
            model_name=request.model_name,
            route=route,
        )
