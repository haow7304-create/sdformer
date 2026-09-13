#!/usr/bin/env python3
"""End-to-end smoke and output-contract checks for forecast_api V2."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast_api import (  # noqa: E402
    EXCLUDED_INDICATORS,
    PRED_LEN,
    SEQ_LEN,
    ForecastService,
    ServiceError,
)


def _numbers(value):
    return list(value.values()) if isinstance(value, dict) else [value]


def _check_typed_values(service, data):
    schema = service.schema
    for indicator, indicator_type in schema["indicator_types"].items():
        values = _numbers(data[indicator])
        assert values and all(isinstance(value, (int, float)) for value in values)
        assert all(math.isfinite(float(value)) for value in values)
        if indicator_type == "distribution":
            assert all(value >= 0 for value in values)
            total = sum(values)
            assert math.isclose(total, 0.0, abs_tol=1e-8) or math.isclose(
                total, 1.0, abs_tol=1e-8
            )
        elif indicator_type == "count":
            assert all(isinstance(value, int) and value >= 0 for value in values)
        elif indicator_type == "bounded_index":
            assert all(0.0 <= value <= 1.0 for value in values)


def main() -> None:
    service = ForecastService("cuda:0")
    service.load()
    assert len(service.events) == 3
    assert service.schema["retained_indicator_count"] == 41
    assert service.schema["excluded_indicator_count"] == 7
    summaries = []

    for event_key, runtime in service.events.items():
        position = min(len(runtime.dates) - PRED_LEN - 1, max(SEQ_LEN + 24, int(len(runtime.dates) * 0.75)))
        target_time = runtime.dates[position]
        response = service.predict(event_key, target_time.strftime("%Y-%m-%d %H:%M:%S"), 24)
        history = response["data"]["history"]
        future = response["data"]["future"]
        assert history["points"] == len(history["items"]) == 24
        assert future["points"] == len(future["items"]) == 48
        assert future["predicted_indicator_count"] == 41
        assert future["actual_indicator_count"] == 7
        assert future["business_variable_count"] == 309
        assert future["model_internal_variable_count"] == 320
        assert set(future["predicted_indicators"]).isdisjoint(EXCLUDED_INDICATORS)
        assert set(future["actual_indicators"]) == set(EXCLUDED_INDICATORS)

        for item in history["items"] + future["items"]:
            assert len(item) == 49
            _check_typed_values(service, item)

        first_future = future["items"][0]
        timestamp = pd.Timestamp(first_future["timestep"])
        raw = runtime.raw_store.rows([timestamp])[0]
        for indicator in EXCLUDED_INDICATORS:
            assert first_future[indicator] == raw[indicator]
        stage = "ch1_1_3_5_舆情传播的阶段_calc_1_3_5"
        assert first_future[stage] == raw[stage]
        json.dumps(response, ensure_ascii=False, allow_nan=False)
        summaries.append(
            {
                "event": runtime.spec.display_name,
                "target": target_time.strftime("%Y-%m-%d %H:%M:%S"),
                "history_points": 24,
                "future_points": 48,
                "indicators": 48,
            }
        )

    try:
        service.predict("不存在的事件", "2025-01-01 00:00:00", 24)
    except ServiceError as error:
        assert error.status_code == 400
    else:
        raise AssertionError("Unknown event should be rejected")

    print(json.dumps({"status": "passed", "events": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
