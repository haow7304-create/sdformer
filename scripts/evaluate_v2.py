#!/usr/bin/env python3
"""Evaluate V2 saved test predictions in public/original business units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


EVENTS = {
    "russia_ukraine": ("俄乌冲突.csv", "russia_ukraine_v2"),
    "america_iran": ("美伊战争.csv", "america_iran_v2"),
    "china_japan": ("中日外交.csv", "china_japan_v2"),
}


def _short_name(indicator: str) -> str:
    return indicator.split("_", 4)[4].split("_calc", 1)[0]


def _round(value: float) -> float:
    return round(float(value), 6)


def _inverse_scaled(
    values: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.asarray(values[..., indices], dtype=np.float64) * scale[indices] + mean[indices]


def _restore_constants(
    values: np.ndarray,
    columns: list[str],
    constant_columns: Dict[str, float],
) -> None:
    for offset, column in enumerate(columns):
        if column in constant_columns:
            values[..., offset] = constant_columns[column]


def _normalize_distribution(
    components: np.ndarray,
    valid: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    result = np.maximum(components, 0.0)
    sums = result.sum(axis=-1, keepdims=True)
    fallback_sums = fallback.sum(axis=-1, keepdims=True)
    safe_fallback = np.divide(
        fallback,
        fallback_sums,
        out=np.full_like(fallback, 1.0 / fallback.shape[-1]),
        where=fallback_sums > 1e-12,
    )
    result = np.divide(result, sums, out=np.zeros_like(result), where=sums > 1e-12)
    result = np.where((sums <= 1e-12), safe_fallback, result)
    return np.where(valid[..., None], result, 0.0)


def evaluate_event(
    root: Path,
    event_key: str,
    csv_name: str,
    model_id: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    checkpoint_dir = root / "new_checkpoints_v2" / event_key
    with (checkpoint_dir / "preprocessor.json").open("r", encoding="utf-8") as handle:
        preprocessor = json.load(handle)
    with (checkpoint_dir / "calibration.json").open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)

    setting = (
        f"new_long_term_forecast_{model_id}_SDformer_custom_ftM_sl96_ll48_pl48_"
        "dm128_nh8_el4_dl1_df128_fc1_ebtimeF_dtTrue_V2_0"
    )
    result_dir = root / "results" / setting
    prediction_scaled = np.load(result_dir / "pred.npy", mmap_mode="r")
    truth_scaled = np.load(result_dir / "true.npy", mmap_mode="r")
    official_metrics = np.load(result_dir / "metrics.npy")

    frame = pd.read_csv(root / "dataset" / "processed_v2" / csv_name)
    model_columns = list(frame.columns[1:])
    if model_columns != preprocessor["model_columns"] or model_columns != schema["model_columns"]:
        raise ValueError(f"V2 column order mismatch for {event_key}")
    transformed = frame[model_columns].to_numpy(dtype=np.float64)
    column_to_index = {column: index for index, column in enumerate(model_columns)}
    mean = np.asarray(preprocessor["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(preprocessor["scaler"]["scale"], dtype=np.float64)
    constant_columns = preprocessor.get("constant_columns", {})

    num_test = int(len(frame) * 0.2)
    first_target = len(frame) - num_test - 1
    current = transformed[first_target : first_target + len(prediction_scaled)]
    if len(prediction_scaled) != num_test - prediction_scaled.shape[1] + 1:
        raise ValueError(f"Unexpected test result length for {event_key}")

    indicator_metrics = []
    validity_metrics = []
    all_finite = True
    all_constraints_valid = True

    for indicator, indicator_columns in schema["indicator_to_columns"].items():
        indices = np.asarray([column_to_index[column] for column in indicator_columns])
        predicted = _inverse_scaled(prediction_scaled, indices, mean, scale)
        truth = _inverse_scaled(truth_scaled, indices, mean, scale)
        current_values = current[:, indices]
        _restore_constants(predicted, indicator_columns, constant_columns)

        baseline = np.repeat(current_values[:, None, :], prediction_scaled.shape[1], axis=1)
        indicator_type = schema["indicator_types"][indicator]

        if indicator_type == "distribution":
            validity_column = schema["validity_columns"][indicator]
            validity_index = column_to_index[validity_column]
            predicted_validity = _inverse_scaled(
                prediction_scaled,
                np.asarray([validity_index]),
                mean,
                scale,
            )[..., 0]
            truth_validity = _inverse_scaled(
                truth_scaled,
                np.asarray([validity_index]),
                mean,
                scale,
            )[..., 0]
            if validity_column in constant_columns:
                predicted_validity.fill(constant_columns[validity_column])
            predicted_valid = predicted_validity >= 0.5
            truth_valid = truth_validity >= 0.5
            current_valid = current[:, validity_index] >= 0.5
            baseline_validity = np.repeat(
                current[:, validity_index][:, None], prediction_scaled.shape[1], axis=1
            )
            fallback = np.repeat(current_values[:, None, :], prediction_scaled.shape[1], axis=1)
            predicted = _normalize_distribution(predicted, predicted_valid, fallback)
            truth = _normalize_distribution(truth, truth_valid, fallback)
            baseline = _normalize_distribution(
                baseline,
                np.repeat(current_valid[:, None], prediction_scaled.shape[1], axis=1),
                baseline,
            )
            validity_alpha = np.asarray(
                calibration["validity_alpha_by_horizon"][indicator], dtype=np.float64
            )[None, :]
            calibrated_validity = baseline_validity + validity_alpha * (
                predicted_validity - baseline_validity
            )
            calibrated_valid = calibrated_validity >= 0.5
            validity_metrics.append(
                {
                    "name": _short_name(indicator),
                    "accuracy": _round(np.mean(calibrated_valid == truth_valid)),
                    "actual_valid_rate": _round(np.mean(truth_valid)),
                }
            )
        elif indicator_type == "count":
            predicted = np.maximum(np.expm1(predicted), 0.0)
            truth = np.rint(np.maximum(np.expm1(truth), 0.0))
            baseline = np.maximum(np.expm1(baseline), 0.0)
        elif indicator_type == "bounded_index":
            predicted = np.clip(predicted, 0.0, 1.0)
            truth = np.clip(truth, 0.0, 1.0)
            baseline = np.clip(baseline, 0.0, 1.0)

        alpha = np.asarray(
            calibration["indicator_alpha_by_horizon"][indicator], dtype=np.float64
        )[None, :, None]
        predicted = baseline + alpha * (predicted - baseline)

        if indicator_type == "distribution":
            predicted = _normalize_distribution(
                predicted, calibrated_valid, fallback
            )
            sums = predicted.sum(axis=-1)
            constraint_ok = np.all(
                np.isclose(sums, 0.0, atol=1e-8)
                | np.isclose(sums, 1.0, atol=1e-8)
            ) and np.all(predicted >= 0)
            all_constraints_valid = all_constraints_valid and bool(constraint_ok)
        elif indicator_type == "count":
            predicted = np.rint(np.maximum(predicted, 0.0))
            baseline = np.rint(np.maximum(baseline, 0.0))
            constraint_ok = np.all(predicted >= 0) and np.all(
                predicted == np.rint(predicted)
            )
            all_constraints_valid = all_constraints_valid and bool(constraint_ok)
        elif indicator_type == "bounded_index":
            predicted = np.clip(predicted, 0.0, 1.0)
            all_constraints_valid = all_constraints_valid and bool(
                np.all((predicted >= 0) & (predicted <= 1))
            )

        model_mae = np.mean(np.abs(predicted - truth))
        baseline_mae = np.mean(np.abs(baseline - truth))
        metric: Dict[str, Any] = {
            "name": _short_name(indicator),
            "type": indicator_type,
            "model_mae": _round(model_mae),
            "baseline_mae": _round(baseline_mae),
            "skill_pct": (
                None
                if baseline_mae < 1e-10
                else _round(100.0 * (1.0 - model_mae / baseline_mae))
            ),
        }
        if indicator_type == "count":
            denominator = np.sum(np.abs(truth))
            metric["model_wape_pct"] = _round(
                100.0 * np.sum(np.abs(predicted - truth)) / denominator
            )
            metric["baseline_wape_pct"] = _round(
                100.0 * np.sum(np.abs(baseline - truth)) / denominator
            )
        indicator_metrics.append(metric)
        all_finite = all_finite and bool(np.isfinite(predicted).all())

    comparable = [metric for metric in indicator_metrics if metric["skill_pct"] is not None]
    return {
        "test_windows": int(len(prediction_scaled)),
        "official_standardized_mae_all_320": _round(official_metrics[0]),
        "all_finite": all_finite,
        "all_output_constraints_valid": all_constraints_valid,
        "better_than_persistence": sum(metric["skill_pct"] > 0 for metric in comparable),
        "worse_than_or_equal_persistence": sum(metric["skill_pct"] <= 0 for metric in comparable),
        "constant_not_comparable": len(indicator_metrics) - len(comparable),
        "validity_metrics": validity_metrics,
        "count_metrics": [m for m in indicator_metrics if m["type"] == "count"],
        "worst_vs_persistence": sorted(
            comparable, key=lambda metric: metric["skill_pct"]
        )[:10],
        "indicator_metrics": indicator_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    with (root / "dataset" / "processed_v2" / "feature_schema.json").open(
        "r", encoding="utf-8"
    ) as handle:
        schema = json.load(handle)

    report = {
        event_key: evaluate_event(root, event_key, csv_name, model_id, schema)
        for event_key, (csv_name, model_id) in EVENTS.items()
    }
    output = args.output or root / "new_checkpoints_v2" / "evaluation_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
