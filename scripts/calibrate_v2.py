#!/usr/bin/env python3
"""Fit horizon-wise model/persistence blending using validation data only."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import Dataset_Custom
from models.SDformer import Model as SDformerModel


EVENTS = {
    "russia_ukraine": "俄乌冲突.csv",
    "america_iran": "美伊战争.csv",
    "china_japan": "中日外交.csv",
}


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=96, label_len=48, pred_len=48,
        enc_in=320, dec_in=320, c_out=320, d_model=128, n_heads=8,
        e_layers=4, d_layers=1, d_ff=128, factor=1, embed="timeF",
        freq="h", dropout=0.1, activation="gelu", output_attention=False,
        top_k=5, window_size=8, p=2,
    )


def _fit_alpha(model: np.ndarray, baseline: np.ndarray, truth: np.ndarray) -> list[float]:
    delta = model - baseline
    target_delta = truth - baseline
    axes = tuple(axis for axis in range(delta.ndim) if axis != 1)
    numerator = np.sum(delta * target_delta, axis=axes)
    denominator = np.sum(delta * delta, axis=axes)
    alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.clip(alpha, 0.0, 1.0).tolist()


def _normalize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    sums = values.sum(axis=-1, keepdims=True)
    values = np.divide(values, sums, out=np.zeros_like(values), where=sums > 1e-12)
    return np.where(valid[..., None], values, 0.0)


def _gate_alpha(
    alpha: list[float],
    model: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    holdout_start: int,
) -> tuple[list[float], dict[str, float | bool]]:
    alpha_array = np.asarray(alpha, dtype=np.float64)[None, :, None]
    calibrated = baseline + alpha_array * (model - baseline)
    baseline_mae = float(np.mean(np.abs(baseline[holdout_start:] - truth[holdout_start:])))
    calibrated_mae = float(np.mean(np.abs(calibrated[holdout_start:] - truth[holdout_start:])))
    accepted = calibrated_mae < baseline_mae * 0.995
    if not accepted:
        alpha = [0.0] * len(alpha)
        calibrated_mae = baseline_mae
    return alpha, {
        "accepted": accepted,
        "holdout_baseline_mae": baseline_mae,
        "holdout_calibrated_mae": calibrated_mae,
    }


def main() -> None:
    root = PROJECT_ROOT
    processed = root / "dataset" / "processed_v2"
    with (processed / "feature_schema.json").open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for event_key, csv_name in EVENTS.items():
        checkpoint_dir = root / "new_checkpoints_v2" / event_key
        with (checkpoint_dir / "preprocessor.json").open("r", encoding="utf-8") as handle:
            preprocessor = json.load(handle)
        columns = preprocessor["model_columns"]
        column_to_index = {column: index for index, column in enumerate(columns)}
        mean = np.asarray(preprocessor["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(preprocessor["scaler"]["scale"], dtype=np.float64)
        constants = preprocessor.get("constant_columns", {})

        dataset = Dataset_Custom(
            root_path=str(processed) + "/", flag="val", size=[96, 48, 48],
            features="M", data_path=csv_name, target="OT", timeenc=1, freq="h",
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        model = SDformerModel(_model_config()).float().to(device)
        state = torch.load(checkpoint_dir / "checkpoint.pth", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        prediction_parts = []
        truth_parts = []
        current_parts = []
        with torch.inference_mode():
            for batch_x, batch_y, batch_x_mark, _ in loader:
                output = model(
                    batch_x.float().to(device),
                    batch_x_mark.float().to(device),
                    None,
                    None,
                )
                prediction_parts.append(output.cpu().numpy())
                truth_parts.append(batch_y[:, -48:, :].numpy())
                current_parts.append(batch_x[:, -1, :].numpy())
        prediction_scaled = np.concatenate(prediction_parts)
        truth_scaled = np.concatenate(truth_parts)
        current_scaled = np.concatenate(current_parts)

        calibration = {
            "version": 1,
            "source_split": "validation",
            "validation_windows": len(prediction_scaled),
            "fit_windows": len(prediction_scaled) // 2,
            "holdout_windows": len(prediction_scaled) - len(prediction_scaled) // 2,
            "indicator_alpha_by_horizon": {},
            "validity_alpha_by_horizon": {},
            "validation_gate": {},
        }
        holdout_start = len(prediction_scaled) // 2
        for indicator, indicator_columns in schema["indicator_to_columns"].items():
            indices = np.asarray([column_to_index[column] for column in indicator_columns])
            model_values = prediction_scaled[..., indices] * scale[indices] + mean[indices]
            truth_values = truth_scaled[..., indices] * scale[indices] + mean[indices]
            current_values = current_scaled[:, indices] * scale[indices] + mean[indices]
            for offset, column in enumerate(indicator_columns):
                if column in constants:
                    model_values[..., offset] = constants[column]
            baseline = np.repeat(current_values[:, None, :], 48, axis=1)
            indicator_type = schema["indicator_types"][indicator]

            if indicator_type == "distribution":
                validity_column = schema["validity_columns"][indicator]
                validity_index = column_to_index[validity_column]
                model_validity = (
                    prediction_scaled[..., validity_index] * scale[validity_index]
                    + mean[validity_index]
                )
                truth_validity = (
                    truth_scaled[..., validity_index] * scale[validity_index]
                    + mean[validity_index]
                )
                current_validity = (
                    current_scaled[:, validity_index] * scale[validity_index]
                    + mean[validity_index]
                )
                if validity_column in constants:
                    model_validity.fill(constants[validity_column])
                baseline_validity = np.repeat(current_validity[:, None], 48, axis=1)
                validity_alpha = _fit_alpha(
                    model_validity[:holdout_start, ..., None],
                    baseline_validity[:holdout_start, ..., None],
                    truth_validity[:holdout_start, ..., None],
                )
                validity_alpha, validity_gate = _gate_alpha(
                    validity_alpha,
                    model_validity[..., None],
                    baseline_validity[..., None],
                    truth_validity[..., None],
                    holdout_start,
                )
                calibration["validity_alpha_by_horizon"][indicator] = validity_alpha
                model_values = _normalize(model_values, model_validity >= 0.5)
                truth_values = _normalize(truth_values, truth_validity >= 0.5)
                baseline = _normalize(baseline, baseline_validity >= 0.5)
            elif indicator_type == "count":
                model_values = np.maximum(np.expm1(model_values), 0.0)
                truth_values = np.maximum(np.expm1(truth_values), 0.0)
                baseline = np.maximum(np.expm1(baseline), 0.0)
            elif indicator_type == "bounded_index":
                model_values = np.clip(model_values, 0.0, 1.0)
                truth_values = np.clip(truth_values, 0.0, 1.0)
                baseline = np.clip(baseline, 0.0, 1.0)

            alpha = _fit_alpha(
                model_values[:holdout_start],
                baseline[:holdout_start],
                truth_values[:holdout_start],
            )
            alpha, indicator_gate = _gate_alpha(
                alpha, model_values, baseline, truth_values, holdout_start
            )
            calibration["indicator_alpha_by_horizon"][indicator] = alpha
            calibration["validation_gate"][indicator] = {
                "indicator": indicator_gate,
                "validity": validity_gate if indicator_type == "distribution" else None,
            }

        with (checkpoint_dir / "calibration.json").open("w", encoding="utf-8") as handle:
            json.dump(calibration, handle, ensure_ascii=False, indent=2)
        print(event_key, "validation_windows=", len(prediction_scaled))


if __name__ == "__main__":
    main()
