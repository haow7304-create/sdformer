#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated SDformer V2 event forecasting API.

For each request, all 48 indicators are returned for history and the next 48
hours. Future data contains 41 calibrated model predictions and seven fields
copied from the corresponding raw JSON, including propagation stage.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.SDformer import Model as SDformerModel
from utils.timefeatures import time_features


LOGGER = logging.getLogger("sdformer.forecast_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PROJECT_ROOT = Path(
    os.getenv("SDFORMER_PROJECT_ROOT", Path(__file__).resolve().parent)
).resolve()
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_v2"
CHECKPOINT_DIR = PROJECT_ROOT / "new_checkpoints_v2"
FEATURE_SCHEMA_PATH = PROCESSED_DIR / "feature_schema.json"

SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 48
NUM_BUSINESS_VARIATES = 309
NUM_AUXILIARY_VARIATES = 11
NUM_MODEL_VARIATES = 320
PREDICTED_INDICATOR_COUNT = 41
RAW_ACTUAL_INDICATOR_COUNT = 7
MAX_HISTORY_HOURS = int(os.getenv("MAX_HISTORY_HOURS", "720"))
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
MISSING_STRINGS = {"", "null", "none", "nan", "na", "n/a", "-"}
KEY_ALIASES = {"中性": "中立"}


EXCLUDED_INDICATORS = (
    "ch1_1_2_3_最具影响力用户_calc_1_2_3",
    "ch1_1_2_4_最具正能量用户_calc_1_2_4",
    "ch1_1_2_5_最具负能量用户_calc_1_2_5",
    "ch2_2_2_4_相关热搜关键词_calc_2_2_4",
    "ch3_3_2_1_最具热度平台_calc_3_2_1",
    "ch3_3_4_1_平台首次响应时间_calc_3_4_1",
    "ch1_1_3_5_舆情传播的阶段_calc_1_3_5",
)


@dataclass(frozen=True)
class EventSpec:
    key: str
    display_name: str
    aliases: Tuple[str, ...]
    processed_csv: Path
    raw_json: Path
    checkpoint_dir: Path

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / "checkpoint.pth"

    @property
    def preprocessor(self) -> Path:
        return self.checkpoint_dir / "preprocessor.json"

    @property
    def calibration(self) -> Path:
        return self.checkpoint_dir / "calibration.json"


EVENT_SPECS: "OrderedDict[str, EventSpec]" = OrderedDict(
    [
        (
            "russia_ukraine",
            EventSpec(
                "russia_ukraine", "俄乌冲突",
                ("俄乌冲突", "俄乌", "russia_ukraine", "russia-ukraine"),
                PROCESSED_DIR / "俄乌冲突.csv",
                PROJECT_ROOT / "dataset" / "俄乌冲突_raw_time_series_data.json",
                CHECKPOINT_DIR / "russia_ukraine",
            ),
        ),
        (
            "america_iran",
            EventSpec(
                "america_iran", "美伊战争",
                ("美伊战争", "美伊", "america_iran", "america-iran", "america_iraq_war"),
                PROCESSED_DIR / "美伊战争.csv",
                PROJECT_ROOT / "dataset" / "美伊战争_raw_time_series_data.json",
                CHECKPOINT_DIR / "america_iran",
            ),
        ),
        (
            "china_japan",
            EventSpec(
                "china_japan", "中日外交",
                ("中日外交", "中日", "china_japan", "china-japan", "cn_jan"),
                PROCESSED_DIR / "中日外交.csv",
                PROJECT_ROOT / "dataset" / "中日外交_raw_time_series_data.json",
                CHECKPOINT_DIR / "china_japan",
            ),
        ),
    ]
)


class PredictRequest(BaseModel):
    p: str = Field(min_length=1, description="事件名称或事件 key")
    t: str = Field(min_length=1, description="预测基准时间 YYYY-MM-DD HH:MM:SS")
    w: int = Field(ge=1, le=MAX_HISTORY_HOURS, description="返回的历史小时数")


class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 422, **details: Any):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=SEQ_LEN, label_len=LABEL_LEN,
        pred_len=PRED_LEN, enc_in=NUM_MODEL_VARIATES, dec_in=NUM_MODEL_VARIATES,
        c_out=NUM_MODEL_VARIATES, d_model=128, n_heads=8, e_layers=4,
        d_layers=1, d_ff=128, factor=1, embed="timeF", freq="h",
        dropout=0.1, activation="gelu", output_attention=False, top_k=5,
        window_size=8, p=2,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net", "params"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                checkpoint = nested
                break
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    state_dict = dict(checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    return state_dict


def _parse_hour(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            "t 不是有效时间，格式应为 YYYY-MM-DD HH:MM:SS", received=value
        ) from exc
    if timestamp.tzinfo is not None:
        raise ServiceError("t 必须是不带时区的本地时间", received=value)
    if timestamp.minute or timestamp.second or timestamp.microsecond or timestamp.nanosecond:
        raise ServiceError("t 必须是整点时间", received=value)
    return timestamp


def _format_time(value: pd.Timestamp) -> str:
    return value.strftime(DATE_FORMAT)


def _hour_range(start: pd.Timestamp, periods: int) -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, periods=periods, freq="h"))


def _canonical_key(value: object) -> str:
    key = str(value).strip()
    return KEY_ALIASES.get(key, key)


def _parse_raw_number(value: Any, percentage: bool = False) -> Optional[float]:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, (bool, np.bool_)):
        result, had_percent_sign = float(value), False
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result, had_percent_sign = float(value), False
    else:
        text = str(value).strip()
        if text.lower() in MISSING_STRINGS:
            return None
        had_percent_sign = text.endswith("%")
        if had_percent_sign:
            text = text[:-1].strip()
        text = text.replace(",", "")
        if not NUMBER_RE.fullmatch(text):
            return None
        result = float(text)
    if not math.isfinite(result):
        return None
    return result / 100.0 if percentage or had_percent_sign else result


def _normalize_distribution(
    components: np.ndarray, valid: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    result = np.maximum(np.asarray(components, dtype=np.float64), 0.0)
    fallback = np.maximum(np.asarray(fallback, dtype=np.float64), 0.0)
    sums = result.sum(axis=-1, keepdims=True)
    fallback_sums = fallback.sum(axis=-1, keepdims=True)
    safe_fallback = np.divide(
        fallback, fallback_sums,
        out=np.full_like(fallback, 1.0 / fallback.shape[-1]),
        where=fallback_sums > 1e-12,
    )
    result = np.divide(result, sums, out=np.zeros_like(result), where=sums > 1e-12)
    result = np.where(sums <= 1e-12, safe_fallback, result)
    return np.where(np.asarray(valid)[..., None], result, 0.0)


class RawHistoryStore:
    def __init__(self, spec: EventSpec):
        self.spec = spec
        self._lock = threading.Lock()
        self._rows_by_time: Optional[Dict[pd.Timestamp, Dict[str, Any]]] = None
        self._indicator_order: Optional[List[str]] = None

    def _ensure_loaded(self) -> None:
        if self._rows_by_time is not None:
            return
        with self._lock:
            if self._rows_by_time is not None:
                return
            LOGGER.info("Loading raw JSON for %s", self.spec.display_name)
            with self.spec.raw_json.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"Raw JSON is not a non-empty array: {self.spec.raw_json}")
            rows_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
            indicator_order: Optional[List[str]] = None
            for row in payload:
                if not isinstance(row, dict) or "timestep" not in row:
                    raise RuntimeError(f"Invalid raw row in {self.spec.raw_json}")
                rows_by_time[_parse_hour(str(row["timestep"]))] = row
                if indicator_order is None:
                    indicator_order = [key for key in row if key != "timestep"]
            if indicator_order is None or len(indicator_order) != 48:
                raise RuntimeError(
                    f"Expected 48 raw indicators in {self.spec.raw_json}, "
                    f"got {len(indicator_order or [])}"
                )
            self._rows_by_time = rows_by_time
            self._indicator_order = indicator_order

    @property
    def indicator_order(self) -> Sequence[str]:
        self._ensure_loaded()
        assert self._indicator_order is not None
        return self._indicator_order

    @property
    def cached(self) -> bool:
        return self._rows_by_time is not None

    def rows(self, timestamps: Sequence[pd.Timestamp]) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._rows_by_time is not None
        missing = [timestamp for timestamp in timestamps if timestamp not in self._rows_by_time]
        if missing:
            raise ServiceError(
                "原始 JSON 缺少请求时间范围内的数据",
                missing_timestamps=[_format_time(value) for value in missing[:10]],
                missing_count=len(missing),
            )
        return [self._rows_by_time[timestamp] for timestamp in timestamps]

    def future_actual_rows(self, timestamps: Sequence[pd.Timestamp]) -> List[Dict[str, Any]]:
        rows = self.rows(timestamps)
        missing_fields = []
        for timestamp, row in zip(timestamps, rows):
            for indicator in EXCLUDED_INDICATORS:
                if indicator not in row or row[indicator] is None:
                    missing_fields.append(
                        {"timestep": _format_time(timestamp), "indicator": indicator}
                    )
        if missing_fields:
            raise ServiceError(
                "未来48小时的原始 JSON 中缺少需要拼接的真实指标",
                missing_actual_fields=missing_fields[:20],
                missing_count=len(missing_fields),
            )
        return rows


class EventRuntime:
    def __init__(self, spec: EventSpec, schema: Mapping[str, Any], device: torch.device):
        self.spec = spec
        self.schema = schema
        self.device = device
        self.indicator_to_columns = OrderedDict(schema["indicator_to_columns"])
        self.indicator_types = schema["indicator_types"]
        self.percentage_indicators = {
            indicator
            for indicator in schema["unit_sum_distributions"]
            if indicator.startswith("ch1_1_1_")
        }
        self.raw_store = RawHistoryStore(spec)

        preprocessor = _load_json(spec.preprocessor)
        calibration = _load_json(spec.calibration)
        frame = pd.read_csv(spec.processed_csv, encoding="utf-8")
        if list(frame.columns) != list(schema["columns"]):
            raise RuntimeError(f"CSV columns do not match schema: {spec.processed_csv}")
        feature_columns = list(frame.columns[1:])
        if feature_columns != preprocessor.get("model_columns"):
            raise RuntimeError(f"CSV columns do not match preprocessor: {spec.processed_csv}")
        if len(feature_columns) != NUM_MODEL_VARIATES:
            raise RuntimeError(f"Expected {NUM_MODEL_VARIATES} model variables: {spec.processed_csv}")

        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise RuntimeError(f"Invalid time index: {spec.processed_csv}")
        if not (dates.diff().dropna() == pd.Timedelta(hours=1)).all():
            raise RuntimeError(f"CSV is not hourly continuous: {spec.processed_csv}")

        values = frame[feature_columns].to_numpy(dtype=np.float64)
        mean = np.asarray(preprocessor["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(preprocessor["scaler"]["scale"], dtype=np.float64)
        if mean.shape != (NUM_MODEL_VARIATES,) or scale.shape != (NUM_MODEL_VARIATES,):
            raise RuntimeError(f"Invalid scaler shape: {spec.preprocessor}")
        if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise RuntimeError(f"Non-finite V2 data or scaler: {spec.display_name}")
        if np.any(scale <= 0):
            raise RuntimeError(f"Scaler contains non-positive scale: {spec.preprocessor}")
        scaled_values = ((values - mean) / scale).astype(np.float32)

        indicator_alpha = calibration.get("indicator_alpha_by_horizon", {})
        validity_alpha = calibration.get("validity_alpha_by_horizon", {})
        if set(indicator_alpha) != set(self.indicator_to_columns):
            raise RuntimeError(f"Calibration indicator set mismatch: {spec.calibration}")
        for indicator in schema["unit_sum_distributions"]:
            if indicator not in validity_alpha:
                raise RuntimeError(f"Missing validity calibration for {indicator}")
        for name, alpha_values in list(indicator_alpha.items()) + list(validity_alpha.items()):
            if len(alpha_values) != PRED_LEN or not np.isfinite(alpha_values).all():
                raise RuntimeError(f"Invalid 48-hour calibration for {name}")

        model = SDformerModel(_model_config()).float().to(device)
        model.load_state_dict(_load_state_dict(spec.checkpoint, device), strict=True)
        model.eval()

        self.frame = frame
        self.dates = pd.DatetimeIndex(dates)
        self.date_to_position = {timestamp: i for i, timestamp in enumerate(self.dates)}
        self.feature_columns = feature_columns
        self.column_to_position = {column: i for i, column in enumerate(feature_columns)}
        self.values = values
        self.scaled_values = scaled_values
        self.mean = mean
        self.scale = scale
        self.constant_columns = preprocessor.get("constant_columns", {})
        self.calibration = calibration
        self.model = model

    def available_range(self) -> Dict[str, str]:
        return {
            "processed_start": _format_time(self.dates[0]),
            "processed_end": _format_time(self.dates[-1]),
            "earliest_model_time": _format_time(self.dates[SEQ_LEN - 1]),
        }

    def validate_target_time(self, target_time: pd.Timestamp) -> int:
        position = self.date_to_position.get(target_time)
        if position is None:
            raise ServiceError(
                "t 不在该事件的处理后时序数据中", status_code=404,
                event=self.spec.display_name, requested_time=_format_time(target_time),
                **self.available_range(),
            )
        if position < SEQ_LEN - 1:
            raise ServiceError(
                f"t 之前不足模型要求的 {SEQ_LEN} 小时输入数据",
                event=self.spec.display_name, requested_time=_format_time(target_time),
                earliest_model_time=_format_time(self.dates[SEQ_LEN - 1]),
            )
        return position

    def _model_predict(self, position: int) -> np.ndarray:
        start = position - SEQ_LEN + 1
        input_dates = self.dates[start : position + 1]
        marks = time_features(input_dates, freq="h").transpose(1, 0).astype(np.float32)
        x_enc = torch.from_numpy(self.scaled_values[start : position + 1]).unsqueeze(0).to(self.device)
        x_mark = torch.from_numpy(marks).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(x_enc, x_mark, None, None)
        result = output.squeeze(0).detach().cpu().numpy().astype(np.float64)
        if result.shape != (PRED_LEN, NUM_MODEL_VARIATES):
            raise RuntimeError(f"Unexpected model output shape: {result.shape}")
        if not np.isfinite(result).all():
            raise RuntimeError(f"Model produced NaN or infinity: {self.spec.display_name}")
        return result

    def predict(self, target_time: pd.Timestamp) -> List[Dict[str, Any]]:
        position = self.validate_target_time(target_time)
        prediction_scaled = self._model_predict(position)
        current = self.values[position]
        rows: List[Dict[str, Any]] = [OrderedDict() for _ in range(PRED_LEN)]

        for indicator, columns_value in self.indicator_to_columns.items():
            columns = list(columns_value)
            indices = np.asarray([self.column_to_position[column] for column in columns])
            model_values = prediction_scaled[:, indices] * self.scale[indices] + self.mean[indices]
            for offset, column in enumerate(columns):
                if column in self.constant_columns:
                    model_values[:, offset] = float(self.constant_columns[column])
            baseline = np.repeat(current[indices][None, :], PRED_LEN, axis=0)
            indicator_type = self.indicator_types[indicator]

            if indicator_type == "distribution":
                validity_column = self.schema["validity_columns"][indicator]
                validity_index = self.column_to_position[validity_column]
                model_validity = (
                    prediction_scaled[:, validity_index] * self.scale[validity_index]
                    + self.mean[validity_index]
                )
                if validity_column in self.constant_columns:
                    model_validity.fill(float(self.constant_columns[validity_column]))
                baseline_validity = np.repeat(current[validity_index], PRED_LEN)
                fallback = baseline.copy()
                model_values = _normalize_distribution(
                    model_values, model_validity >= 0.5, fallback
                )
                baseline = _normalize_distribution(
                    baseline, baseline_validity >= 0.5, baseline
                )
                validity_alpha = np.asarray(
                    self.calibration["validity_alpha_by_horizon"][indicator], dtype=np.float64
                )
                calibrated_valid = (
                    baseline_validity
                    + validity_alpha * (model_validity - baseline_validity)
                ) >= 0.5
            elif indicator_type == "count":
                model_values = np.maximum(np.expm1(model_values), 0.0)
                baseline = np.maximum(np.expm1(baseline), 0.0)
            elif indicator_type == "bounded_index":
                model_values = np.clip(model_values, 0.0, 1.0)
                baseline = np.clip(baseline, 0.0, 1.0)

            alpha = np.asarray(
                self.calibration["indicator_alpha_by_horizon"][indicator], dtype=np.float64
            )[:, None]
            output_values = baseline + alpha * (model_values - baseline)
            if indicator_type == "distribution":
                output_values = _normalize_distribution(output_values, calibrated_valid, fallback)
            elif indicator_type == "count":
                output_values = np.rint(np.maximum(output_values, 0.0))
            elif indicator_type == "bounded_index":
                output_values = np.clip(output_values, 0.0, 1.0)

            prefix = f"{indicator}__"
            for horizon in range(PRED_LEN):
                if len(columns) == 1 and (columns[0] == indicator or columns[0] == "OT"):
                    value: Any = output_values[horizon, 0]
                    rows[horizon][indicator] = int(value) if indicator_type == "count" else float(value)
                else:
                    mapping: Dict[str, Any] = OrderedDict()
                    for offset, column in enumerate(columns):
                        if not column.startswith(prefix):
                            raise RuntimeError(f"Invalid schema column {column} for {indicator}")
                        value = output_values[horizon, offset]
                        mapping[column[len(prefix):]] = (
                            int(value) if indicator_type == "count" else float(value)
                        )
                    rows[horizon][indicator] = mapping
        return rows

    def normalize_actual_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Return actual values using exactly the same public units as forecasts."""
        result: Dict[str, Any] = OrderedDict(timestep=str(row["timestep"]))
        for indicator in self.raw_store.indicator_order:
            raw_value = row.get(indicator)
            if indicator in EXCLUDED_INDICATORS:
                result[indicator] = raw_value
                continue
            columns = list(self.indicator_to_columns[indicator])
            indicator_type = self.indicator_types[indicator]
            if len(columns) == 1:
                value = _parse_raw_number(raw_value)
                if value is None:
                    raise RuntimeError(f"Non-numeric actual value: {indicator}={raw_value!r}")
                if indicator_type == "count":
                    result[indicator] = int(round(max(value, 0.0)))
                elif indicator_type == "bounded_index":
                    result[indicator] = float(np.clip(value, 0.0, 1.0))
                else:
                    result[indicator] = float(value)
                continue

            if not isinstance(raw_value, Mapping):
                raise RuntimeError(f"Expected actual mapping: {indicator}")
            canonical = {_canonical_key(key): value for key, value in raw_value.items()}
            prefix = f"{indicator}__"
            parsed, subkeys = [], []
            for column in columns:
                subkey = column[len(prefix):]
                value = _parse_raw_number(
                    canonical.get(subkey, 0.0),
                    percentage=indicator in self.percentage_indicators,
                )
                if value is None:
                    raise RuntimeError(f"Non-numeric actual mapping value: {indicator}/{subkey}")
                parsed.append(value)
                subkeys.append(subkey)
            values = np.asarray(parsed, dtype=np.float64)
            if indicator_type == "distribution":
                total = float(values.sum())
                values = values / total if total > 1e-12 else np.zeros_like(values)
            elif indicator_type == "count":
                values = np.rint(np.maximum(values, 0.0))
            result[indicator] = OrderedDict(
                (subkey, int(value) if indicator_type == "count" else float(value))
                for subkey, value in zip(subkeys, values)
            )
        return result


class ForecastService:
    def __init__(self, device_name: Optional[str] = None):
        requested_device = device_name or os.getenv("FORECAST_DEVICE")
        self.device = torch.device(
            requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FORECAST_DEVICE requests CUDA, but CUDA is unavailable")
        self._inference_lock = threading.Lock()
        self.schema: Dict[str, Any] = {}
        self.events: Dict[str, EventRuntime] = {}
        self.alias_to_key: Dict[str, str] = {}

    def load(self) -> None:
        required_paths = [FEATURE_SCHEMA_PATH]
        for spec in EVENT_SPECS.values():
            required_paths.extend(
                [spec.processed_csv, spec.raw_json, spec.checkpoint,
                 spec.preprocessor, spec.calibration]
            )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"Required files are missing: {missing}")

        schema = _load_json(FEATURE_SCHEMA_PATH)
        expected = {
            "schema_version": 2,
            "raw_indicator_count": 48,
            "retained_indicator_count": PREDICTED_INDICATOR_COUNT,
            "excluded_indicator_count": RAW_ACTUAL_INDICATOR_COUNT,
            "business_variable_count": NUM_BUSINESS_VARIATES,
            "auxiliary_variable_count": NUM_AUXILIARY_VARIATES,
            "model_variable_count": NUM_MODEL_VARIATES,
        }
        for key, value in expected.items():
            if schema.get(key) != value:
                raise RuntimeError(f"feature_schema.json: expected {key}={value}")
        schema_excluded = tuple(item["name"] for item in schema["excluded_indicators"])
        if schema_excluded != EXCLUDED_INDICATORS:
            raise RuntimeError("Excluded indicator list does not match feature_schema.json")

        LOGGER.info("Loading three calibrated V2 SDformer models on %s", self.device)
        for key, spec in EVENT_SPECS.items():
            self.events[key] = EventRuntime(spec, schema, self.device)
            for alias in (key, spec.display_name, *spec.aliases):
                self.alias_to_key[alias.strip().lower()] = key
        self.schema = schema
        LOGGER.info("All calibrated V2 SDformer models loaded successfully")

    def resolve_event(self, value: str) -> EventRuntime:
        key = self.alias_to_key.get(value.strip().lower())
        if key is None:
            raise ServiceError(
                "不支持的事件 p", status_code=400, received=value,
                supported_events=[spec.display_name for spec in EVENT_SPECS.values()],
            )
        return self.events[key]

    def predict(self, event_value: str, time_value: str, history_hours: int) -> Dict[str, Any]:
        runtime = self.resolve_event(event_value)
        target_time = _parse_hour(time_value)
        runtime.validate_target_time(target_time)
        history_start = target_time - pd.Timedelta(hours=history_hours - 1)
        history_times = _hour_range(history_start, history_hours)
        future_times = _hour_range(target_time + pd.Timedelta(hours=1), PRED_LEN)
        history_raw_rows = runtime.raw_store.rows(history_times)
        future_actual_rows = runtime.raw_store.future_actual_rows(future_times)
        with self._inference_lock:
            predicted_rows = runtime.predict(target_time)

        history_items = [runtime.normalize_actual_row(row) for row in history_raw_rows]
        future_items: List[Dict[str, Any]] = []
        raw_order = runtime.raw_store.indicator_order
        for timestamp, predicted, actual in zip(future_times, predicted_rows, future_actual_rows):
            merged: Dict[str, Any] = OrderedDict(timestep=_format_time(timestamp))
            for indicator in raw_order:
                merged[indicator] = (
                    actual[indicator] if indicator in EXCLUDED_INDICATORS else predicted[indicator]
                )
            future_items.append(merged)

        return {
            "code": 0, "message": "success",
            "data": {
                "event": {"key": runtime.spec.key, "name": runtime.spec.display_name},
                "request": {"time": _format_time(target_time), "history_hours": history_hours},
                "history": {
                    "source": "actual", "indicator_count": 48,
                    "start": _format_time(history_times[0]), "end": _format_time(history_times[-1]),
                    "points": len(history_items), "items": history_items,
                },
                "future": {
                    "indicator_count": 48,
                    "predicted_indicator_count": PREDICTED_INDICATOR_COUNT,
                    "actual_indicator_count": RAW_ACTUAL_INDICATOR_COUNT,
                    "business_variable_count": NUM_BUSINESS_VARIATES,
                    "model_internal_variable_count": NUM_MODEL_VARIATES,
                    "start": _format_time(future_times[0]), "end": _format_time(future_times[-1]),
                    "points": len(future_items),
                    "predicted_indicators": list(runtime.indicator_to_columns),
                    "actual_indicators": list(EXCLUDED_INDICATORS),
                    "items": future_items,
                },
                "units": {
                    "distribution": "0-1 ratio; components sum to 0 or 1",
                    "count": "non-negative integer", "bounded_index": "0-1",
                    "unbounded_index": "numeric", "excluded_actual": "raw JSON value",
                },
                "model": {
                    "name": "SDformer", "version": "v2-calibrated",
                    "checkpoint": runtime.spec.checkpoint_dir.name, "device": str(self.device),
                    "input_hours": SEQ_LEN, "forecast_hours": PRED_LEN,
                    "business_variables": NUM_BUSINESS_VARIATES,
                    "internal_variables": NUM_MODEL_VARIATES,
                },
            },
        }

    def event_summaries(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": runtime.spec.key, "name": runtime.spec.display_name,
                "checkpoint": runtime.spec.checkpoint_dir.name,
                "model_version": "v2-calibrated", "raw_json_cached": runtime.raw_store.cached,
                **runtime.available_range(),
            }
            for runtime in self.events.values()
        ]


@asynccontextmanager
async def lifespan(application: FastAPI):
    service = ForecastService()
    service.load()
    application.state.forecast_service = service
    yield
    application.state.forecast_service = None
    if service.device.type == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="SDformer Event Forecast API", version="2.0.0", lifespan=lifespan)
cors_value = os.getenv("CORS_ALLOW_ORIGINS", "*")
cors_origins = [item.strip() for item in cors_value.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"], allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _service(request: Request) -> ForecastService:
    service = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail={"message": "服务尚未就绪"})
    return service


def _raise_http(error: ServiceError) -> None:
    raise HTTPException(
        status_code=error.status_code,
        detail={"message": error.message, **error.details},
    ) from error


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    service = _service(request)
    return {
        "status": "ok", "version": "v2-calibrated", "device": str(service.device),
        "models_loaded": len(service.events),
        "events": [runtime.spec.display_name for runtime in service.events.values()],
    }


@app.get("/api/v1/events")
def events(request: Request) -> Dict[str, Any]:
    return {"events": _service(request).event_summaries()}


@app.get("/api/v1/schema")
def schema(request: Request) -> Dict[str, Any]:
    return _service(request).schema


@app.post("/api/v1/predict")
@app.post("/predict", include_in_schema=False)
def predict(payload: PredictRequest, request: Request) -> Dict[str, Any]:
    service = _service(request)
    try:
        return service.predict(payload.p, payload.t, payload.w)
    except ServiceError as error:
        _raise_http(error)
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail={"message": "模型预测失败", "error": str(error)}
        ) from error
