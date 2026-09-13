#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将三个事件 JSON 转换为具有统一字段结构的 SDformer 训练 CSV。

处理原则：
1. 原始 48 个指标中，排除用户 ID、关键词、平台类别、字符串时间和离散阶段等 7 个指标。
2. 保留 41 个适合回归的数值指标；数值字典按固定子项展开。
3. 三个数据集共同建立一份字段结构，输出列名、列数和顺序完全一致。
4. 舆情传播阶段作为离散业务字段排除，由后端从原始 JSON 读取真实值。
5. 百分数统一转换为 0～1 比例，长尾计数使用 log1p，分布增加有效性标记。
6. 输出格式为 ``date, feature_1, ..., OT``，可直接交给 Dataset_Custom。

默认用法：
    python3 dataset/process_json_time_series.py

指定输入或输出目录：
    python3 dataset/process_json_time_series.py \
        dataset/俄乌冲突_raw_time_series_data.json \
        --output-dir dataset/processed_v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERN = "*_raw_time_series_data.json"
DEFAULT_TARGET = "ch2_2_1_1_热度指数_calc_2_1_1"
DEFAULT_TARGET_NAME = "OT"
SEPARATOR = "__"

EXPECTED_RAW_INDICATORS = 48
EXPECTED_RETAINED_INDICATORS = 41
EXPECTED_BUSINESS_VARIABLES = 309
EXPECTED_AUXILIARY_VARIABLES = 11
EXPECTED_NUMERIC_VARIABLES = 320
SCHEMA_VERSION = 2


# 这些字段的语义是 ID、文本、类别或字符串日期，不构造数值时序变量。
EXCLUDED_INDICATORS: "OrderedDict[str, str]" = OrderedDict(
    [
        (
            "ch1_1_2_3_最具影响力用户_calc_1_2_3",
            "用户 ID 列表，跨事件没有稳定的连续数值含义",
        ),
        (
            "ch1_1_2_4_最具正能量用户_calc_1_2_4",
            "用户 ID 列表，跨事件无法使用统一字段表示",
        ),
        (
            "ch1_1_2_5_最具负能量用户_calc_1_2_5",
            "用户 ID 列表，跨事件无法使用统一字段表示",
        ),
        (
            "ch2_2_2_4_相关热搜关键词_calc_2_2_4",
            "文本关键词列表，当前数值时序模型不处理文本语义",
        ),
        (
            "ch3_3_2_1_最具热度平台_calc_3_2_1",
            "平台类别字段，且原始记录存在字典和列表混用",
        ),
        (
            "ch3_3_4_1_平台首次响应时间_calc_3_4_1",
            "YYYY-MM 字符串日期，在单个事件中基本不随时间变化",
        ),
        (
            "ch1_1_3_5_舆情传播的阶段_calc_1_3_5",
            "离散阶段变量，不适合作为连续时序回归目标；由原始 JSON 返回真实值",
        ),
    ]
)


# 这些分布在原始数据中使用百分数（包括带 % 和不带 % 两种形式）。
# 统一除以 100，转换为 0～1 比例。
PERCENTAGE_DISTRIBUTIONS = {
    "ch1_1_1_1_国内用户地域分布_calc_1_1_1",
    "ch1_1_1_2_国际用户地域分布_calc_1_1_2",
    "ch1_1_1_3_用户年龄分布_calc_1_1_3",
    "ch1_1_1_4_用户性别分布_calc_1_1_4",
    "ch1_1_1_5_用户兴趣分布_calc_1_1_5",
    "ch1_1_1_6_用户职业分布_calc_1_1_6",
    "ch1_1_1_7_用户受教育程度分布_calc_1_1_7",
    "ch1_1_1_8_用户收入水平分布_calc_1_1_8",
    "ch1_1_1_9_用户政治倾向分布_calc_1_1_9",
}


# 这些分布转换后每个时刻各子项之和应约等于 1，用于质量检查。
UNIT_SUM_DISTRIBUTIONS = PERCENTAGE_DISTRIBUTIONS | {
    "ch2_2_5_1_信息情绪极性分布_calc_2_5_1",
    "ch3_3_3_3_跨平台情感分布_calc_3_3_3",
}


# 数量变量先做 log1p，再进入 StandardScaler，减轻长尾峰值对训练的影响。
COUNT_INDICATORS = {
    "ch1_1_2_1_五类用户（意见领袖、普通网民、当事人、官方媒体和网络媒体）影响力_calc_1_2_1",
    "ch1_1_2_2_五类用户（意见领袖、普通网民、当事人、官方媒体和网络媒体）活跃度_calc_1_2_2",
    "ch1_1_3_1_潜在受众数量_calc_1_3_1",
    "ch1_1_3_2_知情者数量_calc_1_3_2",
    "ch1_1_3_3_传播者数量_calc_1_3_3",
    "ch1_1_3_4_停滞者数量_calc_1_3_4",
    "ch2_2_7_2_敏感话题数量_calc_2_7_2",
    "ch2_2_7_3_敏感用户（重点监控对象）数量_calc_2_7_3",
    "ch3_3_4_2_平台响应用户数量_calc_3_4_2",
    "ch3_3_4_3_平台响应互动数量_calc_3_4_3",
}


BOUNDED_INDEX_INDICATORS = {
    "ch2_2_1_1_热度指数_calc_2_1_1",
    "ch2_2_1_2_争议性指数_calc_2_1_2",
    "ch2_2_1_3_影响力指数_calc_2_1_3",
    "ch2_2_1_4_参与性指数_calc_2_1_4",
    "ch2_2_2_1_传播速度_calc_2_2_1",
    "ch2_2_2_2_覆盖范围_calc_2_2_2",
    "ch2_2_2_3_内容载体多样性_calc_2_2_3",
    "ch2_2_3_1_最强关联信息_calc_2_3_1",
    "ch2_2_4_1_可信性指数_calc_2_4_1",
    "ch2_2_6_1_地理聚集指数_calc_2_6_1",
    "ch2_2_6_2_人群聚集指数_calc_2_6_2",
    "ch2_2_7_1_敏感性指数_calc_2_7_1",
    "ch2_2_8_1_AI生成内容检测率_calc_2_8_1",
    "ch2_2_8_2_AI合成内容传播指数_calc_2_8_2",
    "ch3_3_1_2_跨平台扩展潜力_calc_3_1_2",
    "ch3_3_3_1_跨平台传播速度_calc_3_3_1",
    "ch3_3_3_2_跨平台情绪协调性_calc_3_3_2",
    "ch3_3_3_4_跨平台争议性分布_calc_3_3_4",
}


UNBOUNDED_INDEX_INDICATORS = {
    "ch3_3_1_1_平台类型多样性_calc_3_1_1",
    "ch2_2_6_3_平台聚集指数_calc_2_6_3",
}


def _validity_column(indicator: str) -> str:
    return f"{indicator}{SEPARATOR}valid"


# 三份数据对同一情绪类别使用了两个词，统一为一个字段。
KEY_ALIASES = {"中性": "中立"}

NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
MISSING_STRINGS = {"", "null", "none", "nan", "na", "n/a", "-"}


@dataclass
class IndicatorSpec:
    name: str
    kind: Optional[str] = None  # scalar 或 mapping
    subkeys: "OrderedDict[str, None]" = field(default_factory=OrderedDict)

    @property
    def columns(self) -> List[str]:
        if self.kind == "scalar":
            return [self.name]
        if self.kind == "mapping":
            return [f"{self.name}{SEPARATOR}{key}" for key in self.subkeys]
        return []


def _canonical_key(value: object) -> str:
    key = str(value).strip()
    return KEY_ALIASES.get(key, key)


def _parse_numeric(value: Any, indicator: str) -> Optional[float]:
    """把语义数值转换为有限 float，并统一百分数单位。"""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, (bool, np.bool_)):
        result = float(value)
        had_percent_sign = False
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        had_percent_sign = False
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
    if indicator in PERCENTAGE_DISTRIBUTIONS or had_percent_sign:
        result /= 100.0
    return result


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: 顶层必须是非空 JSON 数组")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: 数组元素必须全部为 JSON 对象")
    return rows


def _raw_indicator_order(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return [key for key in rows[0] if key != "timestep"]


def _validate_raw_indicator_set(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    expected_order: Optional[Sequence[str]],
) -> List[str]:
    order = _raw_indicator_order(rows)
    if len(order) != EXPECTED_RAW_INDICATORS:
        raise ValueError(
            f"{path}: 预期 {EXPECTED_RAW_INDICATORS} 个原始指标，实际 {len(order)} 个"
        )
    missing_exclusions = set(EXCLUDED_INDICATORS) - set(order)
    if missing_exclusions:
        raise ValueError(f"{path}: 找不到预期排除指标 {sorted(missing_exclusions)}")
    if expected_order is not None and set(order) != set(expected_order):
        missing = sorted(set(expected_order) - set(order))
        extra = sorted(set(order) - set(expected_order))
        raise ValueError(f"{path}: 指标集合不一致；缺少={missing}，新增={extra}")
    return order


def _observe_indicator(spec: IndicatorSpec, value: Any, path: Path) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        if spec.kind not in (None, "mapping"):
            raise ValueError(f"{path}: 保留指标 {spec.name} 同时出现标量和字典")
        spec.kind = "mapping"
        for key, child in value.items():
            canonical = _canonical_key(key)
            if isinstance(child, (dict, list)):
                raise ValueError(f"{path}: 保留指标 {spec.name}/{canonical} 不是标量")
            if child is not None and _parse_numeric(child, spec.name) is None:
                raise ValueError(
                    f"{path}: 保留指标 {spec.name}/{canonical} 含非数值 {child!r}"
                )
            spec.subkeys.setdefault(canonical, None)
        return
    if isinstance(value, list):
        raise ValueError(f"{path}: 保留指标 {spec.name} 意外出现列表")
    if spec.kind not in (None, "scalar"):
        raise ValueError(f"{path}: 保留指标 {spec.name} 同时出现字典和标量")
    spec.kind = "scalar"
    if _parse_numeric(value, spec.name) is None:
        raise ValueError(f"{path}: 保留指标 {spec.name} 含非数值 {value!r}")


def build_shared_schema(inputs: Sequence[Path]) -> Tuple[List[str], List[IndicatorSpec]]:
    """扫描全部输入，建立三份输出共用的固定数值字段结构。"""
    raw_order: Optional[List[str]] = None
    retained_order: List[str] = []
    specs: "OrderedDict[str, IndicatorSpec]" = OrderedDict()

    for path in inputs:
        rows = _read_rows(path)
        current_order = _validate_raw_indicator_set(path, rows, raw_order)
        if raw_order is None:
            raw_order = current_order
            retained_order = [
                name for name in raw_order if name not in EXCLUDED_INDICATORS
            ]
            if len(retained_order) != EXPECTED_RETAINED_INDICATORS:
                raise ValueError(
                    f"应保留 {EXPECTED_RETAINED_INDICATORS} 个指标，实际 {len(retained_order)} 个"
                )
            specs = OrderedDict(
                (name, IndicatorSpec(name=name)) for name in retained_order
            )

        for row in rows:
            for name in retained_order:
                _observe_indicator(specs[name], row.get(name), path)

    assert raw_order is not None
    untyped = [spec.name for spec in specs.values() if spec.kind is None]
    if untyped:
        raise ValueError(f"以下保留指标在所有输入中均为空: {untyped}")

    variable_count = sum(len(spec.columns) for spec in specs.values())
    if variable_count != EXPECTED_BUSINESS_VARIABLES:
        raise ValueError(
            f"统一字段应为 {EXPECTED_BUSINESS_VARIABLES} 个业务数值变量，实际 {variable_count} 个；"
            "请检查原始字典子项是否发生变化"
        )
    return raw_order, list(specs.values())


def _flatten_rows(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[IndicatorSpec]
) -> pd.DataFrame:
    business_columns = [column for spec in specs for column in spec.columns]
    validity_columns = [
        _validity_column(spec.name)
        for spec in specs
        if spec.name in UNIT_SUM_DISTRIBUTIONS
    ]
    columns = [*business_columns, *validity_columns]
    flattened: List[Dict[str, Any]] = []

    for row in rows:
        output: Dict[str, Any] = {"date": row.get("timestep")}
        for spec in specs:
            value = row.get(spec.name)
            if spec.kind == "scalar":
                parsed = _parse_numeric(value, spec.name)
                if parsed is not None and spec.name in COUNT_INDICATORS:
                    if parsed < 0:
                        raise ValueError(f"数量指标 {spec.name} 不能为负数: {parsed}")
                    parsed = math.log1p(parsed)
                output[spec.name] = parsed
                continue

            if value is None:
                for column in spec.columns:
                    output[column] = np.nan
                continue
            if not isinstance(value, dict):
                raise ValueError(f"指标 {spec.name} 预期为字典，实际为 {type(value).__name__}")

            canonical_values = {
                _canonical_key(key): child for key, child in value.items()
            }
            parsed_columns: List[str] = []
            for key in spec.subkeys:
                column = f"{spec.name}{SEPARATOR}{key}"
                # 父字典存在但某分类键不存在时，该分类按 0 处理。
                parsed = _parse_numeric(
                    canonical_values.get(key, 0.0), spec.name
                )
                if parsed is not None and spec.name in COUNT_INDICATORS:
                    if parsed < 0:
                        raise ValueError(f"数量指标 {spec.name}/{key} 不能为负数: {parsed}")
                    parsed = math.log1p(parsed)
                output[column] = parsed
                parsed_columns.append(column)

            if spec.name in UNIT_SUM_DISTRIBUTIONS:
                if any(output[column] is None for column in parsed_columns):
                    output[_validity_column(spec.name)] = None
                else:
                    total = sum(output[column] for column in parsed_columns)
                    if total < 0:
                        raise ValueError(f"分布指标 {spec.name} 的分量和不能为负数")
                    output[_validity_column(spec.name)] = float(total > 0)
                    if total > 0:
                        for column in parsed_columns:
                            output[column] /= total
        flattened.append(output)

    return pd.DataFrame(flattened, columns=["date", *columns])


def _distribution_checks(
    frame: pd.DataFrame, specs: Sequence[IndicatorSpec]
) -> Dict[str, Dict[str, float]]:
    checks: Dict[str, Dict[str, float]] = {}
    for spec in specs:
        if spec.name not in UNIT_SUM_DISTRIBUTIONS or spec.kind != "mapping":
            continue
        sums = frame[spec.columns].sum(axis=1)
        nonzero_sums = sums[sums.ne(0.0)]
        deviations = (nonzero_sums - 1.0).abs()
        checks[spec.name] = {
            "minimum_sum": float(sums.min()),
            "maximum_sum": float(sums.max()),
            "zero_sum_rows": int(sums.eq(0.0).sum()),
            "maximum_abs_deviation_from_one_excluding_zero_rows": (
                float(deviations.max()) if len(deviations) else 0.0
            ),
        }
    return checks


def _output_name(input_path: Path) -> str:
    suffix = "_raw_time_series_data"
    stem = input_path.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}.csv"


def process_file(
    input_path: Path,
    output_path: Path,
    specs: Sequence[IndicatorSpec],
    target: str,
    target_name: str,
    duplicate_policy: str,
) -> Dict[str, Any]:
    print(f"\n正在处理: {input_path.name}")
    rows = _read_rows(input_path)
    frame = _flatten_rows(rows, specs)
    input_rows = len(frame)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    invalid_date_rows = int(frame["date"].isna().sum())
    if invalid_date_rows:
        raise ValueError(f"{input_path}: 存在 {invalid_date_rows} 行无效 timestep")

    frame = frame.replace([np.inf, -np.inf], np.nan).sort_values("date")
    duplicate_rows = int(frame.duplicated("date", keep=False).sum())
    if duplicate_rows:
        if duplicate_policy == "error":
            raise ValueError(f"{input_path}: 存在 {duplicate_rows} 行重复时间戳")
        if duplicate_policy == "mean":
            frame = frame.groupby("date", as_index=False, sort=True).mean(numeric_only=True)
        else:
            frame = frame.drop_duplicates("date", keep=duplicate_policy)

    if target not in frame.columns:
        raise ValueError(f"{input_path}: 找不到目标字段 {target!r}")
    if target_name != target and target_name in frame.columns:
        raise ValueError(f"{input_path}: 输出目标名 {target_name!r} 与已有字段冲突")

    value_columns = [column for column in frame.columns if column != "date"]
    incomplete_mask = frame[value_columns].isna().any(axis=1)
    incomplete_dates = frame.loc[incomplete_mask, "date"]
    dropped_incomplete_rows = int(incomplete_mask.sum())
    frame = frame.loc[~incomplete_mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{input_path}: 删除不完整记录后没有可用数据")

    date_diffs = frame["date"].diff().dropna()
    expected_interval = pd.Timedelta(hours=1)
    irregular_gap_count = int((date_diffs != expected_interval).sum())
    if irregular_gap_count:
        irregular = frame.loc[
            frame["date"].diff().ne(expected_interval) & frame.index.to_series().ne(0),
            "date",
        ].head(5)
        raise ValueError(
            f"{input_path}: 删除不完整记录后产生 {irregular_gap_count} 个非小时连续间隔；"
            f"示例={irregular.dt.strftime('%Y-%m-%d %H:%M:%S').tolist()}"
        )

    distribution_checks = _distribution_checks(frame, specs)

    frame = frame.rename(columns={target: target_name})
    feature_columns = [
        column for column in frame.columns if column not in ("date", target_name)
    ]
    frame = frame[["date", *feature_columns, target_name]]

    model_columns = list(frame.columns[1:])
    numeric = frame[model_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{input_path}: 输出仍包含 NaN 或无穷值")
    if len(frame.columns) - 1 != EXPECTED_NUMERIC_VARIABLES:
        raise ValueError(
            f"{input_path}: 输出变量数应为 {EXPECTED_NUMERIC_VARIABLES}，"
            f"实际 {len(frame.columns) - 1}"
        )

    constant_columns = {
        column: float(frame[column].iloc[0])
        for column in model_columns
        if frame[column].nunique(dropna=False) <= 1
    }
    num_train = int(len(frame) * 0.7)
    scaler = StandardScaler().fit(numeric[:num_train])

    preprocessor: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(input_path),
        "output": str(output_path),
        "model_variable_count": EXPECTED_NUMERIC_VARIABLES,
        "business_variable_count": EXPECTED_BUSINESS_VARIABLES,
        "auxiliary_variable_count": EXPECTED_AUXILIARY_VARIABLES,
        "model_columns": model_columns,
        "log1p_columns": [
            target_name if column == target else column
            for spec in specs
            if spec.name in COUNT_INDICATORS
            for column in spec.columns
        ],
        "distribution_columns": {
            spec.name: list(spec.columns)
            for spec in specs
            if spec.name in UNIT_SUM_DISTRIBUTIONS
        },
        "validity_columns": {
            spec.name: _validity_column(spec.name)
            for spec in specs
            if spec.name in UNIT_SUM_DISTRIBUTIONS
        },
        "bounded_indicators": sorted(BOUNDED_INDEX_INDICATORS),
        "constant_columns": constant_columns,
        "scaler": {
            "fit_rows": num_train,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    preprocessor_path = output_path.with_suffix(".preprocessor.json")
    with preprocessor_path.open("w", encoding="utf-8") as handle:
        json.dump(preprocessor, handle, ensure_ascii=False, indent=2)

    metadata: Dict[str, Any] = {
        "source": str(input_path),
        "output": str(output_path),
        "schema_version": SCHEMA_VERSION,
        "raw_indicator_count": EXPECTED_RAW_INDICATORS,
        "retained_indicator_count": EXPECTED_RETAINED_INDICATORS,
        "excluded_indicator_count": len(EXCLUDED_INDICATORS),
        "business_variable_count": EXPECTED_BUSINESS_VARIABLES,
        "auxiliary_variable_count": EXPECTED_AUXILIARY_VARIABLES,
        "model_variable_count": EXPECTED_NUMERIC_VARIABLES,
        "input_rows": input_rows,
        "output_rows": len(frame),
        "duplicate_rows_found": duplicate_rows,
        "dropped_incomplete_rows": dropped_incomplete_rows,
        "dropped_incomplete_date_start": (
            incomplete_dates.min().strftime("%Y-%m-%d %H:%M:%S")
            if len(incomplete_dates)
            else None
        ),
        "dropped_incomplete_date_end": (
            incomplete_dates.max().strftime("%Y-%m-%d %H:%M:%S")
            if len(incomplete_dates)
            else None
        ),
        "date_start": frame["date"].iloc[0],
        "date_end": frame["date"].iloc[-1],
        "frequency": "1h",
        "irregular_gap_count": irregular_gap_count,
        "target_source": target,
        "target_output": target_name,
        "constant_columns_retained_for_shared_schema": list(constant_columns),
        "distribution_sum_checks": distribution_checks,
        "preprocessor": str(preprocessor_path),
        "columns": list(frame.columns),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(
        f"完成: {output_path} | {len(frame)} 行, {EXPECTED_NUMERIC_VARIABLES} 个数值变量, "
        f"删除不完整记录={dropped_incomplete_rows}, 重复时间行={duplicate_rows}"
    )
    return metadata


def write_feature_schema(
    output_dir: Path,
    inputs: Sequence[Path],
    raw_order: Sequence[str],
    specs: Sequence[IndicatorSpec],
    target: str,
    target_name: str,
) -> Path:
    indicator_to_columns = OrderedDict()
    business_columns = []
    for spec in specs:
        renamed = [target_name if column == target else column for column in spec.columns]
        indicator_to_columns[spec.name] = renamed
        business_columns.extend(renamed)

    # 与 CSV 一致：业务字段之后追加分布有效性标记，并将 OT 移到最后。
    business_columns = [column for column in business_columns if column != target_name]
    business_columns.append(target_name)
    validity_columns = OrderedDict(
        (spec.name, _validity_column(spec.name))
        for spec in specs
        if spec.name in UNIT_SUM_DISTRIBUTIONS
    )
    model_columns = [
        *[column for column in business_columns if column != target_name],
        *validity_columns.values(),
        target_name,
    ]

    indicator_types = OrderedDict()
    for spec in specs:
        if spec.name in UNIT_SUM_DISTRIBUTIONS:
            indicator_types[spec.name] = "distribution"
        elif spec.name in COUNT_INDICATORS:
            indicator_types[spec.name] = "count"
        elif spec.name in BOUNDED_INDEX_INDICATORS:
            indicator_types[spec.name] = "bounded_index"
        elif spec.name in UNBOUNDED_INDEX_INDICATORS:
            indicator_types[spec.name] = "unbounded_index"
        else:
            raise ValueError(f"没有为保留指标声明类型: {spec.name}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_files": [str(path) for path in inputs],
        "raw_indicator_count": len(raw_order),
        "retained_indicator_count": len(specs),
        "excluded_indicator_count": len(EXCLUDED_INDICATORS),
        "business_variable_count": len(business_columns),
        "auxiliary_variable_count": len(validity_columns),
        "model_variable_count": len(model_columns),
        "excluded_indicators": [
            {"name": name, "reason": reason}
            for name, reason in EXCLUDED_INDICATORS.items()
        ],
        "retained_indicators": [spec.name for spec in specs],
        "indicator_to_columns": indicator_to_columns,
        "indicator_types": indicator_types,
        "unit_sum_distributions": list(validity_columns),
        "validity_columns": validity_columns,
        "log1p_indicators": sorted(COUNT_INDICATORS),
        "bounded_indicators": sorted(BOUNDED_INDEX_INDICATORS),
        "unbounded_indicators": sorted(UNBOUNDED_INDEX_INDICATORS),
        "percentage_rule": "用户属性百分数统一转换为 0～1 比例",
        "distribution_rule": "分量和为0时 valid=0 且分量保持0；否则 valid=1 并归一化为和1",
        "count_rule": "训练CSV存储 log1p(value)，接口返回时使用 expm1 并取非负整数",
        "missing_value_rule": "不填充；删除包含任一保留变量缺失的记录",
        "constant_column_rule": "保留常数列，保证三个数据集字段结构完全一致",
        "target_source": target,
        "target_output": target_name,
        "business_columns": business_columns,
        "model_columns": model_columns,
        "columns": ["date", *model_columns],
    }
    if len(business_columns) != EXPECTED_BUSINESS_VARIABLES:
        raise ValueError(f"业务变量数错误: {len(business_columns)}")
    if len(validity_columns) != EXPECTED_AUXILIARY_VARIABLES:
        raise ValueError(f"辅助变量数错误: {len(validity_columns)}")
    if len(model_columns) != EXPECTED_NUMERIC_VARIABLES:
        raise ValueError(f"模型变量数错误: {len(model_columns)}")
    path = output_dir / "feature_schema.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _resolve_inputs(raw_inputs: Iterable[str]) -> List[Path]:
    inputs = [Path(value).expanduser().resolve() for value in raw_inputs]
    if not inputs:
        inputs = sorted(SCRIPT_DIR.glob(DEFAULT_PATTERN))
    if not inputs:
        raise FileNotFoundError(
            f"未找到输入文件；请传入 JSON 路径，或把文件放到 {SCRIPT_DIR}/{DEFAULT_PATTERN}"
        )
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"输入文件不存在: {missing}")
    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把48指标事件 JSON 转换为 V2 版统一320内部变量的 SDformer 训练 CSV"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=f"输入 JSON；不传时处理 dataset/{DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "processed_v2",
        help="输出目录（默认: dataset/processed_v2）",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="作为最后一列的保留标量字段（默认: 热度指数）",
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET_NAME,
        help="目标字段输出名（默认: OT）",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("mean", "first", "last", "error"),
        default="mean",
        help="重复时间戳处理方式（默认: mean）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = _resolve_inputs(args.inputs)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描 {len(inputs)} 个数据集并建立统一字段结构……")
    raw_order, specs = build_shared_schema(inputs)

    if args.target in EXCLUDED_INDICATORS:
        raise ValueError(f"目标字段 {args.target!r} 属于已排除的非数值指标")
    scalar_names = {spec.name for spec in specs if spec.kind == "scalar"}
    if args.target not in scalar_names:
        raise ValueError(f"目标字段 {args.target!r} 必须是保留的数值标量指标")

    summaries = []
    for input_path in inputs:
        summaries.append(
            process_file(
                input_path=input_path,
                output_path=output_dir / _output_name(input_path),
                specs=specs,
                target=args.target,
                target_name=args.target_name,
                duplicate_policy=args.duplicate_policy,
            )
        )

    schema_path = write_feature_schema(
        output_dir=output_dir,
        inputs=inputs,
        raw_order=raw_order,
        specs=specs,
        target=args.target,
        target_name=args.target_name,
    )
    print(
        f"\n全部完成：{len(summaries)} 个数据集均为 "
        f"{EXPECTED_BUSINESS_VARIABLES} 个业务变量 + "
        f"{EXPECTED_AUXILIARY_VARIABLES} 个辅助变量"
    )
    print(f"统一字段清单: {schema_path}")


if __name__ == "__main__":
    main()
