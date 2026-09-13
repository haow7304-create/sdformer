# -*- coding: utf-8 -*-
"""
原始数据列数据类型统计脚本
- 统计每列的数据类型（按实际内容推断）
- 每种类型输出一个示例（列名 + 示例值）
- 将统计结果保存到文件
"""

import os
import re
import json
import ast
import argparse
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# 默认路径 train_data_file3.csv 得闲谨制_最小指标集_原始.csv
DEFAULT_RAW_NAME = "train_data_file1.csv"
DEFAULT_OUTPUT_NAME = "column_dtype_stats.json"
ENCODINGS = ["utf-8", "gbk", "utf-8-sig"]


def _load_csv(path: str) -> pd.DataFrame:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"无法用 {ENCODINGS} 解码文件: {path}")


def _infer_cell_type(cell: Any) -> str:
    """
    根据单元格内容推断语义类型。
    返回: datetime | int | float | list_float | list_int | list_str | str | empty
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return "empty"

    if isinstance(cell, (int, np.integer)):
        return "int"
    if isinstance(cell, (float, np.floating)):
        return "float"

    s = str(cell).strip()
    if not s:
        return "empty"

    # 尝试解析为 datetime
    if re.match(r"^\d{4}-\d{2}-\d{2}", s) or re.match(r"^\d{4}/\d{2}/\d{2}", s):
        try:
            pd.to_datetime(s)
            return "datetime"
        except Exception:
            pass

    # 尝试解析为数字列表 [0.1, 0.2] 或 [1, 2]
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                first = parsed[0]
                if isinstance(first, (int, float)):
                    return "list_float" if isinstance(first, float) else "list_int"
                if isinstance(first, str):
                    return "list_str"
            elif isinstance(parsed, list):
                return "list_empty"
        except (ValueError, SyntaxError, TypeError):
            pass

    # 尝试纯数字
    try:
        v = float(s)
        if v == int(v):
            return "int"
        return "float"
    except (ValueError, TypeError):
        pass

    return "str"


def _get_column_type(series: pd.Series) -> Tuple[str, Any]:
    """
    根据整列非空值推断列的类型；若存在多种类型则取最常见且非 empty 的类型。
    返回 (类型名, 示例值)。
    """
    type_counts = defaultdict(int)
    examples: Dict[str, Any] = {}

    for v in series.dropna().head(500):  # 采样前 500 个非空值
        t = _infer_cell_type(v)
        type_counts[t] += 1
        if t not in examples:
            # 示例截断，避免过长
            raw = v if not isinstance(v, str) or len(v) <= 200 else v[:200] + "..."
            examples[t] = raw

    # 优先选非 empty 且出现最多的类型
    candidates = [(c, t) for t, c in type_counts.items() if t != "empty"]
    if not candidates:
        return "empty", None
    dtype = max(candidates, key=lambda x: x[0])[1]
    example = examples.get(dtype)
    return dtype, example


def stats_column_dtypes(
    data_path: str,
    output_path: str,
    sample_example_len: int = 200,
) -> Dict[str, Any]:
    """
    统计原始数据中各列的数据类型，每种类型输出一个示例，并保存结果。

    Parameters
    ----------
    data_path : str
        原始数据文件路径（如 CSV）。
    output_path : str
        统计结果保存路径（如 .json 或 .txt）。
    sample_example_len : int
        示例值字符串最大长度，超过则截断。

    Returns
    -------
    dict
        统计结果，包含列名->类型、按类型分组的列、每类示例等。
    """
    df = _load_csv(data_path)
    columns = list(df.columns)

    # 列 -> (类型, 示例)
    col_types: Dict[str, Tuple[str, Any]] = {}
    for col in columns:
        col_types[col] = _get_column_type(df[col])

    # 按类型分组
    type_to_columns: Dict[str, List[str]] = defaultdict(list)
    for col, (dtype, _) in col_types.items():
        type_to_columns[dtype].append(col)

    # 每种类型取一个示例：取该类型下第一列的示例值
    type_to_example: Dict[str, Dict[str, Any]] = {}
    for dtype, cols in type_to_columns.items():
        col = cols[0]
        _, example = col_types[col]
        if example is not None and isinstance(example, str) and len(example) > sample_example_len:
            example = example[:sample_example_len] + "..."
        type_to_example[dtype] = {
            "column": col,
            "example": example,
        }

    result = {
        "source_file": os.path.abspath(data_path),
        "total_columns": len(columns),
        "column_dtypes": {col: dtype for col, (dtype, _) in col_types.items()},
        "columns_by_type": dict(type_to_columns),
        "example_per_type": type_to_example,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def print_summary(result: Dict[str, Any]) -> None:
    """在控制台打印每种类型及一个示例。"""
    print("=" * 60)
    print("列数据类型统计摘要")
    print("=" * 60)
    print(f"数据文件: {result['source_file']}")
    print(f"总列数: {result['total_columns']}\n")

    for dtype, info in result["example_per_type"].items():
        cols = result["columns_by_type"][dtype]
        print(f"【{dtype}】 共 {len(cols)} 列")
        print(f"  示例列: {info['column']}")
        ex = info["example"]
        if ex is not None:
            ex_str = str(ex)
            if len(ex_str) > 120:
                ex_str = ex_str[:120] + "..."
            print(f"  示例值: {ex_str}")
        else:
            print("  示例值: (空)")
        print()
    print("完整结果已保存到:", result.get("_output_path", "(未设置)"))


def main():
    parser = argparse.ArgumentParser(description="统计原始数据列数据类型并保存结果")
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_RAW_NAME),
        help="原始数据 CSV 路径",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_OUTPUT_NAME),
        help="统计结果输出路径（JSON）",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="不打印摘要到控制台",
    )
    args = parser.parse_args()

    result = stats_column_dtypes(args.input, args.output)
    result["_output_path"] = os.path.abspath(args.output)

    if not args.no_print:
        print_summary(result)

    return result


if __name__ == "__main__":
    main()
