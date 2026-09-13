# -*- coding: utf-8 -*-
"""
得闲谨制时序数据预处理脚本
- 将向量列展开为多列参与预测
- 将字符串列通过字典映射为整数值后参与训练
- 输出模型可用的训练 CSV 及字典映射文件
"""

import os
import re
import json
import ast
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union

# 默认路径：脚本所在目录为数据集目录
DEFAULT_RAW_NAME = "得闲谨制_最小指标集_原始.csv"
DEFAULT_OUTPUT_NAME = "train_data.csv"
DEFAULT_XLSX_NAME = "train_data.xlsx"
DEFAULT_MAPPING_NAME = "str_mappings.json"
ENCODINGS = ["utf-8", "gbk", "utf-8-sig"]



# 不参与预测的列：在最终训练数据中去除
EXCLUDE_COLUMNS = [
    "全量用户id（向量）",
    "热搜关键词（向量）",
    "最强关联信息",
    "平台列表（向量）",
]


def _load_csv(path: str) -> pd.DataFrame:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"无法用 {ENCODINGS} 解码文件: {path}")


def _parse_vector_cell(cell: Any) -> Optional[List[float]]:
    """若单元格为数字向量（如 '[0.1, 0.2]' 或 list），返回浮点数列表，否则返回 None。"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, (list, np.ndarray)):
        try:
            return [float(x) for x in cell]
        except (TypeError, ValueError):
            return None
    s = str(cell).strip()
    if not s or s[0] != "[":
        return None
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [float(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def _parse_str_list_cell(cell: Any) -> Optional[str]:
    """若单元格为字符串列表（如 "['a','b']"），返回拼接后的字符串用于映射；否则返回 None 表示按普通字符串处理。"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, str) and cell.strip().startswith("["):
        try:
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return "|".join(str(x).strip() for x in parsed)
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    将列分为：向量列、字符串列（需映射）、已是数值列。
    向量列：首行非空值可解析为数字列表的列。
    字符串列：object 且非向量，或首行解析为 str 列表的列。
    """
    vector_cols = []
    str_cols = []
    numeric_cols = []

    for col in df.columns:
        if col in ("时间", "date"):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            numeric_cols.append(col)
            continue
        first_val = s.iloc[0]
        vec = _parse_vector_cell(first_val)
        if vec is not None:
            vector_cols.append(col)
            continue
        if s.dtype == object or s.dtype.name == "string":
            str_cols.append(col)
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except Exception:
            str_cols.append(col)

    return vector_cols, str_cols, numeric_cols


def expand_vector_columns(df: pd.DataFrame, vector_cols: List[str]) -> pd.DataFrame:
    """将向量列展开为多列，列名为 原列名_0, 原列名_1, ..."""
    out = df.drop(columns=vector_cols, errors="ignore")
    for col in vector_cols:
        rows = []
        max_len = 0
        for v in df[col]:
            parsed = _parse_vector_cell(v)
            if parsed is not None:
                rows.append(parsed)
                max_len = max(max_len, len(parsed))
            else:
                rows.append([])
        if max_len == 0:
            continue
        for i in range(max_len):
            out[f"{col}_v{i}"] = [
                row[i] if i < len(row) else np.nan for row in rows
            ]
    return out


def build_str_mappings_and_replace(
    df: pd.DataFrame, str_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    为每个字符串列建立 值 -> 整数 映射，并替换该列；返回新 DataFrame 与 mappings。
    对形如 "['a','b']" 的单元格先转为 "a|b" 再映射。
    """
    mappings = {}
    out = df.copy()
    for col in str_cols:
        uniques = set()
        normal_values = []
        for v in out[col]:
            s_list = _parse_str_list_cell(v)
            if s_list is not None:
                normal_values.append(s_list)
                uniques.add(s_list)
            else:
                s = str(v).strip() if pd.notna(v) else ""
                normal_values.append(s)
                uniques.add(s)
        uniques = sorted(uniques, key=lambda x: (x == "", x))
        mapping = {val: i for i, val in enumerate(uniques)}
        mappings[col] = mapping
        out[col] = [mapping.get(nv, mapping.get("", 0)) for nv in normal_values]
    return out, mappings


def process(
    raw_path: str,
    output_csv_path: str,
    output_xlsx_path: str,
    mapping_path: str,
    date_column: str = "时间",
    out_date_name: str = "date",
) -> None:
    """
    主流程：读入原始 CSV，展开向量列，字符串列映射为整数，写出训练用 CSV 与映射文件。
    """
    df = _load_csv(raw_path)

    # 去除不参与预测的列
    to_drop = [c for c in EXCLUDE_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        print(f"已排除不参与预测的列: {to_drop}")

    if date_column in df.columns:
        df = df.rename(columns={date_column: out_date_name})
    date_col = out_date_name
    if date_col not in df.columns:
        raise ValueError(f"未找到日期列: {date_column} / {out_date_name}")

    # 只保留有日期且可用的行
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    vector_cols, str_cols, numeric_cols = classify_columns(df)
    str_cols = [c for c in str_cols if c != date_col]
    print(f"向量列（将展开）: {len(vector_cols)} 个")
    print(f"字符串列（将映射）: {len(str_cols)} 个")
    print(f"已是数值列: {len(numeric_cols)} 个")

    # 1) 展开向量列
    df = expand_vector_columns(df, vector_cols)

    # 2) 字符串列映射
    str_cols = [c for c in str_cols if c in df.columns]
    if str_cols:
        df, str_mappings = build_str_mappings_and_replace(df, str_cols)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(str_mappings, f, ensure_ascii=False, indent=2)
        print(f"已保存字符串映射: {mapping_path}")
    else:
        str_mappings = {}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    # 3) 确保除 date 外均为数值
    feature_cols = [c for c in df.columns if c != date_col]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    feature_cols = [c for c in df.columns if c != date_col]
    df = df[[date_col] + feature_cols]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx_path)) or ".", exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    df.to_excel(output_xlsx_path, index=False)
    # print(f"已保存训练数据: {output_csv_path}  形状: {df.shape}  特征数: {len(feature_cols)}")
    print(
        f"已保存训练数据:\n"
        f"  CSV : {output_csv_path}\n"
        f"  XLSX: {output_xlsx_path}\n"
        f"  形状: {df.shape}  特征数: {len(feature_cols)}"
    )

def main():
    parser = argparse.ArgumentParser(description="得闲谨制数据预处理：向量展开 + 字符串映射 -> 训练 CSV")
    parser.add_argument(
        "--raw",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_RAW_NAME),
        help="原始 CSV 路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_OUTPUT_NAME),
        help="输出训练 CSV 路径",
    )
    parser.add_argument(
        "--mapping",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_MAPPING_NAME),
        help="字符串映射 JSON 路径",
    )
    parser.add_argument("--date-col", default="时间", help="原始日期列名")
    parser.add_argument("--out-date", default="date", help="输出日期列名")
    parser.add_argument(
        "--xlsx",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_XLSX_NAME),
        help="输出训练 XLSX 路径",
    )
    args = parser.parse_args()
    process(
        raw_path=args.raw,
        output_csv_path=args.output,
        output_xlsx_path=args.xlsx,
        mapping_path=args.mapping,
        date_column=args.date_col,
        out_date_name=args.out_date,
    )


if __name__ == "__main__":
    main()
