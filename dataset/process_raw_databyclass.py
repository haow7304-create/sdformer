# -*- coding: utf-8 -*-
"""
得闲谨制时序数据预处理脚本（按字段分组分别输出）
- 将向量列展开为多列参与预测
- 将字符串列通过字典映射为整数值后参与训练
- 按预设字段分为 3 份数据分别输出 CSV、XLSX、映射文件
"""

import os
import json
import ast
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

# 默认路径：脚本所在目录为数据集目录
DEFAULT_RAW_NAME = "得闲谨制_最小指标集_原始.csv"
ENCODINGS = ["utf-8", "gbk", "utf-8-sig"]

# 不参与预测的列：若存在则先删除
EXCLUDE_COLUMNS = [
    "全量用户id（向量）",
    "热搜关键词（向量）",
    "最强关联信息",
    "平台列表（向量）",
]

# ========== 三个文件对应字段 ==========
FILE1_COLUMNS = [
    "国内用户总数（万人）",
    "省级行政区分布（向量）",
    "国际用户总数（万人）",
    "国家或地区分布（向量）",
    "年龄段分布（向量）",
    "男性数量",
    "女性数量",
    "兴趣分布（向量）",
    "职业分布（向量）",
    "受教育程度分布（向量）",
    "收入分布（向量）",
    "意识形态分布（向量）",
    "意见领袖的平均粉丝数",
    "意见领袖的发帖次数总和",
    "意见领袖的转发次数总和",
    "意见领袖的评论次数总和",
    "意见领袖的点赞次数总和",
    "普通网民的平均粉丝数",
    "普通网民的发帖次数总和",
    "普通网民的转发次数总和",
    "普通网民的评论次数总和",
    "普通网民的点赞次数总和",
    "当事人的平均粉丝数",
    "当事人的发帖次数总和",
    "当事人的转发次数总和",
    "当事人的评论次数总和",
    "当事人的点赞次数总和",
    "官方媒体的平均粉丝数",
    "官方媒体的发帖次数总和",
    "官方媒体的转发次数总和",
    "官方媒体的评论次数总和",
    "官方媒体的点赞次数总和",
    "网络媒体的平均粉丝数",
    "网络媒体的发帖次数总和",
    "网络媒体的转发次数总和",
    "网络媒体的评论次数总和",
    "网络媒体的点赞次数总和",
    "全量用户各自的转评赞总和（向量）",
    "全量用户各自的粉丝数（向量）",
    "全量用户各自发布内容的正面情感比例（向量）",
    "全量用户各自发布内容的负面情感比例（向量）",
    "潜在受众数量",
    "知情者数量",
    "传播者数量",
    "停滞者数量",
]

FILE2_COLUMNS = [
    "事件搜索量",
    "事件浏览量",
    "否定观点数量",
    "肯定观点数量",
    "覆盖平台个数",
    "文章类内容数量",
    "图片类内容数量",
    "视频类内容数量",
    "正面帖子数量",
    "中性帖子数量",
    "负面帖子数量",
    "各平台用户分布（向量）",
    "敏感话题数量",
    "事件涉及的话题数量",
    "敏感用户数量",
    "AI生成内容检测率",
    "AI 合成内容传播指数",
]

FILE3_COLUMNS = [
    "不同平台的帖子数量（向量）",
    "跨平台扩展潜力",
    "各个平台的浏览次数（向量）",
    "各个平台的转评赞数量（向量）",
    "跨平台传播速度",
    "跨平台情绪协调性",
    "各个平台的正面情感比例（向量）",
    "各个平台的负面情感比例（向量）",
    "各个平台的中性情感比例（向量）",
    "各个平台首次响应时间（向量）",
    "各个平台的响应用户数量（向量）",
]

FILE_CONFIGS = [
    {
        "name": "file1",
        "csv": "train_data_file1.csv",
        "xlsx": "train_data_file1.xlsx",
        "mapping": "str_mappings_file1.json",
        "columns": FILE1_COLUMNS,
    },
    {
        "name": "file2",
        "csv": "train_data_file2.csv",
        "xlsx": "train_data_file2.xlsx",
        "mapping": "str_mappings_file2.json",
        "columns": FILE2_COLUMNS,
    },
    {
        "name": "file3",
        "csv": "train_data_file3.csv",
        "xlsx": "train_data_file3.xlsx",
        "mapping": "str_mappings_file3.json",
        "columns": FILE3_COLUMNS,
    },
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
    """若单元格为字符串列表（如 "['a','b']"），返回拼接后的字符串用于映射；否则返回 None。"""
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


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    将列分为：向量列、字符串列（需映射）、已是数值列。
    向量列：首个非空值可解析为数字列表的列。
    字符串列：object/string 且非向量，或无法强转为数值的列。
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
            try:
                pd.to_numeric(df[col], errors="raise")
                numeric_cols.append(col)
            except Exception:
                str_cols.append(col)
            continue

        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except Exception:
            str_cols.append(col)

    return vector_cols, str_cols, numeric_cols


def expand_vector_columns(df: pd.DataFrame, vector_cols: List[str]) -> pd.DataFrame:
    """将向量列展开为多列，列名为 原列名_v0, 原列名_v1, ..."""
    out = df.drop(columns=vector_cols, errors="ignore").copy()
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
            out[f"{col}_v{i}"] = [row[i] if i < len(row) else np.nan for row in rows]

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
        normalized_values = []

        for v in out[col]:
            s_list = _parse_str_list_cell(v)
            if s_list is not None:
                normalized_values.append(s_list)
                uniques.add(s_list)
            else:
                s = str(v).strip() if pd.notna(v) else ""
                normalized_values.append(s)
                uniques.add(s)

        uniques = sorted(uniques, key=lambda x: (x == "", x))
        mapping = {val: i for i, val in enumerate(uniques)}
        mappings[col] = mapping
        out[col] = [mapping.get(v, mapping.get("", 0)) for v in normalized_values]

    return out, mappings


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def select_existing_columns(df: pd.DataFrame, wanted_cols: List[str], date_col: str) -> Tuple[List[str], List[str]]:
    existing = [c for c in wanted_cols if c in df.columns]
    missing = [c for c in wanted_cols if c not in df.columns]
    ordered = [date_col] + existing if date_col in df.columns else existing
    return ordered, missing


def process_subset(
    df_raw: pd.DataFrame,
    selected_columns: List[str],
    output_csv_path: str,
    output_xlsx_path: str,
    mapping_path: str,
    subset_name: str,
    date_col: str = "date",
) -> None:
    """
    处理单个字段分组：
    1) 仅保留日期列 + 该组字段
    2) 展开向量列
    3) 字符串列映射
    4) 输出 CSV、XLSX、映射 JSON
    """
    cols_to_use, missing_cols = select_existing_columns(df_raw, selected_columns, date_col)
    if date_col not in cols_to_use:
        raise ValueError(f"[{subset_name}] 未找到日期列: {date_col}")

    if len(cols_to_use) == 1:
        raise ValueError(f"[{subset_name}] 除日期列外，没有任何可用字段。")

    df = df_raw[cols_to_use].copy()
    df = df.dropna(subset=[date_col]).reset_index(drop=True)

    vector_cols, str_cols, numeric_cols = classify_columns(df)
    str_cols = [c for c in str_cols if c != date_col]

    print(f"\n===== 处理 {subset_name} =====")
    if missing_cols:
        print(f"缺失字段（原始数据中不存在，共 {len(missing_cols)} 个）: {missing_cols}")
    print(f"向量列（将展开）: {len(vector_cols)} 个")
    print(f"字符串列（将映射）: {len(str_cols)} 个")
    print(f"已是数值列: {len(numeric_cols)} 个")

    # 1) 展开向量列
    df = expand_vector_columns(df, vector_cols)

    # 2) 字符串列映射
    str_cols = [c for c in str_cols if c in df.columns]
    if str_cols:
        df, str_mappings = build_str_mappings_and_replace(df, str_cols)
    else:
        str_mappings = {}

    ensure_parent_dir(mapping_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(str_mappings, f, ensure_ascii=False, indent=2)
    print(f"已保存字符串映射: {mapping_path}")

    # 3) 确保除 date 外尽量转成数值
    feature_cols = [c for c in df.columns if c != date_col]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除全空列
    df = df.dropna(axis=1, how="all")
    feature_cols = [c for c in df.columns if c != date_col]
    df = df[[date_col] + feature_cols]

    # 输出
    ensure_parent_dir(output_csv_path)
    ensure_parent_dir(output_xlsx_path)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    df.to_excel(output_xlsx_path, index=False)

    print(
        f"已保存训练数据:\n"
        f"  CSV : {output_csv_path}\n"
        f"  XLSX: {output_xlsx_path}\n"
        f"  形状: {df.shape}  特征数: {len(feature_cols)}"
    )


def process_all(
    raw_path: str,
    output_dir: str,
    date_column: str = "时间",
    out_date_name: str = "date",
) -> None:
    """
    主流程：
    - 读入原始 CSV
    - 删除排除列
    - 统一日期列名
    - 按 3 组字段分别预处理并输出
    """
    df = _load_csv(raw_path)

    # 去除不参与预测的列
    to_drop = [c for c in EXCLUDE_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        print(f"已排除不参与预测的列: {to_drop}")

    # 重命名日期列
    if date_column in df.columns:
        df = df.rename(columns={date_column: out_date_name})

    if out_date_name not in df.columns:
        raise ValueError(f"未找到日期列: {date_column} / {out_date_name}")

    # 只保留有日期的行
    df = df.dropna(subset=[out_date_name]).reset_index(drop=True)

    # 逐个输出
    for cfg in FILE_CONFIGS:
        process_subset(
            df_raw=df,
            selected_columns=cfg["columns"],
            output_csv_path=os.path.join(output_dir, cfg["csv"]),
            output_xlsx_path=os.path.join(output_dir, cfg["xlsx"]),
            mapping_path=os.path.join(output_dir, cfg["mapping"]),
            subset_name=cfg["name"],
            date_col=out_date_name,
        )


def main():
    parser = argparse.ArgumentParser(
        description="得闲谨制数据预处理：按指定字段分组分别输出训练 CSV / XLSX"
    )
    parser.add_argument(
        "--raw",
        default=os.path.join(os.path.dirname(__file__), DEFAULT_RAW_NAME),
        help="原始 CSV 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(__file__),
        help="输出目录",
    )
    parser.add_argument(
        "--date-col",
        default="时间",
        help="原始日期列名",
    )
    parser.add_argument(
        "--out-date",
        default="date",
        help="输出日期列名",
    )
    args = parser.parse_args()

    process_all(
        raw_path=args.raw,
        output_dir=args.output_dir,
        date_column=args.date_col,
        out_date_name=args.out_date,
    )


if __name__ == "__main__":
    main()
