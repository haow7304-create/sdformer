import argparse
import json
import os
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch


try:
    import reformer_pytorch  # type: ignore
except Exception:
    import types

    reformer_pytorch = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise NotImplementedError("LSHSelfAttention shim is not intended for use")

    reformer_pytorch.LSHSelfAttention = LSHSelfAttention
    sys.modules["reformer_pytorch"] = reformer_pytorch


PROJECT_ROOT = Path(__file__).resolve().parent
TIME_SERIES_ROOT = PROJECT_ROOT.parent / "time_series"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TIME_SERIES_ROOT) not in sys.path:
    sys.path.append(str(TIME_SERIES_ROOT))

from models.SDformer import Model
from index_cal_1 import transform_res1_to_json
from index_cal_2 import transform_res2_to_json
from index_cal_3 import transform_res3_to_json


Config = namedtuple(
    "Config",
    [
        "task_name",
        "seq_len",
        "pred_len",
        "label_len",
        "enc_in",
        "dec_in",
        "c_out",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "factor",
        "embed",
        "freq",
        "dropout",
        "activation",
        "output_attention",
        "top_k",
        "window_size",
        "p",
    ],
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 96
PRED_LEN = 48
LABEL_LEN = 48

MODEL_ARCH = {
    "task_name": "long_term_forecast",
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 4,
    "d_layers": 1,
    "d_ff": 128,
    "factor": 1,
    "embed": "timeF",
    "freq": "h",
    "dropout": 0.1,
    "activation": "gelu",
    "output_attention": False,
    "top_k": 5,
    "window_size": 8,
    "p": 2,
}

EVENT_SPECS = {
    "diplomacy": {
        "display_name": "中日外交",
        "aliases": ["中日外交", "中日", "日中", "外交", "日本", "中国"],
        "dataset_dir": PROJECT_ROOT / "dataset" / "diplomacy",
        "processed_csv": PROJECT_ROOT / "dataset" / "diplomacy" / "中日外交_最小指标集_processed.csv",
        "train_files": {
            "file1": "train_data_file1.csv",
            "file2": "train_data_file2.csv",
            "file3": "train_data_file3.csv",
        },
        "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "diplomacy",
    },
    "war": {
        "display_name": "美伊战争",
        "aliases": ["美伊战争", "美伊", "伊朗", "伊拉克", "战争"],
        "dataset_dir": PROJECT_ROOT / "dataset" / "war",
        "processed_csv": PROJECT_ROOT / "dataset" / "war" / "美伊战争_最小指标集_processed.csv",
        "train_files": {
            "file1": "train_data_file1.csv",
            "file2": "train_data_file2.csv",
            "file3": "train_data_file3.csv",
        },
        "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "war",
    },
}

EVENT_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int


MODEL_SPECS = {
    "file1": ModelSpec("file1", MODEL_ARCH["d_model"], MODEL_ARCH["n_heads"], MODEL_ARCH["e_layers"], MODEL_ARCH["d_ff"]),
    "file2": ModelSpec("file2", MODEL_ARCH["d_model"], MODEL_ARCH["n_heads"], MODEL_ARCH["e_layers"], MODEL_ARCH["d_ff"]),
    "file3": ModelSpec("file3", MODEL_ARCH["d_model"], MODEL_ARCH["n_heads"], MODEL_ARCH["e_layers"], MODEL_ARCH["d_ff"]),
}


def _load_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    if isinstance(checkpoint, Mapping):
        return dict(checkpoint)
    raise ValueError("Unsupported checkpoint format")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict or not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def build_config(feature_count: int, spec: ModelSpec) -> Config:
    return Config(
        task_name=MODEL_ARCH["task_name"],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        label_len=LABEL_LEN,
        enc_in=feature_count,
        dec_in=feature_count,
        c_out=feature_count,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        e_layers=spec.e_layers,
        d_layers=MODEL_ARCH["d_layers"],
        d_ff=spec.d_ff,
        factor=MODEL_ARCH["factor"],
        embed=MODEL_ARCH["embed"],
        freq=MODEL_ARCH["freq"],
        dropout=MODEL_ARCH["dropout"],
        activation=MODEL_ARCH["activation"],
        output_attention=MODEL_ARCH["output_attention"],
        top_k=MODEL_ARCH["top_k"],
        window_size=MODEL_ARCH["window_size"],
        p=MODEL_ARCH["p"],
    )


def load_model_file(model_path: Path, config: Config):
    model = Model(config).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = _strip_module_prefix(_load_state_dict(checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _normalize_time_series(df: pd.DataFrame, time_col: str = "date") -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out.set_index(time_col, inplace=True)
    out.sort_index(inplace=True)
    return out


def _load_event_data(event_key: str):
    spec = EVENT_SPECS[event_key]
    data_frames: Dict[str, pd.DataFrame] = {}

    for file_key, file_name in spec["train_files"].items():
        file_path = spec["dataset_dir"] / file_name
        df = pd.read_csv(file_path, encoding="utf-8")
        data_frames[file_key] = _normalize_time_series(df, "date")

    processed_df = pd.read_csv(spec["processed_csv"], encoding="utf-8")
    processed_time_col = "时间" if "时间" in processed_df.columns else "date"
    processed_times = (
        pd.to_datetime(processed_df[processed_time_col], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .tolist()
    )

    models: Dict[str, torch.nn.Module] = {}
    for file_key, df in data_frames.items():
        file_spec = MODEL_SPECS[file_key]
        checkpoint_name = (
            f"{file_key}_new_long_term_forecast_custom_96_96_SDformer_custom_ftM_sl96_ll48_pl48_"
            f"dm128_nh8_el4_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0"
        )
        checkpoint_path = spec["checkpoint_dir"] / checkpoint_name / "checkpoint.pth"
        config = build_config(df.shape[1], file_spec)
        models[file_key] = load_model_file(checkpoint_path, config)

    EVENT_CACHE[event_key] = {
        "spec": spec,
        "data_frames": data_frames,
        "processed_times": processed_times,
        "models": models,
    }
    return EVENT_CACHE[event_key]


def get_event_context(event_key: str):
    if event_key not in EVENT_CACHE:
        return _load_event_data(event_key)
    return EVENT_CACHE[event_key]


def _flatten_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.replace("，", ",").replace("；", ",").replace(";", ",")
        return [item.strip() for item in normalized.split(",") if item.strip()]
    if isinstance(value, Mapping):
        items: List[str] = []
        for item in value.values():
            items.extend(_flatten_values(item))
        return items
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            items.extend(_flatten_values(item))
        return items
    return [str(value).strip()]


def _collect_request_keywords(req_data: Mapping[str, Any]) -> List[str]:
    collected: List[str] = []
    for key in ("keywords", "keyword", "event_keywords", "event_keyword", "event", "event_name", "topic", "query"):
        if key in req_data:
            collected.extend(_flatten_values(req_data.get(key)))
    return [item for item in dict.fromkeys(token for token in collected if token)]


def _match_event_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    lowered = text.lower()
    scores: Dict[str, int] = {}
    for event_key, spec in EVENT_SPECS.items():
        score = 0
        for alias in spec["aliases"]:
            if alias and alias.lower() in lowered:
                score += 1
        scores[event_key] = score

    best_event = max(scores, key=scores.get)
    if scores[best_event] == 0:
        return None
    top_score = scores[best_event]
    tied_events = [event_key for event_key, score in scores.items() if score == top_score]
    if len(tied_events) > 1:
        for preferred in ("diplomacy", "war"):
            if preferred in tied_events:
                return preferred
        return None
    return best_event


def infer_event_key(req_data: Mapping[str, Any]) -> Optional[str]:
    explicit_event = None
    for key in ("event_key", "event_type", "dataset", "scene"):
        value = req_data.get(key)
        if isinstance(value, str) and value.strip():
            explicit_event = value.strip().lower()
            break

    if explicit_event in EVENT_SPECS:
        return explicit_event

    keyword_text = " ".join(_collect_request_keywords(req_data))
    matched_event = _match_event_from_text(keyword_text)
    if matched_event:
        return matched_event

    if explicit_event:
        return _match_event_from_text(explicit_event)
    return None


def parse_time_point(time_point: str):
    try:
        return pd.to_datetime(time_point)
    except Exception:
        return None


def preprocess_data_fast(df: pd.DataFrame, time_point: str):
    try:
        target_time = pd.to_datetime(time_point)
        start_time = target_time - timedelta(hours=SEQ_LEN - 1)
        df_slice = df.loc[start_time:target_time]

        if len(df_slice) < SEQ_LEN:
            df_slice = df.loc[:target_time].tail(SEQ_LEN)
        if len(df_slice) < SEQ_LEN:
            return None

        data_np = df_slice.ffill().bfill().values
        return torch.tensor(data_np, dtype=torch.float32).to(DEVICE)
    except Exception as exc:
        print(f"数据处理错误: {exc}")
        return None


def model_predict_safe(model, input_tensor):
    if input_tensor is None:
        return None
    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0)
        outputs = model(input_tensor, None, None, None)
        return outputs.cpu().numpy().squeeze(0)


def get_history_raw_slice(df: pd.DataFrame, time_point: str, left_cnt: int):
    target_time = pd.to_datetime(time_point)
    if left_cnt is None:
        left_cnt = 1
    left_cnt = int(left_cnt)
    if left_cnt <= 0:
        return pd.DataFrame(columns=df.columns)
    start_time = target_time - timedelta(hours=left_cnt - 1)
    return df.loc[start_time:target_time].copy()


def align_history_data(event_context: Mapping[str, Any], time_point: str, left_cnt: int):
    data_frames = event_context["data_frames"]
    hist1 = get_history_raw_slice(data_frames["file1"], time_point, left_cnt)
    hist2 = get_history_raw_slice(data_frames["file2"], time_point, left_cnt)
    hist3 = get_history_raw_slice(data_frames["file3"], time_point, left_cnt)

    common_index = hist1.index.intersection(hist2.index).intersection(hist3.index)
    hist1 = hist1.loc[common_index].sort_index()
    hist2 = hist2.loc[common_index].sort_index()
    hist3 = hist3.loc[common_index].sort_index()
    return hist1, hist2, hist3


def df_to_filled_numpy(df: pd.DataFrame):
    if df is None or df.empty:
        return np.empty((0, 0))
    return df.ffill().bfill().values


def safe_time_str(dt):
    if isinstance(dt, pd.Timestamp):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def normalize_transform_output(data, default_start_time=None):
    if data is None:
        return []
    if isinstance(data, list):
        result = []
        base_time = pd.to_datetime(default_start_time) if default_start_time else None
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                new_item = dict(item)
            else:
                new_item = {"value": item}
            if "timestep" not in new_item and base_time is not None:
                new_item["timestep"] = (base_time + pd.Timedelta(hours=idx)).strftime("%Y-%m-%d %H:%M:%S")
            result.append(new_item)
        return result
    if isinstance(data, dict):
        for key in ("data", "result", "results", "items", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return normalize_transform_output(value, default_start_time=default_start_time)
        single_item = dict(data)
        if "timestep" not in single_item and default_start_time is not None:
            single_item["timestep"] = str(default_start_time)
        return [single_item]
    return [{"value": data, "timestep": str(default_start_time) if default_start_time else None}]


def merge_three_analytics_by_timestep(file1_list, file2_list, file3_list, start_time=None):
    file1_list = normalize_transform_output(file1_list, default_start_time=start_time)
    file2_list = normalize_transform_output(file2_list, default_start_time=start_time)
    file3_list = normalize_transform_output(file3_list, default_start_time=start_time)

    total_len = min(len(file1_list), len(file2_list), len(file3_list))
    merged_list = []
    base_time = pd.to_datetime(start_time) if start_time else None

    for idx in range(total_len):
        item1 = file1_list[idx] if isinstance(file1_list[idx], dict) else {}
        item2 = file2_list[idx] if isinstance(file2_list[idx], dict) else {}
        item3 = file3_list[idx] if isinstance(file3_list[idx], dict) else {}
        timestep = item1.get("timestep") or item2.get("timestep") or item3.get("timestep")
        if timestep is None and base_time is not None:
            timestep = (base_time + pd.Timedelta(hours=idx)).strftime("%Y-%m-%d %H:%M:%S")

        merged_item = {}
        for item in (item1, item2, item3):
            for key, value in item.items():
                if key != "timestep":
                    merged_item[key] = value
        merged_item["timestep"] = timestep
        merged_list.append(merged_item)

    return merged_list


def _build_keyword_reference_csv(event_context: Mapping[str, Any], keyword_text: str, event_name: str) -> Path:
    processed_times = event_context["processed_times"]
    if not processed_times:
        raise ValueError("No timestamps available for keyword reference CSV")

    timestamps = list(processed_times)
    last_time = pd.to_datetime(timestamps[-1])
    timestamps.extend(
        (last_time + pd.Timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M:%S")
        for offset in range(1, PRED_LEN + 1)
    )
    reference_df = pd.DataFrame(
        {
            "时间": timestamps,
            "热搜关键词（向量）": [keyword_text] * len(timestamps),
            "最强关联信息": [event_name] * len(timestamps),
        }
    )
    temp_path = Path(tempfile.gettempdir()) / f"sdformer_keywords_{event_name}_{os.getpid()}_{torch.randint(0, 1_000_000, (1,)).item()}.csv"
    reference_df.to_csv(temp_path, index=False, encoding="utf-8")
    return temp_path


def calc_history_analytics(event_context: Mapping[str, Any], time_point: str, left_cnt: int, reference_csv: Path):
    hist1_df, hist2_df, hist3_df = align_history_data(event_context, time_point, left_cnt)
    if hist1_df.empty or hist2_df.empty or hist3_df.empty:
        return {
            "history_start_time": None,
            "history_end_time": time_point,
            "history_points": 0,
            "file1_analytics": [],
            "file2_analytics": [],
            "file3_analytics": [],
        }

    hist_res1 = df_to_filled_numpy(hist1_df)
    hist_res2 = df_to_filled_numpy(hist2_df)
    hist_res3 = df_to_filled_numpy(hist3_df)
    history_start_time = safe_time_str(hist1_df.index[0])

    downstream_hist_file1 = transform_res1_to_json(hist_res1, history_start_time)
    downstream_hist_file2 = transform_res2_to_json(hist_res1, hist_res2, history_start_time, str(reference_csv))
    downstream_hist_file3 = transform_res3_to_json(hist_res3, history_start_time)
    return {
        "history_start_time": history_start_time,
        "history_end_time": time_point,
        "history_points": len(hist1_df),
        "file1_analytics": downstream_hist_file1,
        "file2_analytics": downstream_hist_file2,
        "file3_analytics": downstream_hist_file3,
    }


def calc_future_analytics(event_context: Mapping[str, Any], time_point: str, reference_csv: Path):
    data_frames = event_context["data_frames"]
    models = event_context["models"]

    feat1 = preprocess_data_fast(data_frames["file1"], time_point)
    feat2 = preprocess_data_fast(data_frames["file2"], time_point)
    feat3 = preprocess_data_fast(data_frames["file3"], time_point)

    res1 = model_predict_safe(models["file1"], feat1)
    res2 = model_predict_safe(models["file2"], feat2)
    res3 = model_predict_safe(models["file3"], feat3)
    if res1 is None or res2 is None or res3 is None:
        return None

    future_start_time = (pd.to_datetime(time_point) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    downstream_file1 = transform_res1_to_json(res1, future_start_time)
    downstream_file2 = transform_res2_to_json(res1, res2, future_start_time, str(reference_csv))
    downstream_file3 = transform_res3_to_json(res3, future_start_time)
    return {
        "future_start_time": future_start_time,
        "future_points": PRED_LEN,
        "file1_analytics": downstream_file1,
        "file2_analytics": downstream_file2,
        "file3_analytics": downstream_file3,
    }


def build_processed_data(history_data, future_data):
    history_merged = merge_three_analytics_by_timestep(
        history_data.get("file1_analytics", []),
        history_data.get("file2_analytics", []),
        history_data.get("file3_analytics", []),
        start_time=history_data.get("history_start_time"),
    )
    future_merged = merge_three_analytics_by_timestep(
        future_data.get("file1_analytics", []),
        future_data.get("file2_analytics", []),
        future_data.get("file3_analytics", []),
        start_time=future_data.get("future_start_time"),
    )
    return history_merged + future_merged


def handle_predict_request(req_data: Mapping[str, Any]):
    keywords_raw = req_data.get("keywords") or req_data.get("keyword")
    left_cnt = req_data.get("left_cnt", 96)
    time_point = req_data.get("time") or req_data.get("time_point")

    if not time_point:
        return {"error": "Missing time"}, 400
    if parse_time_point(time_point) is None:
        return {"error": "Invalid time format"}, 400
    try:
        left_cnt = int(left_cnt)
    except Exception:
        return {"error": "left_cnt must be an integer"}, 400
    if left_cnt < 0:
        return {"error": "left_cnt must be >= 0"}, 400

    event_key = infer_event_key(req_data)
    if event_key is None:
        return {
            "error": "Unable to infer event from keywords",
            "supported_events": {key: spec["display_name"] for key, spec in EVENT_SPECS.items()},
        }, 400

    event_context = get_event_context(event_key)
    spec = event_context["spec"]

    keyword_tokens = _collect_request_keywords(req_data)
    if isinstance(keywords_raw, str):
        normalized_keywords = keywords_raw.replace("，", ",").replace("；", ",").replace(";", ",")
    elif isinstance(keywords_raw, (list, tuple, set)):
        normalized_keywords = ",".join(_flatten_values(keywords_raw))
    else:
        normalized_keywords = ",".join(keyword_tokens) if keyword_tokens else spec["display_name"]

    reference_csv = _build_keyword_reference_csv(event_context, normalized_keywords, spec["display_name"])
    try:
        history_data = calc_history_analytics(event_context, time_point, left_cnt, reference_csv)
        future_data = calc_future_analytics(event_context, time_point, reference_csv)
        if future_data is None:
            return {"error": "Data insufficient for the given time point"}, 400

        processed_data = build_processed_data(history_data, future_data)
        response_data = {
            "status": "success",
            "model_family": "SDformer",
            "event_key": event_key,
            "event_name": spec["display_name"],
            "time_point": time_point,
            "keywords": keywords_raw if keywords_raw is not None else normalized_keywords,
            "left_cnt": left_cnt,
            "right_cnt": PRED_LEN,
            "processed_data": processed_data,
        }

        save_dir = PROJECT_ROOT / "prediction_results"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_time = time_point.replace(" ", "_").replace(":", "-")
        file_path = save_dir / f"result_{event_key}_{safe_time}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)
            print(f"✅ 结果已保存至: {file_path}")
        except Exception as exc:
            print(f"❌ 保存文件失败: {exc}")
        return response_data, 200
    finally:
        try:
            reference_csv.unlink(missing_ok=True)
        except Exception:
            pass


def handle_health_request():
    return {"status": "ok", "model_family": "SDformer"}, 200


class BackendHandler(BaseHTTPRequestHandler):
    server_version = "SDformerEventBackend/1.0"

    def _send_json(self, payload: Mapping[str, Any], status: int = 200):
        body = json.dumps(payload, ensure_ascii=False, indent=4).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            payload, status = handle_health_request()
            self._send_json(payload, status)
            return
        self._send_json({"error": "Not Found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") != "/predict":
            self._send_json({"error": "Not Found"}, 404)
            return
        req_data = self._read_json()
        if req_data is None:
            self._send_json({"error": "Invalid JSON body"}, 400)
            return
        payload, status = handle_predict_request(req_data)
        self._send_json(payload, status)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Keyword-aware SDformer backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BackendHandler)
    print(f"Serving SDformer backend on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
