import json
import math
import os
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import VideoMAEConfig, VideoMAEImageProcessor

from models.model import TICModel
from utils.common import dict2namespace


def load_model_and_processor(config_path, checkpoint_path, device=None):
    """加载面部抽动二分类模型和预处理器。"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    model_path = config.model.model_name
    model_config = VideoMAEConfig.from_pretrained(model_path)
    model = TICModel(
        model_config,
        num_classes=config.model.num_classes,
        use_lora=config.model.use_lora,
    ).to(device)
    processor = VideoMAEImageProcessor.from_pretrained(model_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    checkpoint_classifier_weight_shape = None
    for key in state_dict.keys():
        if "classifier.1.weight" in key:
            checkpoint_classifier_weight_shape = state_dict[key].shape
            break

    if checkpoint_classifier_weight_shape is not None:
        current_shape = model.classifier[1].weight.shape
        if checkpoint_classifier_weight_shape != current_shape:
            num_classes_from_checkpoint = checkpoint_classifier_weight_shape[0]
            hidden_size = model_config.hidden_size
            model.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, num_classes_from_checkpoint),
            ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return config, model, processor


def _to_float_frame(frame):
    """将 OpenCV 帧转换为模型输入所需的 float32 RGB 帧。"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0


def _infer_batch(model, processor, batch_windows, device, positive_class_index):
    """对一个 batch 的滑窗做推理，返回正类概率。"""
    inputs = processor(batch_windows, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(inputs["pixel_values"])
        probs = torch.softmax(logits, dim=1)[:, positive_class_index]

    return probs.detach().cpu().numpy().tolist()


def _smooth_scores(scores, smooth_window):
    """对时间序列做简单移动平均平滑。"""
    if len(scores) == 0:
        return []

    smooth_window = max(1, int(smooth_window))
    if smooth_window == 1 or len(scores) == 1:
        return list(scores)

    if smooth_window % 2 == 0:
        smooth_window += 1

    values = np.asarray(scores, dtype=np.float32)
    pad_left = smooth_window // 2
    pad_right = smooth_window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.tolist()


def _merge_intervals(candidates, merge_gap_sec):
    """合并时间上相邻或重叠的候选区间。"""
    if len(candidates) == 0:
        return []

    candidates = sorted(candidates, key=lambda x: x["start_sec"])
    merged = [candidates[0].copy()]
    merged[0]["score_sum"] = merged[0]["smoothed_score"]
    merged[0]["window_count"] = 1

    for item in candidates[1:]:
        last = merged[-1]
        if item["start_sec"] <= last["end_sec"] + merge_gap_sec:
            last["end_sec"] = max(last["end_sec"], item["end_sec"])
            last["start_frame"] = min(last["start_frame"], item["start_frame"])
            last["end_frame"] = max(last["end_frame"], item["end_frame"])
            last["score_sum"] += item["smoothed_score"]
            last["window_count"] += 1
            last["peak_score"] = max(last["peak_score"], item["smoothed_score"])
            last["raw_peak_score"] = max(last["raw_peak_score"], item["raw_score"])
        else:
            new_item = item.copy()
            new_item["score_sum"] = new_item["smoothed_score"]
            new_item["window_count"] = 1
            merged.append(new_item)

    for item in merged:
        item["duration_sec"] = item["end_sec"] - item["start_sec"]
        item["mean_score"] = item["score_sum"] / max(1, item["window_count"])
        item["mean_score"] = float(item["mean_score"])
        item["peak_score"] = float(item["peak_score"])
        item["raw_peak_score"] = float(item["raw_peak_score"])
        item["smoothed_score"] = float(item["smoothed_score"])
        item["raw_score"] = float(item["raw_score"])
        item["start_sec"] = float(item["start_sec"])
        item["end_sec"] = float(item["end_sec"])

    return merged


def _export_clip(video_path, start_sec, end_sec, output_path, fps=None):
    """把指定时间区间裁剪成单独视频。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = max(start_frame, int(math.ceil(end_sec * fps)) - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"无法读取裁剪起始帧: {video_path}")

    height, width = frame.shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = start_frame
    while ret and current_frame <= end_frame:
        writer.write(frame)
        ret, frame = cap.read()
        current_frame += 1

    writer.release()
    cap.release()
    return output_path


def detect_tic_intervals(
    video_path,
    model,
    processor,
    device,
    sample_frames=16,
    stride_frames=4,
    batch_size=8,
    positive_class_index=1,
    threshold=0.5,
    smooth_window=5,
    min_duration_sec=0.4,
    merge_gap_sec=0.3,
    pad_sec=0.0,
    save_clips_dir=None,
):
    """对单个长视频做抽动区间检测。"""
    if sample_frames <= 0:
        raise ValueError("sample_frames 必须大于 0")
    if stride_frames <= 0:
        raise ValueError("stride_frames 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    buffer = deque(maxlen=sample_frames)
    pending_windows = []
    pending_meta = []
    window_metas = []
    raw_scores = []
    total_frames = 0
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = _to_float_frame(frame)
        buffer.append(last_frame)

        current_frame_idx = total_frames
        total_frames += 1

        start_frame_idx = current_frame_idx - sample_frames + 1
        if start_frame_idx >= 0 and start_frame_idx % stride_frames == 0:
            pending_windows.append(list(buffer))
            pending_meta.append(
                {
                    "start_frame": start_frame_idx,
                    "end_frame": current_frame_idx,
                }
            )

            if len(pending_windows) >= batch_size:
                batch_scores = _infer_batch(
                    model=model,
                    processor=processor,
                    batch_windows=pending_windows,
                    device=device,
                    positive_class_index=positive_class_index,
                )
                raw_scores.extend(batch_scores)
                window_metas.extend(pending_meta)
                pending_windows = []
                pending_meta = []

    cap.release()

    if total_frames == 0:
        raise ValueError(f"视频中没有可读取的帧: {video_path}")

    if total_frames < sample_frames:
        if last_frame is None:
            raise ValueError(f"视频帧数不足且无法补帧: {video_path}")
        while len(buffer) < sample_frames:
            buffer.append(last_frame.copy())
        pending_windows.append(list(buffer))
        pending_meta.append(
            {
                "start_frame": 0,
                "end_frame": sample_frames - 1,
            }
        )
    else:
        final_start = max(0, total_frames - sample_frames)
        if len(window_metas) == 0 or window_metas[-1]["start_frame"] != final_start:
            pending_windows.append(list(buffer))
            pending_meta.append(
                {
                    "start_frame": final_start,
                    "end_frame": total_frames - 1,
                }
            )

    if len(pending_windows) > 0:
        batch_scores = _infer_batch(
            model=model,
            processor=processor,
            batch_windows=pending_windows,
            device=device,
            positive_class_index=positive_class_index,
        )
        raw_scores.extend(batch_scores)
        window_metas.extend(pending_meta)

    if len(raw_scores) != len(window_metas):
        raise RuntimeError("窗口预测数量与元信息数量不一致")

    smoothed_scores = _smooth_scores(raw_scores, smooth_window)
    candidates = []
    video_duration_sec = total_frames / fps

    for meta, raw_score, smoothed_score in zip(window_metas, raw_scores, smoothed_scores):
        if smoothed_score < threshold:
            continue

        start_sec = max(0.0, meta["start_frame"] / fps - pad_sec)
        end_sec = min(video_duration_sec, (meta["end_frame"] + 1) / fps + pad_sec)
        candidates.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": int(meta["start_frame"]),
                "end_frame": int(meta["end_frame"]),
                "raw_score": float(raw_score),
                "smoothed_score": float(smoothed_score),
                "peak_score": float(smoothed_score),
                "raw_peak_score": float(raw_score),
            }
        )

    merged = _merge_intervals(candidates, merge_gap_sec)
    intervals = []

    for idx, item in enumerate(merged):
        if item["duration_sec"] < min_duration_sec:
            continue

        interval = {
            "index": idx,
            "start_sec": round(item["start_sec"], 3),
            "end_sec": round(item["end_sec"], 3),
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
            "duration_sec": round(item["duration_sec"], 3),
            "mean_score": round(item["mean_score"], 4),
            "peak_score": round(item["peak_score"], 4),
            "raw_peak_score": round(item["raw_peak_score"], 4),
        }

        if save_clips_dir is not None:
            os.makedirs(save_clips_dir, exist_ok=True)
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            clip_name = (
                f"{video_stem}_tic_{idx + 1:03d}_"
                f"{int(round(interval['start_sec'] * 1000)):08d}_"
                f"{int(round(interval['end_sec'] * 1000)):08d}.mp4"
            )
            clip_path = os.path.join(save_clips_dir, clip_name)
            _export_clip(
                video_path=video_path,
                start_sec=item["start_sec"],
                end_sec=item["end_sec"],
                output_path=clip_path,
                fps=fps,
            )
            interval["clip_path"] = clip_path

        intervals.append(interval)

    result = {
        "video_path": video_path,
        "fps": round(float(fps), 3),
        "frame_count": int(total_frames),
        "sample_frames": int(sample_frames),
        "stride_frames": int(stride_frames),
        "batch_size": int(batch_size),
        "threshold": float(threshold),
        "smooth_window": int(smooth_window),
        "min_duration_sec": float(min_duration_sec),
        "merge_gap_sec": float(merge_gap_sec),
        "pad_sec": float(pad_sec),
        "intervals": intervals,
        "raw_window_count": int(len(raw_scores)),
    }
    return result


def save_result_json(result, output_path):
    """保存检测结果到 JSON。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return output_path
