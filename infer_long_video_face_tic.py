import argparse
import glob
import os

import torch

from utils.long_video_infer import (
    detect_tic_intervals,
    load_model_and_processor,
    save_result_json,
)


def get_args():
    parser = argparse.ArgumentParser(description="Long video face tic inference")
    parser.add_argument(
        "--config",
        default="configs/config_face_binary.yaml",
        type=str,
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        default="ckp/face_binary/best_model.pth",
        type=str,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="输入 mp4 文件或包含 mp4 的目录",
    )
    parser.add_argument(
        "--output_dir",
        default="./long_video_outputs",
        type=str,
        help="输出目录",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        type=str,
        help="单个视频时可直接指定输出 JSON 路径",
    )
    parser.add_argument(
        "--save_clips_dir",
        default=None,
        type=str,
        help="可选：裁剪抽动区间并保存为单独 mp4 的目录",
    )
    parser.add_argument("--sample_frames", type=int, default=16, help="滑窗帧数")
    parser.add_argument("--stride_frames", type=int, default=4, help="滑窗步长")
    parser.add_argument("--batch_size", type=int, default=8, help="推理 batch 大小")
    parser.add_argument("--threshold", type=float, default=0.5, help="抽动区间阈值")
    parser.add_argument("--smooth_window", type=int, default=5, help="平滑窗口长度")
    parser.add_argument("--min_duration_sec", type=float, default=0.4, help="最短区间时长")
    parser.add_argument("--merge_gap_sec", type=float, default=0.3, help="区间合并间隔")
    parser.add_argument("--pad_sec", type=float, default=0.0, help="区间前后补偿时长")
    parser.add_argument(
        "--positive_label",
        default="face-tic",
        type=str,
        help="正类标签名",
    )
    return parser.parse_args()


def _collect_videos(input_path):
    if os.path.isdir(input_path):
        video_list = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
        return video_list
    if os.path.isfile(input_path):
        return [input_path]
    raise ValueError(f"输入路径不存在: {input_path}")


def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config, model, processor = load_model_and_processor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if args.positive_label not in config.data.labels:
        raise ValueError(f"正类标签 {args.positive_label} 不在配置标签中: {config.data.labels}")
    positive_class_index = config.data.labels.index(args.positive_label)

    input_videos = _collect_videos(args.input_path)
    if len(input_videos) == 0:
        raise ValueError(f"没有找到可处理的 mp4 文件: {args.input_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_clips_dir is not None:
        os.makedirs(args.save_clips_dir, exist_ok=True)

    for video_path in input_videos:
        print(f"Processing video: {video_path}")
        result = detect_tic_intervals(
            video_path=video_path,
            model=model,
            processor=processor,
            device=device,
            sample_frames=args.sample_frames,
            stride_frames=args.stride_frames,
            batch_size=args.batch_size,
            positive_class_index=positive_class_index,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            min_duration_sec=args.min_duration_sec,
            merge_gap_sec=args.merge_gap_sec,
            pad_sec=args.pad_sec,
            save_clips_dir=args.save_clips_dir,
        )

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        if args.output_json is not None and len(input_videos) == 1:
            output_json = args.output_json
        else:
            output_json = os.path.join(args.output_dir, f"{video_stem}.json")

        save_result_json(result, output_json)
        print(f"Saved result: {output_json}")
        print(f"Detected intervals: {len(result['intervals'])}")


if __name__ == "__main__":
    main()
