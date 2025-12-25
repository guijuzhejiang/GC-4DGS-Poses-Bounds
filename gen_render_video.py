#!/usr/bin/env python3
"""
images_to_video.py

把指定目录里按顺序的图片 (例如: 00000.png, 00001.png, ...) 合成一个 MP4 视频。
生成的视频会保存到图片目录下。

用法:
    python images_to_video.py /path/to/images --fps 24 --output video.mp4
"""

import argparse
import re
import sys
from pathlib import Path

def numeric_sort_key(path: Path):
    # 从文件名里提取连续的数字作为排序依据（如果没有数字则用文件名本身）
    m = re.search(r'(\d+)', path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (float('inf'), path.name)

def find_images(directory: Path):
    # 支持常见图片后缀
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
    files = []
    for e in exts:
        files.extend(directory.glob(e))
    files = [f for f in files if f.is_file()]
    files.sort(key=numeric_sort_key)
    return files

def write_with_cv2(image_paths, out_path, fps):
    import cv2
    if not image_paths:
        raise ValueError("没有找到图片。")
    # 读取第一张确定尺寸
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"无法读取图片: {image_paths[0]}")
    # 如果含 alpha 通道，转换为 BGR
    if first.shape[-1] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {out_path}")

    # 写入第一帧
    writer.write(first)

    for p in image_paths[1:]:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"警告: 跳过无法读取的图片: {p}", file=sys.stderr)
            continue
        # RGBA -> BGR
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # 如果大小不同，缩放到第一张尺寸
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    print(f"已生成视频: {out_path}")

def write_with_moviepy(image_paths, out_path, fps):
    # 回退用 moviepy（需要安装 moviepy 和 imageio）
    from moviepy.editor import ImageSequenceClip
    if not image_paths:
        raise ValueError("没有找到图片。")
    paths = [str(p) for p in image_paths]
    clip = ImageSequenceClip(paths, fps=fps)
    clip.write_videofile(str(out_path), codec='libx264', audio=False)
    print(f"已生成视频: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="把图片序列合成视频 (输出到图片目录)")
    parser.add_argument("--dir", default='/home/zzg/workspace/pycharm/GC-4DGS/output/dynerf/Take_154814_ud/3_views/MVS/test/chkpnt_best/renders/', help="图片所在目录（例如 /path/to/images）")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率，默认 30")
    parser.add_argument("--output", type=str, default=None, help="输出文件名，例如 out.mp4（默认: images_dir/out.mp4）")
    args = parser.parse_args()

    img_dir = Path(args.dir).expanduser().resolve()
    if not img_dir.is_dir():
        print(f"错误: 指定路径不是目录: {img_dir}", file=sys.stderr)
        sys.exit(2)

    images = find_images(img_dir)
    if not images:
        print(f"错误: 目录中未找到图片: {img_dir}", file=sys.stderr)
        sys.exit(2)

    out_name = args.output if args.output else f"{img_dir.name}.mp4"
    out_path = img_dir / out_name

    # 优先使用 cv2（OpenCV），没有则尝试 moviepy
    try:
        import cv2  # type: ignore
        write_with_cv2(images, out_path, args.fps)
    except Exception as e_cv2:
        print(f"使用 OpenCV 合成失败或 OpenCV 未安装: {e_cv2}\n尝试使用 moviepy 回退...", file=sys.stderr)
        try:
            write_with_moviepy(images, out_path, args.fps)
        except Exception as e_mp:
            print("moviepy 回退也失败。请确认已安装 opencv-python 或 moviepy。", file=sys.stderr)
            print(f"OpenCV 错误: {e_cv2}", file=sys.stderr)
            print(f"moviepy 错误: {e_mp}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
