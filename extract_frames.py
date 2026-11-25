import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, target_fps: int = 30):
    """
    Extract frames from a video at a specified frame rate (FPS).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file '{video_path}'.")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / orig_fps if orig_fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Original FPS: {orig_fps:.2f} | Total frames: {total_frames} | Duration: {duration_sec:.2f}s")
    print(f"Extracting at {target_fps} FPS to '{output_dir}'...")

    frame_interval = 1.0 / target_fps
    current_target_time = 0.0
    saved_count = 0

    while True:
        # Set position in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, current_target_time * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break

        saved_count += 1
        output_filename = output_dir / f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(output_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        current_target_time += frame_interval
        if current_target_time > duration_sec:
            break

    cap.release()
    print(f"Done! Successfully extracted {saved_count} frames.")
    return saved_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames at a specified FPS")
    parser.add_argument("--video", type=str, default="Tennis.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="frames", help="Path to output frames directory")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second (default: 30)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)
