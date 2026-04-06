"""
Visualize MOT17 tracking results by drawing bounding boxes on frames.
Outputs annotated frames and an MP4 video.

Usage:
    python visualize_tracking.py --seq MOT17-04-SDP
    python visualize_tracking.py --seq MOT17-04-SDP --source ucmc  # for output/mot17/val results
"""
import argparse
import os
import csv
import cv2
import numpy as np
from collections import defaultdict


def get_color(track_id):
    """Generate a consistent, visually distinct color for a given track ID."""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


def load_tracking_results(txt_path):
    """Load MOT format results: frame, id, bb_left, bb_top, bb_w, bb_h, ..."""
    results = defaultdict(list)
    with open(txt_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            frame = int(row[0])
            track_id = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            results[frame].append((track_id, x, y, w, h))
    return results


def draw_tracks(img, tracks):
    """Draw bounding boxes and track IDs on image."""
    for track_id, x, y, w, h in tracks:
        color = get_color(track_id)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Draw ID label with background
        label = str(track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize MOT tracking results")
    parser.add_argument("--seq", type=str, default="MOT17-04-SDP",
                        help="Sequence name, e.g. MOT17-04-SDP")
    parser.add_argument("--source", type=str, default="train",
                        choices=["train", "ucmc"],
                        help="'train' for train/ folder results, 'ucmc' for output/mot17/val/")
    parser.add_argument("--mot_dir", type=str,
                        default="/usr/local/data/swen14/Workspace/MOT17/train",
                        help="Path to MOT17 train directory containing sequence folders")
    parser.add_argument("--save_frames", action="store_true",
                        help="Also save individual annotated frames")
    args = parser.parse_args()

    # Resolve paths
    img_dir = os.path.join(args.mot_dir, args.seq, "img1")
    if args.source == "train":
        results_path = os.path.join("train", f"{args.seq}.txt")
    else:
        results_path = os.path.join("output", "mot17", "val", f"{args.seq}.txt")

    out_dir = os.path.join("output", "vis", args.source, args.seq)
    os.makedirs(out_dir, exist_ok=True)

    if args.save_frames:
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

    print(f"Sequence:   {args.seq}")
    print(f"Images:     {img_dir}")
    print(f"Results:    {results_path}")
    print(f"Output:     {out_dir}")

    # Load results
    tracking = load_tracking_results(results_path)
    unique_ids = set()
    for tracks in tracking.values():
        for t in tracks:
            unique_ids.add(t[0])
    print(f"Loaded {sum(len(v) for v in tracking.values())} detections, "
          f"{len(unique_ids)} unique track IDs, "
          f"across {len(tracking)} frames")

    # Get sorted frame images
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    num_frames = len(img_files)
    print(f"Total frames: {num_frames}")

    # Read first frame to get dimensions
    sample = cv2.imread(os.path.join(img_dir, img_files[0]))
    h, w = sample.shape[:2]

    # Setup video writer
    video_path = os.path.join(out_dir, f"{args.seq}_{args.source}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # Process each frame
    for i, fname in enumerate(img_files):
        frame_num = int(fname.split(".")[0])  # e.g. "000001.jpg" -> 1
        img = cv2.imread(os.path.join(img_dir, fname))

        tracks = tracking.get(frame_num, [])
        img = draw_tracks(img, tracks)

        # Add frame info
        cv2.putText(img, f"Frame {frame_num}  |  {len(tracks)} tracks",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(img)

        if args.save_frames:
            cv2.imwrite(os.path.join(frames_dir, fname), img)

        if (i + 1) % 100 == 0 or i == num_frames - 1:
            print(f"  Processed {i + 1}/{num_frames} frames")

    writer.release()
    print(f"\nVideo saved to: {video_path}")


if __name__ == "__main__":
    main()
