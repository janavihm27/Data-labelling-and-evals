import os
import sys
import json
import csv
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.models.detection as d
from torchvision.transforms import functional as F

def get_shot_role(frame_idx: int):
    """
    Returns (active_player, passive_player) for a given frame index:
    'Major' or 'Mustermann'
    Or None if outplay frame.
    """
    if 1 <= frame_idx <= 85:
        return 'Major', 'Mustermann'        # Shot 1: Major serving
    elif 86 <= frame_idx <= 140:
        return 'Mustermann', 'Major'        # Shot 2: Mustermann return
    elif 141 <= frame_idx <= 195:
        return 'Major', 'Mustermann'        # Shot 3: Major groundstroke
    elif 196 <= frame_idx <= 250:
        return 'Mustermann', 'Major'        # Shot 4: Mustermann deep forehand
    elif 251 <= frame_idx <= 295:
        return 'Major', 'Mustermann'        # Shot 5: Major backhand
    elif 296 <= frame_idx <= 354:
        return 'Mustermann', 'Major'        # Shot 6: Mustermann forehand
    elif 355 <= frame_idx <= 487:
        return None, None                   # OUTPLAY FRAMES (Close-up / slow-motion replay)
    elif 488 <= frame_idx <= 545:
        return 'Mustermann', 'Major'        # Shot 7: Mustermann forehand
    elif 546 <= frame_idx <= 595:
        return 'Major', 'Mustermann'        # Shot 8: Major backhand
    elif 596 <= frame_idx <= 658:
        return 'Mustermann', 'Major'        # Shot 9: Mustermann forehand
    else:
        return None, None

def run_annotation():
    base_dir = Path("d:/UserC/Downloads/Tennis-Annotation")
    frames_dir = base_dir / "frames"
    annotated_dir = base_dir / "annotated_frames"
    labels_dir = base_dir / "labels"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("=== TENNIS VIDEO FRAME ANNOTATION SPECIALIST ===")
    print("Loading Faster R-CNN detection model for near player...")
    torch.set_num_threads(8)
    model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights=d.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    model.eval()

    total_frames = 658
    max_frames_to_process = min(total_frames, 1500)
    print(f"Total video frames: {total_frames} | Limit: 1,500 frames | Target: {max_frames_to_process} frames")

    # 1. Track Major (Far Player) across Segment 1 & Segment 3
    print("\nPhase 1: Computing player trajectories...")
    major_boxes = {}
    muster_boxes = {}

    # Segment 1 for Major (Frames 1 - 354)
    img1 = cv2.imread(str(frames_dir / "frame_00001.jpg"))
    j1_box = [360, 36, 24, 40]
    j1_tpl = img1[j1_box[1]:j1_box[1]+j1_box[3], j1_box[0]:j1_box[0]+j1_box[2]]
    cur_j = list(j1_box)
    major_boxes[1] = cur_j

    for f in range(2, 355):
        img = cv2.imread(str(frames_dir / f"frame_{f:05d}.jpg"))
        sjx1 = max(330, cur_j[0] - 15)
        sjy1 = max(34, cur_j[1] - 6)
        sjx2 = min(440, cur_j[0] + cur_j[2] + 15)
        sjy2 = min(80, cur_j[1] + cur_j[3] + 6)
        res_j = cv2.matchTemplate(img[sjy1:sjy2, sjx1:sjx2], j1_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_v_j, _, max_loc_j = cv2.minMaxLoc(res_j)
        cur_j = [sjx1 + max_loc_j[0], sjy1 + max_loc_j[1], j1_box[2], j1_box[3]]
        if max_v_j > 0.55:
            new_tpl = img[cur_j[1]:cur_j[1]+cur_j[3], cur_j[0]:cur_j[0]+cur_j[2]]
            j1_tpl = cv2.addWeighted(j1_tpl, 0.85, new_tpl, 0.15, 0)
        major_boxes[f] = cur_j

    # Segment 3 for Major (Frames 488 - 658)
    img488 = cv2.imread(str(frames_dir / "frame_00488.jpg"))
    j3_box = [346, 36, 24, 40]
    j3_tpl = img488[j3_box[1]:j3_box[1]+j3_box[3], j3_box[0]:j3_box[0]+j3_box[2]]
    cur_j = list(j3_box)
    major_boxes[488] = cur_j

    for f in range(489, 659):
        img = cv2.imread(str(frames_dir / f"frame_{f:05d}.jpg"))
        sjx1 = max(320, cur_j[0] - 15)
        sjy1 = max(34, cur_j[1] - 6)
        sjx2 = min(420, cur_j[0] + cur_j[2] + 15)
        sjy2 = min(80, cur_j[1] + cur_j[3] + 6)
        res_j = cv2.matchTemplate(img[sjy1:sjy2, sjx1:sjx2], j3_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_v_j, _, max_loc_j = cv2.minMaxLoc(res_j)
        cur_j = [sjx1 + max_loc_j[0], sjy1 + max_loc_j[1], j3_box[2], j3_box[3]]
        major_boxes[f] = cur_j

    print("Major tracking completed.")

    # 2. Track Mustermann (Near Player) with Faster R-CNN keyframes and smooth interpolation
    print("Computing Mustermann detections via Faster R-CNN on keyframes...")
    keyframe_indices_seg1 = list(range(1, 355, 6))
    if 354 not in keyframe_indices_seg1:
        keyframe_indices_seg1.append(354)

    keyframe_indices_seg3 = list(range(488, 659, 6))
    if 658 not in keyframe_indices_seg3:
        keyframe_indices_seg3.append(658)

    all_keyframes = keyframe_indices_seg1 + keyframe_indices_seg3
    raw_muster_detections = {}

    for kf in all_keyframes:
        img = cv2.imread(str(frames_dir / f"frame_{kf:05d}.jpg"))
        crop = img[260:432, 100:720]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).unsqueeze(0)
        with torch.no_grad():
            preds = model(tensor)[0]
        best_box = None
        best_score = 0
        for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if label == 1 and score > 0.35:
                x1, y1, x2, y2 = [round(v) for v in box.tolist()]
                gx1, gy1, gx2, gy2 = 100 + x1, 260 + y1, 100 + x2, 260 + y2
                if (gy2 - gy1) >= 60 and score > best_score:
                    best_score = float(score)
                    best_box = [gx1, gy1, gx2 - gx1, gy2 - gy1]
        raw_muster_detections[kf] = best_box

    # Interpolate for Segment 1
    last_known = [600, 294, 50, 118]
    for i in range(len(keyframe_indices_seg1) - 1):
        f_start = keyframe_indices_seg1[i]
        f_end = keyframe_indices_seg1[i + 1]
        b_start = raw_muster_detections[f_start] or last_known
        b_end = raw_muster_detections[f_end] or b_start
        last_known = b_end
        for f in range(f_start, f_end):
            alpha = (f - f_start) / float(f_end - f_start)
            interp = [
                round(b_start[0] + alpha * (b_end[0] - b_start[0])),
                round(b_start[1] + alpha * (b_end[1] - b_start[1])),
                round(b_start[2] + alpha * (b_end[2] - b_start[2])),
                round(b_start[3] + alpha * (b_end[3] - b_start[3])),
            ]
            muster_boxes[f] = interp
    muster_boxes[354] = raw_muster_detections[354] or last_known

    # Interpolate for Segment 3
    last_known = [440, 292, 48, 115]
    for i in range(len(keyframe_indices_seg3) - 1):
        f_start = keyframe_indices_seg3[i]
        f_end = keyframe_indices_seg3[i + 1]
        b_start = raw_muster_detections[f_start] or last_known
        b_end = raw_muster_detections[f_end] or b_start
        last_known = b_end
        for f in range(f_start, f_end):
            alpha = (f - f_start) / float(f_end - f_start)
            interp = [
                round(b_start[0] + alpha * (b_end[0] - b_start[0])),
                round(b_start[1] + alpha * (b_end[1] - b_start[1])),
                round(b_start[2] + alpha * (b_end[2] - b_start[2])),
                round(b_start[3] + alpha * (b_end[3] - b_start[3])),
            ]
            muster_boxes[f] = interp
    muster_boxes[658] = raw_muster_detections[658] or last_known

    print("Mustermann keyframe tracking completed.")

    # 3. Draw annotations on gameplay frames & preserve outplay frames untouched
    print("\nPhase 2: Annotating frames and saving outputs...")
    json_records = []
    csv_rows = [["frame_id", "frame_file", "frame_type", "player_name", "label", "x1", "y1", "x2", "y2", "width", "height"]]

    # High-contrast annotation colors
    COLOR_ACTIVE = (0, 230, 0)      # Vivid Green
    COLOR_PASSIVE = (235, 120, 0)   # Vivid Blue / Cyan
    BG_ACTIVE = (0, 160, 0)
    BG_PASSIVE = (190, 80, 0)

    def draw_player_box(img, box, label, is_active):
        x, y, w, h = box
        color = COLOR_ACTIVE if is_active else COLOR_PASSIVE
        bg_color = BG_ACTIVE if is_active else BG_PASSIVE

        # Tight bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Label tag
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        tag_y2 = y - 4 if y - 4 - th - baseline > 2 else y + th + 6
        tag_y1 = tag_y2 - th - baseline - 4
        tag_x1 = max(0, x)
        tag_x2 = min(img.shape[1], tag_x1 + tw + 8)

        cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), bg_color, -1)
        cv2.putText(img, label, (tag_x1 + 4, tag_y2 - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    outplay_count = 0
    gameplay_count = 0

    for f in range(1, max_frames_to_process + 1):
        frame_name = f"frame_{f:05d}.jpg"
        src_path = frames_dir / frame_name
        dest_path = annotated_dir / frame_name
        yolo_path = labels_dir / f"frame_{f:05d}.txt"

        active_player, passive_player = get_shot_role(f)
        img = cv2.imread(str(src_path))
        img_h, img_w, _ = img.shape

        if active_player is None:
            # === OUTPLAY FRAME ===
            # MUST REMAIN COMPLETELY UNTOUCHED!
            cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            yolo_path.write_text("")
            outplay_count += 1

            json_records.append({
                "frame_id": f,
                "file_name": frame_name,
                "frame_type": "outplay",
                "is_gameplay": False,
                "annotations": []
            })
            csv_rows.append([f, frame_name, "outplay", "None", "None", "", "", "", "", "", ""])

        else:
            # === VALID GAMEPLAY FRAME ===
            gameplay_count += 1
            m_box = muster_boxes[f]
            j_box = major_boxes[f]

            m_is_active = (active_player == 'Mustermann')
            j_is_active = (active_player == 'Major')

            m_label = "Active Player" if m_is_active else "Passive Player"
            j_label = "Active Player" if j_is_active else "Passive Player"

            draw_player_box(img, m_box, m_label, m_is_active)
            draw_player_box(img, j_box, j_label, j_is_active)

            cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            yolo_lines = []
            for box, is_act in [(m_box, m_is_active), (j_box, j_is_active)]:
                cid = 0 if is_act else 1
                xc = (box[0] + box[2] / 2.0) / img_w
                yc = (box[1] + box[3] / 2.0) / img_h
                bw = box[2] / float(img_w)
                bh = box[3] / float(img_h)
                yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            yolo_path.write_text("\n".join(yolo_lines) + "\n")

            frame_annotations = [
                {
                    "player": "Mustermann",
                    "role": m_label,
                    "is_active": m_is_active,
                    "bbox_xywh": m_box,
                    "bbox_xyxy": [m_box[0], m_box[1], m_box[0] + m_box[2], m_box[1] + m_box[3]]
                },
                {
                    "player": "Major",
                    "role": j_label,
                    "is_active": j_is_active,
                    "bbox_xywh": j_box,
                    "bbox_xyxy": [j_box[0], j_box[1], j_box[0] + j_box[2], j_box[1] + j_box[3]]
                }
            ]

            json_records.append({
                "frame_id": f,
                "file_name": frame_name,
                "frame_type": "gameplay",
                "is_gameplay": True,
                "annotations": frame_annotations
            })

            csv_rows.append([f, frame_name, "gameplay", "Mustermann", m_label, m_box[0], m_box[1], m_box[0]+m_box[2], m_box[1]+m_box[3], m_box[2], m_box[3]])
            csv_rows.append([f, frame_name, "gameplay", "Major", j_label, j_box[0], j_box[1], j_box[0]+j_box[2], j_box[1]+j_box[3], j_box[2], j_box[3]])

        if f % 100 == 0 or f == max_frames_to_process:
            print(f"Processed frame {f}/{max_frames_to_process}...")

    # Save JSON and CSV
    json_path = base_dir / "annotations.json"
    with open(json_path, "w") as jf:
        json.dump({
            "video": "Tennis.mp4",
            "total_frames_processed": max_frames_to_process,
            "gameplay_frames_count": gameplay_count,
            "outplay_frames_count": outplay_count,
            "categories": {"0": "Active Player", "1": "Passive Player"},
            "frames": json_records
        }, jf, indent=2)

    csv_path = base_dir / "annotations.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerows(csv_rows)

    print(f"\nMetadata saved:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV:  {csv_path}")
    print(f"  - YOLO labels: {labels_dir}")
    print(f"  - Annotated frames: {annotated_dir}")

    # Generate annotated MP4 video
    print("\nPhase 3: Rendering annotated video (annotated_tennis.mp4)...")
    video_out = base_dir / "annotated_tennis.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(annotated_dir / "frame_%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_out)
    ]
    subprocess.run(cmd, check=True)
    print(f"Annotated video saved successfully to: {video_out}")
    print(f"Task completed: {gameplay_count} gameplay frames annotated, {outplay_count} outplay frames preserved untouched.")

if __name__ == "__main__":
    t_start = time.time()
    run_annotation()
    print(f"Total time elapsed: {time.time() - t_start:.2f}s")
