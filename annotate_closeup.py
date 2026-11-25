"""
Annotate frames 355-487 (close-up broadcast segment).
- Mustermann is the ACTIVE player (mid-swing, only visible player)
- No Passive Player visible in these frames
- Tight bounding box via Faster R-CNN MobileNetV3 on full frame
- Update annotated_frames/, labels/, annotations.json, annotations.csv
- Re-render annotated_tennis.mp4
"""

import os, json, csv, subprocess, time
import cv2
import numpy as np
import torch
import torchvision.models.detection as det
from torchvision.transforms import functional as F
from pathlib import Path
import shutil

BASE    = Path("d:/UserC/Downloads/Tennis-Annotation")
ORIG    = BASE / "frames"
ANN     = BASE / "annotated_frames"
LAB     = BASE / "labels"

FRAME_START = 355
FRAME_END   = 487

# Active player in the close-up is Mustermann (hitting the ball)
ACTIVE_LABEL = "Active Player"

COLOR_ACTIVE = (0, 230, 0)   # Vivid green (BGR)
BG_ACTIVE    = (0, 160, 0)

def draw_player_box(img, box, label, color, bg_color):
    """Draw tight bounding box + label badge on img (in-place)."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    tag_y2 = y - 4 if y - 4 - th - baseline > 2 else y + th + 8
    tag_y1 = tag_y2 - th - baseline - 4
    tag_x1 = max(0, x)
    tag_x2 = min(img.shape[1], tag_x1 + tw + 8)

    cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), bg_color, -1)
    cv2.putText(img, label, (tag_x1 + 4, tag_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def detect_person_boxes(model, img_bgr, min_score=0.35, min_height=80):
    """
    Run Faster R-CNN on a full-frame BGR image.
    Returns list of [x, y, w, h] boxes (class=person, score>threshold, height>min_height).
    Sorted by score descending.
    """
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).unsqueeze(0)
    with torch.no_grad():
        preds = model(tensor)[0]

    boxes = []
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if int(label) == 1 and float(score) > min_score:
            x1, y1, x2, y2 = [round(float(v)) for v in box.tolist()]
            w, h = x2 - x1, y2 - y1
            if h >= min_height:
                boxes.append((float(score), [x1, y1, w, h]))
    boxes.sort(key=lambda b: -b[0])
    return [b[1] for b in boxes]


def run():
    t0 = time.time()
    torch.set_num_threads(8)

    print("Loading Faster R-CNN MobileNetV3 320 FPN...")
    model = det.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    model.eval()

    # ── Phase 1: Detect on every 4th keyframe, interpolate between ──────────
    print(f"Detecting player on keyframes {FRAME_START}–{FRAME_END}...")
    keyframes = list(range(FRAME_START, FRAME_END + 1, 4))
    if FRAME_END not in keyframes:
        keyframes.append(FRAME_END)

    kf_boxes = {}  # frame_num -> [x,y,w,h] or None
    for kf in keyframes:
        img = cv2.imread(str(ORIG / f"frame_{kf:05d}.jpg"))
        boxes = detect_person_boxes(model, img, min_score=0.30, min_height=60)
        kf_boxes[kf] = boxes[0] if boxes else None
        status = kf_boxes[kf] if kf_boxes[kf] else "NOT DETECTED"
        print(f"  KF {kf}: {status}")

    # Fill None gaps using nearest-known box
    # Forward fill
    last = None
    for kf in keyframes:
        if kf_boxes[kf] is not None:
            last = kf_boxes[kf]
        elif last is not None:
            kf_boxes[kf] = last
    # Backward fill (for leading Nones)
    last = None
    for kf in reversed(keyframes):
        if kf_boxes[kf] is not None:
            last = kf_boxes[kf]
        elif last is not None:
            kf_boxes[kf] = last

    # Interpolate all frames between keyframes
    all_boxes = {}
    for i in range(len(keyframes) - 1):
        f_s = keyframes[i]
        f_e = keyframes[i + 1]
        b_s = kf_boxes[f_s]
        b_e = kf_boxes[f_e]
        if b_s is None:
            b_s = b_e
        if b_e is None:
            b_e = b_s
        for f in range(f_s, f_e):
            alpha = (f - f_s) / float(f_e - f_s)
            all_boxes[f] = [
                round(b_s[j] + alpha * (b_e[j] - b_s[j])) for j in range(4)
            ]
    all_boxes[FRAME_END] = kf_boxes[FRAME_END]

    # ── Phase 2: Draw boxes, write annotated frames + YOLO labels ───────────
    print(f"\nDrawing annotations on frames {FRAME_START}–{FRAME_END}...")
    img_w, img_h = 768, 432  # all frames are this size

    new_ann_records = {}   # frame_id -> annotation list (for JSON merge)
    new_csv_rows    = []

    for f in range(FRAME_START, FRAME_END + 1):
        fname = f"frame_{f:05d}.jpg"
        img   = cv2.imread(str(ORIG / fname))

        box   = all_boxes.get(f)
        if box is None:
            # No detection at all — copy original untouched
            shutil.copy2(str(ORIG / fname), str(ANN / fname))
            (LAB / f"frame_{f:05d}.txt").write_text("")
            new_ann_records[f] = []
            new_csv_rows.append([f, fname, "gameplay_closeup", "Mustermann",
                                  "Active Player", "", "", "", "", "", ""])
            continue

        draw_player_box(img, box, ACTIVE_LABEL, COLOR_ACTIVE, BG_ACTIVE)
        cv2.imwrite(str(ANN / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # YOLO label (class 0 = Active Player)
        x, y, w, h = box
        xc = (x + w / 2.0) / img_w
        yc = (y + h / 2.0) / img_h
        bw = w / float(img_w)
        bh = h / float(img_h)
        (LAB / f"frame_{f:05d}.txt").write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        new_ann_records[f] = [{
            "player":    "Mustermann",
            "role":      "Active Player",
            "is_active": True,
            "bbox_xywh": box,
            "bbox_xyxy": [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        }]
        new_csv_rows.append([f, fname, "gameplay_closeup", "Mustermann",
                              "Active Player", box[0], box[1],
                              box[0]+box[2], box[1]+box[3], box[2], box[3]])

        if (f - FRAME_START) % 20 == 0 or f == FRAME_END:
            print(f"  Annotated frame {f}/{FRAME_END}...")

    # ── Phase 3: Merge into annotations.json ────────────────────────────────
    print("\nUpdating annotations.json...")
    json_path = BASE / "annotations.json"
    with open(json_path, "r") as jf:
        data = json.load(jf)

    for rec in data["frames"]:
        fid = rec["frame_id"]
        if FRAME_START <= fid <= FRAME_END:
            rec["frame_type"] = "gameplay_closeup"
            rec["is_gameplay"] = True
            rec.pop("scene_type", None)
            rec.pop("shot_type", None)
            rec.pop("is_outplay", None)
            rec.pop("preservation", None)
            rec["annotations"] = new_ann_records.get(fid, [])

    # Update summary counts
    gameplay_total = sum(1 for r in data["frames"] if r.get("is_gameplay"))
    outplay_total  = sum(1 for r in data["frames"] if not r.get("is_gameplay"))
    data["gameplay_frames_count"] = gameplay_total
    data["outplay_frames_count"]  = outplay_total

    with open(json_path, "w") as jf:
        json.dump(data, jf, indent=2)
    print(f"  JSON updated: {gameplay_total} gameplay, {outplay_total} outplay frames")

    # ── Phase 4: Merge into annotations.csv ─────────────────────────────────
    print("Updating annotations.csv...")
    csv_path = BASE / "annotations.csv"
    with open(csv_path, "r", newline="") as cf:
        reader = csv.reader(cf)
        rows   = list(reader)

    header      = rows[0]
    # Keep rows outside 355-487, replace those inside
    kept_rows   = [header]
    for row in rows[1:]:
        try:
            fid = int(row[0])
        except (ValueError, IndexError):
            continue
        if FRAME_START <= fid <= FRAME_END:
            continue   # will be replaced
        kept_rows.append(row)

    # Insert new_csv_rows in frame order
    all_data_rows = kept_rows[1:] + new_csv_rows
    all_data_rows.sort(key=lambda r: int(r[0]))

    # Ensure scene_type / shot_type columns exist (added in earlier pass)
    # Pad or trim to match header width
    target_len = len(header)
    final_rows = [header]
    for row in all_data_rows:
        while len(row) < target_len:
            row.append("")
        final_rows.append(row[:target_len])

    with open(csv_path, "w", newline="") as cf:
        csv.writer(cf).writerows(final_rows)
    print(f"  CSV updated: {len(final_rows)-1} data rows total")

    # ── Phase 5: Re-render annotated_tennis.mp4 ─────────────────────────────
    print("\nRendering annotated_tennis.mp4...")
    video_out = BASE / "annotated_tennis.mp4"
    result = subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", str(ANN / "frame_%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(video_out)
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg stderr:", result.stderr[-1000:])
    else:
        size = os.path.getsize(str(video_out))
        print(f"  Video saved: {video_out} ({size:,} bytes)")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"  Annotated frames 355–487: {FRAME_END - FRAME_START + 1} frames")
    print(f"  YOLO labels updated for frames 355–487")

if __name__ == "__main__":
    run()
