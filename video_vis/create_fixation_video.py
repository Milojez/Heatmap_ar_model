"""
create_fixation_video.py
========================
Creates a 2-second video for one test sample with the model's predicted
fixations overlaid in time.

Timeline (ms):
  saccade_i  : delta_t_start[i]  — eye moving to next fixation
  fixation_i : T[i]              — eye dwelling on (x, y)

During a fixation → filled circle at (x, y)
After  a fixation → numbered dot stays on screen, path line grows

─── Are predictions deterministic? ───────────────────────────────────────────
NO. Every run produces different fixations because sample_from_heatmap uses:
  1. torch.multinomial  — stochastic cell selection from the probability map
  2. torch.rand         — uniform sub-cell jitter within the chosen cell
Neither is seeded by default, so each run is an independent sample from the
model's learned distribution.  Set INFERENCE_SEED to an integer to fix the
random state and get reproducible results across runs.
──────────────────────────────────────────────────────────────────────────────

Edit the USER SETTINGS block below, then run:
    python create_fixation_video.py
"""

import sys
import os
import json
import re
import torch
import numpy as np
import cv2

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from dataset import ScanpathDataset
from model import HeatmapARModel

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
VIDEO_NUM    = 7     # which video (e.g. 7 → video_7_frame_*.png)
START_SECOND = 66     # start of the 2-second window (e.g. 0, 2, 4, 6 …)
SUBJECT_ID   = None  # participant to use; None = first available in test set

INFERENCE_SEED = None  # set to an int (e.g. 42) for reproducible predictions;
                       # None = new stochastic sample every run
OUTPUT_FPS   = 30         # frames per second of the output video
FIXATION_R   = 28         # fixation circle radius in pixels
FIXATION_CLR = (0, 80, 255)     # BGR — blue circle during active fixation
COMPLETED_CLR = (0, 200, 255)  # BGR — colour of completed fixation markers
PATH_CLR     = (200, 200, 200) # BGR — line connecting completed fixations

# Map video_N index → source mp4 file (add paths as you obtain the files)
_VIDEOS_DIR = "C:/Users/milos/Desktop/fixations_thesis/Yke_data/all/videos"
VIDEO_MAP = {
    1: f"{_VIDEOS_DIR}/Effort_Level_1.mp4",
    2: f"{_VIDEOS_DIR}/Effort_Level_2.mp4",
    3: f"{_VIDEOS_DIR}/Effort_Level_3.mp4",
    4: f"{_VIDEOS_DIR}/Effort_Level_4.mp4",
    5: f"{_VIDEOS_DIR}/Effort_Level_5.mp4",
    6: f"{_VIDEOS_DIR}/Effort_Level_6.mp4",
    7: f"{_VIDEOS_DIR}/Effort_Level_7.mp4",
}
SOURCE_FPS = 50.0   # native FPS of the source videos

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", config.MODEL_NAME)
# ─────────────────────────────────────────────────────────────────────────────


def frame_key(name):
    first = name[0] if isinstance(name, list) else name
    m = re.match(r'.+?_frame_(\d+)', first)
    return int(m.group(1)) if m else 0


def video_id(name):
    first = name[0] if isinstance(name, list) else name
    m = re.match(r'(video_\d+)', first)
    return m.group(1) if m else ''


def find_sample(test_raw, video_num, start_second, subject_id=None):
    """
    Return the raw sample from test_raw whose start frame is closest to
    `start_second` for `video_{video_num}`, optionally filtered by subject.

    Mapping: at ~50 fps each 2-sec clip spans 100 frames, so
        start_frame ≈ start_second * 50 + 1
    """
    target_vid   = f"video_{video_num}"
    target_frame = start_second * 50 + 1

    candidates = [s for s in test_raw if video_id(s['name']) == target_vid]
    if not candidates:
        raise ValueError(f"No test samples found for video_{video_num}.")
    if subject_id is not None:
        candidates = [s for s in candidates if s.get('subject') == subject_id]
        if not candidates:
            raise ValueError(
                f"No test samples for video_{video_num} with subject={subject_id}.")

    best = min(candidates, key=lambda s: abs(frame_key(s['name']) - target_frame))
    actual_second = (frame_key(best['name']) - 1) / 50
    print(f"Found: video_{video_num}  start_frame={frame_key(best['name'])}"
          f"  (~{actual_second:.1f}s)  subject={best.get('subject','?')}")
    return best


def load_model(device):
    ckpt_path = (config.BEST_CHECKPOINT_PATH
                 if os.path.exists(config.BEST_CHECKPOINT_PATH)
                 else config.CHECKPOINT_PATH)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = HeatmapARModel().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {os.path.basename(ckpt_path)}  epoch={ckpt.get('epoch','?')}")
    return model


def run_inference(model, item, device):
    """Returns (x_px, y_px, dt_ms, T_ms) arrays, each of length `length`."""
    from dataset import denorm_delta_t, denorm_duration

    length      = item['length']
    cond_geom   = item['cond_geom'].unsqueeze(0).to(device)
    cond_signal = item['cond_signal'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.generate(
            cond_geom, cond_signal,
            num_fixations=length,
            temperature=config.INFERENCE_TEMPERATURE,
        ).squeeze(0).cpu()   # [length, 4]: x_norm, y_norm, dt_norm, T_norm

    W, H = config.IMAGE_WIDTH, config.IMAGE_HEIGHT

    x_px = ((pred[:, 0] + 1) / 2 * W).numpy()
    y_px = ((pred[:, 1] + 1) / 2 * H).numpy()

    # Load norm_stats to denormalise temporal predictions
    train_ds = ScanpathDataset(config.TRAIN_JSON,
                               max_fixations=config.MAX_FIXATIONS,
                               max_samples=config.MAX_TRAIN_SAMPLES)
    stats = train_ds.norm_stats
    dt_ms = denorm_delta_t(pred[:, 2], stats).numpy()   # ms
    T_ms  = denorm_duration(pred[:, 3], stats).numpy()  # ms

    return x_px[:length], y_px[:length], dt_ms[:length], T_ms[:length]


def build_timeline(x_px, y_px, dt_ms, T_ms):
    """
    Returns list of dicts, one per fixation:
      saccade_start, saccade_end, fixation_start, fixation_end  (all in ms)
      cx, cy  (pixel coordinates)
    """
    events = []
    t = 0.0
    prev_cx, prev_cy = config.IMAGE_WIDTH / 2, config.IMAGE_HEIGHT / 2
    for i, (cx, cy, dt, dur) in enumerate(zip(x_px, y_px, dt_ms, T_ms)):
        events.append({
            'saccade_start':  t,
            'saccade_end':    t + dt,
            'fixation_start': t + dt,
            'fixation_end':   t + dt + dur,
            'cx': float(cx), 'cy': float(cy),
            'prev_cx': prev_cx, 'prev_cy': prev_cy,
            'idx': i,
        })
        t += dt + dur
        prev_cx, prev_cy = float(cx), float(cy)
    return events


def load_keyframes(name_list):
    """
    Extract every frame between f_min and f_max (inclusive) directly from
    the source video file.  All video_N share the same recording, so frame
    numbers map 1-to-1 to video frame indices (1-based in filenames →
    0-based in OpenCV: cv2_idx = frame_num - 1).

    Returns list of (t_ms, img_bgr) sorted by frame number.
    """
    nums = [int(re.search(r'_frame_(\d+)', n).group(1)) for n in name_list]
    f_min, f_max = min(nums), max(nums)
    span = f_max - f_min if f_max != f_min else 1

    video_file = VIDEO_MAP.get(VIDEO_NUM)
    if video_file is None:
        raise ValueError(f"No video mapped for VIDEO_NUM={VIDEO_NUM}")
    if not os.path.exists(video_file):
        raise FileNotFoundError(
            f"Video file not found: {video_file}\n"
            f"Add it to the videos directory or update VIDEO_MAP.")
    cap = cv2.VideoCapture(video_file)
    print(f"  Source: {os.path.basename(video_file)}")
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_FILE}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_min - 1)   # 0-based index

    kf = []
    for fnum in range(f_min, f_max + 1):
        ok, img = cap.read()
        if not ok:
            break
        t_ms = (fnum - f_min) / span * 2000.0
        kf.append((t_ms, img))

    cap.release()
    print(f"  Extracted {len(kf)} frames from video "
          f"(frames {f_min}–{f_max})")
    return kf


def get_frame_at(kf, t_ms):
    """Return the keyframe image whose timestamp is closest and <= t_ms."""
    img = kf[0][1]
    for t_kf, kf_img in kf:
        if t_kf <= t_ms:
            img = kf_img
        else:
            break
    return img.copy()


def draw_frame(base_img, t_ms, events, completed):
    """Overlay fixation state at time t_ms onto base_img."""
    canvas = base_img.copy()

    # Path line connecting all completed fixation centres in order
    if len(completed) > 1:
        pts = [(int(e['cx']), int(e['cy'])) for e in completed]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, PATH_CLR, 2, cv2.LINE_AA)

    # Completed fixations: numbered dot
    for e in completed:
        cx_i, cy_i = int(e['cx']), int(e['cy'])
        cv2.circle(canvas, (cx_i, cy_i), 10, COMPLETED_CLR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx_i, cy_i), 10, (255, 255, 255), 1, cv2.LINE_AA)
        label = str(e['idx'] + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(canvas, label,
                    (cx_i - tw // 2, cy_i + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)

    # Active fixation: large filled circle
    for e in events:
        if e['fixation_start'] <= t_ms < e['fixation_end']:
            cx_i, cy_i = int(e['cx']), int(e['cy'])
            cv2.circle(canvas, (cx_i, cy_i), FIXATION_R,
                       FIXATION_CLR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx_i, cy_i), FIXATION_R,
                       (255, 255, 255), 2, cv2.LINE_AA)

    # Timestamp
    cv2.putText(canvas, f"{t_ms:.0f} ms",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test JSON and pick sample by video + second
    with open(config.TEST_JSON, encoding='utf-8') as f:
        test_raw = json.load(f)
    raw_s = find_sample(test_raw, VIDEO_NUM, START_SECOND, SUBJECT_ID)
    print(f"length={raw_s['length']}  name={raw_s['name']}")

    # Dataset + model
    train_ds = ScanpathDataset(config.TRAIN_JSON,
                               max_fixations=config.MAX_FIXATIONS,
                               max_samples=config.MAX_TRAIN_SAMPLES)
    test_ds  = ScanpathDataset(config.TEST_JSON,
                               max_fixations=config.MAX_FIXATIONS,
                               norm_stats=train_ds.norm_stats,
                               cond_norm_stats=train_ds.cond_norm_stats)

    # Find this sample in test_ds by (subject, frame_key)
    fk  = frame_key(raw_s['name'])
    pid = raw_s.get('subject')
    idx = next((i for i, s in enumerate(test_ds.data)
                if s.get('subject') == pid and frame_key(s['name']) == fk), None)
    if idx is None:
        raise ValueError(f"Sample (subject={pid}, frame={fk}) not found in test_ds.")

    model = load_model(device)
    item  = test_ds[idx]

    if INFERENCE_SEED is not None:
        torch.manual_seed(INFERENCE_SEED)
        print(f"Inference seed: {INFERENCE_SEED} (reproducible)")
    else:
        print("Inference seed: None (stochastic — different each run)")

    print("Running inference...")
    x_px, y_px, dt_ms, T_ms = run_inference(model, item, device)

    print("Fixation predictions:")
    t = 0.0
    for i in range(len(x_px)):
        print(f"  F{i+1}: saccade={dt_ms[i]:.0f}ms  fix=({x_px[i]:.0f},{y_px[i]:.0f})px  dur={T_ms[i]:.0f}ms")
        t += dt_ms[i] + T_ms[i]
    print(f"  Total: {t:.0f} ms")

    # Build timeline
    events   = build_timeline(x_px, y_px, dt_ms, T_ms)
    clip_end = sum(e['fixation_end'] for e in events[-1:]) or 2000.0
    clip_end = min(clip_end, 2000.0)

    # Load all available frames for this clip
    name_list = raw_s['name'] if isinstance(raw_s['name'], list) else [raw_s['name']]
    kf = load_keyframes(name_list)
    H_img, W_img = kf[0][1].shape[:2]

    # Video writer
    os.makedirs(OUT_DIR, exist_ok=True)
    subj_tag = f"_s{raw_s.get('subject','?')}" if SUBJECT_ID is not None else ""
    seed_tag = f"_seed{INFERENCE_SEED}" if INFERENCE_SEED is not None else ""
    out_path = os.path.join(
        OUT_DIR,
        f"video{VIDEO_NUM}_t{START_SECOND}s{subj_tag}_T{config.INFERENCE_TEMPERATURE}{seed_tag}.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (W_img, H_img))

    n_frames    = int(OUTPUT_FPS * 2.0)          # always exactly 2 seconds
    ms_per_vfr  = 2000.0 / n_frames

    completed = []

    for vfr in range(n_frames):
        t_ms = vfr * ms_per_vfr

        # Move finished fixations to completed
        for e in events:
            if e['fixation_end'] <= t_ms and e not in completed:
                completed.append(e)

        # Events currently in their fixation window
        active = [e for e in events
                  if e['fixation_start'] <= t_ms < e['fixation_end']]

        base = get_frame_at(kf, t_ms)
        out  = draw_frame(base, t_ms, active, completed)
        writer.write(out)

    writer.release()
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
