"""
Configuration for the Heatmap Autoregressive model.

Architecture:
  Transformer encoder over 60 conditioning tokens (6 dials × 10 frames)
  → GRU decoder + cross-attention → 2D heatmap (96×50) per fixation step
"""

# ─────────────────────────────────────────────
MODEL_NAME = "hm_ar_256_ur_dis_gur1"

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_MODEL_ROOT = "C:/Users/milos/Desktop/fixations_thesis/Adapt_yke_dataset/heatmap_ar_model"
_DATA_ROOT  = "C:/Users/milos/Desktop/fixations_thesis/ScanDiff/ScanDiff_video/yke_fix_data_unstructured/split_2sec_1500mHZ"

TRAIN_JSON = f"{_DATA_ROOT}/ykedata_2s_fix_vid_gaze_train_signal.json"
VAL_JSON   = f"{_DATA_ROOT}/ykedata_2s_fix_vid_gaze_validation_signal.json"
TEST_JSON  = f"{_DATA_ROOT}/ykedata_2s_fix_vid_gaze_test_signal.json"
FRAMES_DIR = f"{_MODEL_ROOT}/data/frames"
VIS_DIR    = f"{_MODEL_ROOT}/data/visualizations/{MODEL_NAME}"
CHECKPOINT_DIR       = f"{_MODEL_ROOT}/checkpoints/{MODEL_NAME}"
CHECKPOINT_PATH      = f"{_MODEL_ROOT}/checkpoints/{MODEL_NAME}/{MODEL_NAME}_last.pth"
BEST_CHECKPOINT_PATH = f"{_MODEL_ROOT}/checkpoints/{MODEL_NAME}/{MODEL_NAME}_best.pth"
LOG_PATH             = f"{_MODEL_ROOT}/checkpoints/{MODEL_NAME}/{MODEL_NAME}_log.csv"
PLOT_PATH            = f"{_MODEL_ROOT}/checkpoints/{MODEL_NAME}/{MODEL_NAME}_metrics.png"
PRED_DIR             = f"{_MODEL_ROOT}/data/predictions/{MODEL_NAME}"

# ─────────────────────────────────────────────
# Data / canvas
# ─────────────────────────────────────────────
IMAGE_WIDTH       = 1904
IMAGE_HEIGHT      = 988
MAX_FIXATIONS     = 12
MAX_TRAIN_SAMPLES = 10000    # set None to use all
MAX_VAL_SAMPLES   = 64
NUM_DIALS         = 6
NUM_SIGNAL_FRAMES = 10

# ─────────────────────────────────────────────
# Conditioning signal features
# ─────────────────────────────────────────────
SIGNAL_FEATURES = ['urgency', 'distance']
_FEATURE_DIMS   = {'sin_cos': 2, 'speed': 1, 'urgency': 1, 'distance': 1}
SIGNAL_DIM      = sum(_FEATURE_DIMS[f] for f in SIGNAL_FEATURES)

# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────
HM_W          = 96    # heatmap width  (columns)  → cell = 1904/96 ≈ 19.8 px
HM_H          = 50    # heatmap height (rows)     → cell = 988/50  ≈ 19.8 px
HM_SIGMA_CELLS = 1.5  # Gaussian sigma for target smoothing (2.5 gives~50 px in image space)

# ─────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────
HIDDEN_DIM         = 256
NUM_HEADS          = 4
NUM_ENCODER_BLOCKS = 3    # transformer encoder depth over conditioning tokens
NUM_GRU_LAYERS     = 1    # GRU decoder depth
ATTN_DROPOUT       = 0.1
# Std of Gaussian noise added to cond_signal during training (0 = disabled).
# Signals are normalised to roughly [-1, 1]; 0.05 is a mild perturbation.
COND_SIGNAL_NOISE_STD = 0
# How many previous fixations the GRU carries as hidden state.
#   1  = only the immediately preceding fixation (h_state reset every step)
#   N  = reset every N steps — GRU sees chunks of N consecutive fixations
#   0  = never reset — full accumulated trajectory history (original behaviour)
GRU_HISTORY_STEPS  = 1

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
SEED           = 42
EPOCHS         = 200
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
LR_MIN         = 1e-6
WARMUP_EPOCHS  = 5
LR_RESTART_EVERY = 100  # cosine restart period in epochs (0 = no restarts)
WEIGHT_DECAY   = 1e-4
GRAD_CLIP_NORM = 1.0
ACCUM_STEPS    = 1

# Relative weight of temporal (dt, T) MSE loss vs spatial heatmap CE loss
# Keep small — heatmap loss is the primary objective
LAMBDA_TEMPORAL = 0.1

SAVE_EVERY = 0   # 0 = disabled

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
VAL_BATCH_SIZE       = 64
EVAL_EVERY           = 25
EVAL_RUNS            = 3
INFERENCE_TEMPERATURE = 0.1   # softmax temperature at sampling (1.0 = raw probs)

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
T_RADIUS_SCALE  = 0.25
T_RADIUS_MIN_PX = 5.0
