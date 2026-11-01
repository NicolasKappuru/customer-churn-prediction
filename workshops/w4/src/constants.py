"""Workshop 4 project-wide constants and default paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W4_ROOT = PROJECT_ROOT / "workshops" / "w4"
W1_ROOT = PROJECT_ROOT / "workshops" / "w1"

W1_EXPORT_DIR = W1_ROOT / "export-tabular-v5"
W1_SPLIT_PATH = W1_ROOT / "artifacts" / "sherlock_w1" / "split_indices.npz"

CHECKPOINTS_DIR = W4_ROOT / "checkpoints"
RUNS_DIR        = W4_ROOT / "src" / "runs"
FIGURES_DIR     = RUNS_DIR / "figures"

SEED = 110

Z_DIMS: tuple[int, ...] = (8, 16, 32)

SKEWNESS_THRESHOLD = 1.0
CLIP_LOW  = 0.001
CLIP_HIGH = 0.999
