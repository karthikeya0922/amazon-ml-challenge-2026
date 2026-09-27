"""Paths and global settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project root (contains Dataset/, output/, work/)
DATA = ROOT / "Dataset" / "student_resource" / "dataset"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"

WORK.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

SEED = 2026
VALID_MOD = 10  # S1 entities with hash(id) % VALID_MOD == 0 form the validation split
