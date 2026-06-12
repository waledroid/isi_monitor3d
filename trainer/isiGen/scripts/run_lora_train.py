"""Phase 4 — train the project's SDXL LoRA (fp16 UNet, 12 GB recipe).

  python scripts/run_lora_train.py --project pallets_v1
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_lora


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    print(run_lora(_bootstrap.project_dir(args.project)))


if __name__ == "__main__":
    main()
