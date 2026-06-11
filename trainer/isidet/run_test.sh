#!/usr/bin/env bash
# run_test.sh — quick sanity-check prediction on a RANDOM HALF of a folder's images.
#
# Interactively asks for the images folder and the model (pre-filled with smart
# defaults — just press Enter to accept). Samples ~50% of the images at random,
# runs the trained YOLO model, saves annotated predictions to
# ./mytest_time<YYYYmmdd_HHMMSS>/ (+ stats.txt + sampled_files.txt), and prints
# detection stats.
#
# Usage (run from trainer/isidet/):
#   ./run_test.sh                       # fully interactive (prompts for everything)
#   ./run_test.sh <images_dir>          # pre-fills the folder prompt
#   ./run_test.sh <images_dir> <model.pt>
#
# Requires the isi-train env:  conda activate isi-train

set -euo pipefail

# Reduce CUDA fragmentation OOMs (recommended by the torch error message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ISIDET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISIDET_DIR}/../.." && pwd)"

# Defaults to pre-fill the prompts.
AUTO_MODEL="$(ls -t "${ISIDET_DIR}"/runs/detect/models/yolo/*/weights/best.pt 2>/dev/null | head -n 1 || true)"
SRC_DEFAULT="${1:-${REPO_ROOT}/video/p/frames}"
MODEL_DEFAULT="${2:-${AUTO_MODEL}}"

echo "=== run_test.sh — sanity-check prediction ==="
read -e -i "${SRC_DEFAULT}"   -p "Images folder  : " SRC
read -e -i "${MODEL_DEFAULT}" -p "Model (.pt)    : " MODEL
read -e -i "0.25"             -p "Conf threshold : " CONF
read -e -i "0.5"              -p "Sample fraction: " FRAC
read -e -i "640"             -p "Inference imgsz: " IMGSZ

# Validate.
if [[ ! -d "${SRC}" ]]; then echo "ERROR: '${SRC}' is not a directory"; exit 1; fi
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then echo "ERROR: model not found: '${MODEL}'"; exit 1; fi

TS="$(date +%Y%m%d_%H%M%S)"

# Build a descriptive model tag: for .../<run>/weights/best.pt use "<run>_best";
# for a bare checkpoint like yolo11m.pt use "yolo11m".
WSTEM="$(basename "${MODEL}" .pt)"
if [[ "$(basename "$(dirname "${MODEL}")")" == "weights" ]]; then
  MODEL_TAG="$(basename "$(dirname "$(dirname "${MODEL}")")")_${WSTEM}"
else
  MODEL_TAG="${WSTEM}"
fi
# Sanitize for a folder name (drop anything that isn't alnum . _ -).
MODEL_TAG="$(echo "${MODEL_TAG}" | tr -c 'A-Za-z0-9._-' '_')"
CONF_TAG="conf${CONF}"

RUN_NAME="mytest_${MODEL_TAG}_${CONF_TAG}_${TS}"

echo
echo "Model : ${MODEL}"
echo "Source: ${SRC}"
echo "Output: ${ISIDET_DIR}/${RUN_NAME}"
echo "Conf  : ${CONF} | sample fraction: ${FRAC} | imgsz: ${IMGSZ}"
echo

python - "${SRC}" "${MODEL}" "${ISIDET_DIR}" "${RUN_NAME}" "${CONF}" "${FRAC}" "${IMGSZ}" <<'PY'
import glob, os, random, sys
import numpy as np

src, model_path, project, name, conf, frac, imgsz = sys.argv[1:8]
conf = float(conf); frac = float(frac); imgsz = int(imgsz)
out_dir = os.path.join(project, name)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
imgs = [p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True)
        if p.lower().endswith(EXTS) and not p.endswith("Zone.Identifier")]
if not imgs:
    print("ERROR: no images found in", src); sys.exit(1)

random.seed(42)
k = max(1, round(len(imgs) * frac))
sample = random.sample(imgs, k)
print(f"Images in dir: {len(imgs)}  ->  sampled: {len(sample)} ({frac:.0%})")

from ultralytics import YOLO
from tqdm import tqdm
import torch

model = YOLO(model_path)
names = model.names

def collect(device, half):
    """Predict on the sample at the given device/precision; return stats.

    stream=True yields one Result at a time (effective batch = 1), so memory is
    driven by `imgsz` + model size, not by a batch dimension. imgsz caps the
    input resolution (the real OOM lever for large source frames + a big model).
    """
    n_with = total_det = 0
    confs, per_class = [], {}
    gen = model.predict(sample, conf=conf, imgsz=imgsz, save=True,
                        project=project, name=name, exist_ok=True,
                        verbose=False, stream=True, device=device, half=half)
    tag = "GPU/FP16" if device == 0 else "CPU"
    for r in tqdm(gen, total=len(sample), desc=f"Predicting[{tag}]", unit="img"):
        nb = len(r.boxes)
        total_det += nb
        n_with += 1 if nb else 0
        for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            confs.append(cf)
            per_class[names[int(c)]] = per_class.get(names[int(c)], 0) + 1
    return n_with, total_det, confs, per_class

# GPU at FP16 (half the memory); fall back to CPU on OOM so the check always finishes.
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        n_with, total_det, confs, per_class = collect(0, True)
    else:
        n_with, total_det, confs, per_class = collect("cpu", False)
except torch.cuda.OutOfMemoryError:
    print("\n⚠️  CUDA OOM — clearing cache and retrying on CPU (slower)...")
    torch.cuda.empty_cache()
    n_with, total_det, confs, per_class = collect("cpu", False)

lines = []
lines.append("=" * 48)
lines.append(" SANITY-CHECK PREDICTION STATS")
lines.append("=" * 48)
lines.append(f" model                  : {model_path}")
lines.append(f" source dir             : {src}")
lines.append(f" images in dir          : {len(imgs)}")
lines.append(f" sampled (random {frac:.0%})   : {len(sample)}")
lines.append(f" conf threshold         : {conf}")
lines.append(f" inference imgsz        : {imgsz}")
lines.append("-" * 48)
lines.append(f" images with >=1 det    : {n_with} ({n_with/len(sample):.0%})")
lines.append(f" images with NO det     : {len(sample) - n_with}")
lines.append(f" total detections       : {total_det}")
lines.append(f" avg detections / image : {total_det/len(sample):.2f}")
if confs:
    a = np.array(confs)
    lines.append(f" confidence mean/min/max: {a.mean():.3f} / {a.min():.3f} / {a.max():.3f}")
lines.append(f" per-class counts       : {per_class or '{}'}")
lines.append("=" * 48)
report = "\n".join(lines)
print("\n" + report)

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "stats.txt"), "w") as f:
    f.write(report + "\n")
with open(os.path.join(out_dir, "sampled_files.txt"), "w") as f:
    f.write("\n".join(sorted(sample)) + "\n")
print(f"\nAnnotated images + stats.txt + sampled_files.txt saved to:\n  {out_dir}")
PY

echo "Done."
