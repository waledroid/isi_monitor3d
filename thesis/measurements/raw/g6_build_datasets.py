#!/usr/bin/env python3
"""G6 dataset builder (as executed 2026-07-20). Builds test_real / arm_S / arm_R / arm_RS
under thesis/measurements/G6_ablation from pallet3_yolo_seg (real, class 2=polybag)
and the isiGen black_polybag yolo_seg export (synthetic syn*.png only)."""
import random, shutil
from pathlib import Path

P = Path("/home/aatanda/isi_monitor3d/trainer/isidet/data/pallet3_yolo_seg")
E = Path("/home/aatanda/isi_monitor3d/trainer/isiGen/data/black_polybag/export/yolo_seg")
G6 = Path("/home/aatanda/isi_monitor3d/thesis/measurements/G6_ablation")

def filt_label(src):
    out = []
    for ln in src.read_text().splitlines():
        p = ln.split()
        if p and p[0] == "2":
            out.append(" ".join(["0"] + p[1:]))
    return "\n".join(out) + "\n"

def put(ds, split, img, label_text):
    (G6/ds/"images"/split).mkdir(parents=True, exist_ok=True)
    (G6/ds/"labels"/split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, G6/ds/"images"/split/img.name)
    (G6/ds/"labels"/split/(img.stem + ".txt")).write_text(label_text)

for lf in sorted((P/"labels"/"val").glob("colis-*.txt")):
    txt = filt_label(lf)
    if txt.strip():
        put("test_real", "val", P/"images"/"val"/(lf.stem + ".jpg"), txt)

pool = [lf for lf in sorted((P/"labels"/"train").glob("colis-*.txt")) if filt_label(lf).strip()]
random.Random(42).shuffle(pool)
r_train, r_val = pool[:238], pool[238:238+29]
for ds, split, items in [("arm_R","train",r_train),("arm_R","val",r_val),
                          ("arm_RS","train",r_train),("arm_RS","val",r_val)]:
    for lf in items:
        put(ds, split, P/"images"/"train"/(lf.stem + ".jpg"), filt_label(lf))

for split in ("train","val"):
    for img in sorted((E/"images"/split).glob("syn*")):
        txt = (E/"labels"/split/(img.stem + ".txt")).read_text()
        put("arm_S", split, img, txt)
        put("arm_RS", split, img, txt)

for ds in ("arm_S","arm_R","arm_RS","test_real"):
    (G6/ds/"data.yaml").write_text(
        f"path: {G6/ds}\ntrain: images/{'train' if ds!='test_real' else 'val'}\nval: images/val\nnc: 1\nnames: ['polybag']\n")
