# `models/` — vendored inference weights (not committed)

Binary weights are **gitignored** (`models/*.onnx`, `*.pt`, `*.engine`, `*.trt`).
Fetch them locally; this README is the reproducible record of what goes here.

## Targetless stereo matcher (SuperPoint + LightGlue, ONNX)

The `calibration.feature_extrinsics.OnnxSuperPointLightGlue` matcher (the
**targetless** extrinsic path) needs two **decoupled** ONNX files from
[fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX),
release **v0.1.3** (the decoupled export — SuperPoint extracts keypoints +
descriptors, LightGlue matches them; the fused/end2end exports have a different
I/O contract):

```bash
BASE=https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.3
curl -sL "$BASE/superpoint.onnx"           -o models/superpoint.onnx
curl -sL "$BASE/superpoint_lightglue.onnx" -o models/superpoint_lightglue.onnx
```

I/O contract the matcher relies on (validated against v0.1.3):

- `superpoint.onnx`: in `image (1,1,H,W)` → out `keypoints (1,N,2)`,
  `scores (1,N)`, `descriptors (1,N,256)`. SuperPoint returns *all* detected
  keypoints (~10k on 1080p); the matcher caps to the top `max_keypoints` (1024)
  by score — LightGlue's attention is O(N²) and OOMs otherwise.
- `superpoint_lightglue.onnx`: in `kpts0/kpts1 (1,N,2)`, `desc0/desc1 (1,N,256)`
  → out `matches0/matches1 (1,N)` (**assignment** format: `matches0[i]` = index
  in view B for keypoint `i`, or `-1`), `mscores0/mscores1 (1,N)`. Keypoints must
  be **normalized** to ~[-1,1] before feeding (the decoupled export does not
  normalize internally); the matcher does this.

> Note: feature matching needs **richly-textured natural scenes**. Calibration
> boards (ChArUco/AprilGrid) are repetitive and low-texture — poor matcher input.
> Capture dedicated textured stereo pairs for the targetless method.
