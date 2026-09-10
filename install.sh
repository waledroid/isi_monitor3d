#!/usr/bin/env bash
# install.sh — stage-by-stage installer for ISI Monitor 3D.
#
#   ./install.sh [gpu|cpu] [--dry-run] [--skip STAGE]... [--only STAGE] [--systemd] [--list]
#
# gpu = branch main (two cameras, ONNX Runtime CUDA + TensorRT engines)
# cpu = branch cpu  (one camera, OpenVINO on CPU)
# The variant defaults to the checked-out branch (cpu → cpu, anything else → gpu).
# ENV_NAME=<name> overrides the conda env name (default monitor3d / monitor3d-cpu).
#
# Every stage has a PROBE that inspects the real machine (env importable,
# engines present, broker answering, alias in .bashrc ...). A stage whose
# probe passes is skipped as "done", so the script resumes after a failure
# and recognises steps done by hand. Stages that need a person (models to
# copy, camera URLs, calibration) report what is missing and are marked
# NEEDS YOU; the rest continues.
#
# Exit code: 0 = every automatic stage done, 1 = a stage failed.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE="${MINIFORGE:-$HOME/miniforge3}"
ORT_GPU_VERSION="1.23.2"
TRT_VERSION="10.16.1.11"
COMPOSE_FILE="isicomms/deploy/onprem/docker-compose.yml"
CPU_TRACKED_PREFIX="/home/aatanda/isi_monitor3d_cpu"

VARIANT="${VARIANT:-}"
DRY="${DRY:-0}"
SKIP_LIST=()
ONLY_STAGE=""
WANT_SYSTEMD=0
LIST_ONLY=0

# ---------- output helpers ----------
if [ -t 1 ]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_HEAD=$'\033[1;36m'
else
  C_RESET=""; C_DIM=""; C_OK=""; C_WARN=""; C_ERR=""; C_HEAD=""
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '   %s✔ %s%s\n' "$C_OK" "$*" "$C_RESET"; }
warn() { printf '   %s! %s%s\n' "$C_WARN" "$*" "$C_RESET"; }
err()  { printf '   %s✘ %s%s\n' "$C_ERR" "$*" "$C_RESET"; }
info() { printf '   %s%s%s\n' "$C_DIM" "$*" "$C_RESET"; }

render_bar() {                       # render_bar CURRENT TOTAL → "[####------] c/t"
  local cur=$1 tot=$2 width=30 filled
  filled=$(( cur * width / tot ))
  printf '[%s%s] %2d/%d' "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
                         "$(printf '%*s' $((width - filled)) '' | tr ' ' '-')" "$cur" "$tot"
}

# In dry-run mode: print what would happen and tell the caller to return.
if_dry() { if [ "$DRY" = 1 ]; then info "would: $*"; return 0; fi; return 1; }

variant_from_branch() { case "$1" in cpu|cpu/*) echo cpu ;; *) echo gpu ;; esac; }

# ---------- environment ----------
env_name() { if [ -n "${ENV_NAME:-}" ]; then echo "$ENV_NAME"; elif [ "$VARIANT" = cpu ]; then echo monitor3d-cpu; else echo monitor3d; fi; }
env_py()   { echo "$MINIFORGE/envs/$(env_name)/bin/python"; }
conda_bin() {
  if [ -x "$MINIFORGE/bin/conda" ]; then echo "$MINIFORGE/bin/conda"
  elif command -v conda >/dev/null 2>&1; then command -v conda
  else return 1; fi
}
alias_name() { [ "$VARIANT" = cpu ] && echo 3d_cpu || echo 3d; }
alias_line() {
  local port=""; [ "$VARIANT" = cpu ] && port="MONITOR_WEB_PORT=8200 "
  echo "alias $(alias_name)='cd $REPO && conda activate $(env_name) && ${port}python -m monitor_web'"
}
yaml_get() {                          # yaml_get 'expr' → python expression on the loaded backbone.yaml
  "$(env_py)" - "$REPO/config/backbone.yaml" "$1" <<'PY' 2>/dev/null
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
try:
    v = eval(sys.argv[2], {}, {"c": c})
except Exception:
    v = ""
print("" if v is None else v)
PY
}
backbone_running() { pgrep -f "python -m backbone.runtime" >/dev/null 2>&1; }

# ---------- stage tables ----------
select_stages() {
  VARIANT="$1"
  if [ "$VARIANT" = cpu ]; then
    STAGE_IDS=(prereq miniforge env dashboard paths models comms config alias verify)
    STAGE_TITLES=("Host prerequisites" "Miniforge (conda)" "Conda env monitor3d-cpu" "Dashboard + gateway packages"
                  "Absolute paths in backbone.yaml" "OpenVINO models" "Comms stack (gateway + Mosquitto)"
                  "Site configuration" "Shell alias 3d_cpu" "Test suite")
  else
    STAGE_IDS=(prereq miniforge env ortswap multical dashboard models engines comms config alias verify)
    STAGE_TITLES=("Host prerequisites" "Miniforge (conda)" "Conda env monitor3d" "ONNX Runtime GPU + TensorRT"
                  "Multical calibration venv" "Dashboard + gateway packages" "Model weights" "TensorRT engines"
                  "Comms stack (gateway + Mosquitto)" "Site configuration" "Shell alias 3d" "Test suite")
  fi
  if [ "$WANT_SYSTEMD" = 1 ]; then STAGE_IDS+=(systemd); STAGE_TITLES+=("systemd units"); fi
}

should_skip() {
  local id=$1 s
  if [ -n "$ONLY_STAGE" ] && [ "$ONLY_STAGE" != "$id" ]; then return 0; fi
  for s in ${SKIP_LIST[@]+"${SKIP_LIST[@]}"}; do [ "$s" = "$id" ] && return 0; done
  return 1
}

# ---------- stage: prereq ----------
probe_prereq() {
  command -v git >/dev/null && command -v curl >/dev/null || return 1
  if [ "$VARIANT" = gpu ]; then nvidia-smi >/dev/null 2>&1 || return 1; fi
  docker compose version >/dev/null 2>&1 || return 1
}
run_prereq() {
  local missing=()
  command -v git  >/dev/null || missing+=("git   → sudo apt install -y git")
  command -v curl >/dev/null || missing+=("curl  → sudo apt install -y curl")
  if [ "$VARIANT" = gpu ] && ! nvidia-smi >/dev/null 2>&1; then
    missing+=("NVIDIA driver (nvidia-smi fails) → install the driver on the host / Windows side")
  fi
  docker compose version >/dev/null 2>&1 || missing+=("docker compose v2 → Docker Engine or Docker Desktop with WSL integration")
  if [ "$VARIANT" = gpu ] && ! command -v cmake >/dev/null; then
    warn "cmake/libopencv-dev/libeigen3-dev missing: only needed for AprilGrid calibration (sudo apt install -y cmake libopencv-dev libeigen3-dev)"
  fi
  if [ ${#missing[@]} -eq 0 ]; then return 0; fi
  for m in "${missing[@]}"; do err "$m"; done
  return 2
}

# ---------- stage: miniforge ----------
probe_miniforge() { conda_bin >/dev/null 2>&1; }
run_miniforge() {
  if_dry "download Miniforge3 and install to $MINIFORGE, then conda init bash" && return 0
  local url=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  local tmp; tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/Miniforge3.sh" "$url" || { err "download failed: $url"; return 1; }
  bash "$tmp/Miniforge3.sh" -b -p "$MINIFORGE" || { err "installer failed"; return 1; }
  "$MINIFORGE/bin/conda" init bash >/dev/null
  rm -rf "$tmp"
  info "open a new shell afterwards so 'conda activate' works"
}

# ---------- stage: env ----------
# Probes import from a neutral cwd: from the repo root a bare "import monitor_web"
# would resolve the source directory itself and report an uninstalled package as done.
py_probe() { ( cd / && "$(env_py)" -c "$1" ) >/dev/null 2>&1; }
probe_env() { py_probe "import backbone, isistream, calibration"; }
run_env() {
  local cb; cb="$(conda_bin)" || { err "conda not found (run the miniforge stage)"; return 1; }
  if [ -d "$MINIFORGE/envs/$(env_name)" ]; then
    if_dry "conda env update -f environment.yml -n $(env_name) --prune" && return 0
    "$cb" env update -f "$REPO/environment.yml" -n "$(env_name)" --prune || return 1
  else
    if_dry "conda env create -f environment.yml -n $(env_name)" && return 0
    "$cb" env create -f "$REPO/environment.yml" -n "$(env_name)" || return 1
  fi
  probe_env
}

# ---------- stage: ortswap (gpu) ----------
# environment.yml now installs onnxruntime-gpu + tensorrt through its pip
# section, so on a clean env this stage is a no-op; it stays as the repair path
# for envs created from an older environment.yml (conda onnxruntime present).
probe_ortswap() { py_probe "import onnxruntime as ort, tensorrt; assert 'CUDAExecutionProvider' in ort.get_available_providers()"; }
run_ortswap() {
  if_dry "conda remove --force onnxruntime; pip install onnxruntime-gpu==$ORT_GPU_VERSION tensorrt-cu12==$TRT_VERSION" && return 0
  local cb; cb="$(conda_bin)" || return 1
  "$cb" remove -n "$(env_name)" --force -y onnxruntime >/dev/null 2>&1 || true
  # --force-reinstall: removing the conda onnxruntime deletes files the pip
  # wheel shares (onnxruntime/capi/...), so a wheel that pip considers
  # "already satisfied" can still be broken. Reinstall it unconditionally.
  "$(env_py)" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==$ORT_GPU_VERSION" || return 1
  "$(env_py)" -m pip install "onnxruntime-gpu==$ORT_GPU_VERSION" "tensorrt-cu12==$TRT_VERSION" || return 1
  rm -f "$MINIFORGE/envs/$(env_name)"/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_* 2>/dev/null
  probe_ortswap
}

# ---------- stage: multical (gpu) ----------
probe_multical() { [ -x "$REPO/calibration/.venv-multical/bin/multical" ]; }
run_multical() {
  if_dry "bash calibration/setup_multical.sh (MULTICAL_VIEWER=${MULTICAL_VIEWER:-0})" && return 0
  ( cd "$REPO" && MULTICAL_VIEWER="${MULTICAL_VIEWER:-0}" PYTHON="$(env_py)" bash calibration/setup_multical.sh ) || return 1
  probe_multical
}

# ---------- stage: dashboard ----------
probe_dashboard() { py_probe "import monitor_web, isicomms"; }
run_dashboard() {
  if [ "$VARIANT" = cpu ]; then
    if_dry "pip install --no-deps -e monitor_web -e isicomms" && return 0
    "$(env_py)" -m pip install --no-deps -e "$REPO/monitor_web" -e "$REPO/isicomms" || return 1
  else
    if_dry "pip install -e monitor_web[dev] -e isicomms[dev]" && return 0
    "$(env_py)" -m pip install -e "$REPO/monitor_web[dev]" -e "$REPO/isicomms[dev]" || return 1
  fi
  probe_dashboard
}

# ---------- stage: paths (cpu) ----------
probe_paths() {
  [ "$REPO" = "$CPU_TRACKED_PREFIX" ] && return 0
  ! grep -q "$CPU_TRACKED_PREFIX" "$REPO/config/backbone.yaml"
}
run_paths() {
  if_dry "sed -i 's#$CPU_TRACKED_PREFIX#$REPO#g' config/backbone.yaml" && return 0
  sed -i "s#$CPU_TRACKED_PREFIX#$REPO#g" "$REPO/config/backbone.yaml" && probe_paths
}

# ---------- stage: models ----------
model_paths() {                        # prints the configured model files, one per line
  if [ "$VARIANT" = cpu ]; then
    yaml_get 'c["detection"].get("model_xml","")'; yaml_get 'c["detection"].get("pose_model_xml","")'
  else
    yaml_get 'c["detection"].get("onnx_path","")';  yaml_get 'c["detection"].get("pose_onnx_path","")'
  fi
}
probe_models() {
  local p any=0
  while IFS= read -r p; do
    [ -z "$p" ] && continue; any=1
    case "$p" in /*) ;; *) p="$REPO/$p" ;; esac
    if [ "$VARIANT" = gpu ]; then
      # a .engine path counts if the engine OR its source .onnx exists
      [ -e "$p" ] || [ -e "${p%.engine}.onnx" ] || return 1
    else
      [ -e "$p" ] || return 1
    fi
  done < <(model_paths)
  [ "$any" = 1 ]
}
run_models() {
  local p
  while IFS= read -r p; do
    [ -z "$p" ] && { err "detection model path not set in config/backbone.yaml"; continue; }
    case "$p" in /*) ;; *) p="$REPO/$p" ;; esac
    if [ -e "$p" ] || { [ "$VARIANT" = gpu ] && [ -e "${p%.engine}.onnx" ]; }; then ok "$p"; else err "missing: $p  (copy the file here, or point config/backbone.yaml at it)"; fi
  done < <(model_paths)
  return 2
}

# ---------- stage: engines (gpu) ----------
probe_engines() {
  # Done when every configured model resolves: a built .engine (with its
  # provenance sidecar) or an .onnx that ONNX Runtime will run directly.
  local p
  while IFS= read -r p; do
    [ -z "$p" ] && return 1
    case "$p" in /*) ;; *) p="$REPO/$p" ;; esac
    if [[ "$p" == *.engine ]]; then [ -e "$p" ] && [ -e "$p.json" ] || return 1
    else [ -e "$p" ] || return 1; fi
  done < <(model_paths)
}
run_engines() {
  if backbone_running; then err "the system is running (backbone.runtime); STOP it before building engines"; return 2; fi
  local det pose rc=0
  det="$(yaml_get 'c["detection"].get("onnx_path","")')"; pose="$(yaml_get 'c["detection"].get("pose_onnx_path","")')"
  build_one() {                        # build_one PATH IMGSZ EXTRA...
    local p=$1 imgsz=$2; shift 2
    case "$p" in /*) ;; *) p="$REPO/$p" ;; esac
    local onnx="${p%.engine}"; onnx="${onnx%.onnx}.onnx"; local engine="${onnx%.onnx}.engine"
    [ -e "$onnx" ] || { err "no ONNX source for $p"; return 1; }
    if [ -e "$engine" ] && [ -e "$engine.json" ]; then ok "engine present: $engine"; return 0; fi
    if [[ "$p" == *.onnx ]] && [ -e "$p" ]; then info "$p runs on ONNX Runtime; build an engine for ~2x: tools/onnx_to_engine.py $onnx --imgsz $imgsz $*"; return 0; fi
    if_dry "python tools/onnx_to_engine.py $onnx --imgsz $imgsz $*" && return 0
    ( cd "$REPO" && "$(env_py)" tools/onnx_to_engine.py "$onnx" --imgsz "$imgsz" "$@" ) || return 1
    [[ "$p" == *.onnx ]] && warn "set config/backbone.yaml to $engine (the .engine is what the runtime should load)"
    return 0
  }
  [ -n "$det" ]  && { build_one "$det" 320 --min-batch 1 --opt-batch 8 --max-batch 32 || rc=1; }
  [ -n "$pose" ] && { build_one "$pose" 640 --max-batch 4 || rc=1; }
  [ "$rc" = 0 ] || return 1
  [ "$DRY" = 1 ] && return 0
  probe_engines || return 2
}

# ---------- stage: comms ----------
probe_comms() { curl -sf -m 3 http://127.0.0.1:8080/healthz >/dev/null 2>&1; }
run_comms() {
  docker compose version >/dev/null 2>&1 || { err "docker compose not available"; return 2; }
  if ss -ltn 2>/dev/null | grep -q ':1883 ' && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q mosquitto; then
    err "port 1883 is held by a non-docker broker (sudo systemctl disable --now mosquitto), or another stack"; return 2
  fi
  if_dry "docker compose -p on-prem -f $COMPOSE_FILE up -d --build" && return 0
  ( cd "$REPO" && docker compose -p on-prem -f "$COMPOSE_FILE" up -d --build ) || return 1
  local i; for i in $(seq 1 30); do probe_comms && return 0; sleep 1; done
  err "gateway did not answer on :8080 within 30 s"; return 1
}

# ---------- stage: config ----------
config_problems() {                    # prints one problem per line
  local node cal urls
  node="$(yaml_get 'c.get("node_id","")')"
  [ -n "$node" ] || echo "node_id is empty (must be unique per PC)"
  cal="$(yaml_get 'c.get("calibration_path","")')"
  if [ -z "$cal" ]; then echo "calibration_path not set"
  else case "$cal" in /*) ;; *) cal="$REPO/$cal" ;; esac; [ -e "$cal" ] || echo "calibration file missing: $cal (isical Export / calibrate single-cam)"; fi
  urls="$(yaml_get '" ".join(str(v.get("source",{}).get("url","")) for v in c.get("cameras",{}).values()) if isinstance(c.get("cameras"),dict) else ""')"
  [ -n "$urls" ] || echo "no camera source URLs in cameras:"
  echo "$urls" | grep -q "192.168.1.10/Streaming\|192.168.1.11/Streaming" && echo "cameras still use the example URLs"
  [ "$(yaml_get 'c["ingestion"].get("mode","")')" = points ] || echo "ingestion.mode must be points"
  return 0
}
probe_config() { [ -z "$(config_problems)" ]; }
run_config() { local l; while IFS= read -r l; do [ -n "$l" ] && err "$l"; done < <(config_problems); info "edit config/backbone.yaml (or use the dashboard Settings), then re-run"; return 2; }

# ---------- stage: alias ----------
probe_alias() { grep -q "^alias $(alias_name)=" "$HOME/.bashrc" 2>/dev/null; }
run_alias() {
  probe_alias && return 0                # idempotent: never append twice
  if_dry "append to ~/.bashrc: $(alias_line)" && return 0
  printf '\n# ISI Monitor 3D launcher (install.sh)\n%s\n' "$(alias_line)" >> "$HOME/.bashrc"
  probe_alias
}

# ---------- stage: verify ----------
probe_verify() { return 1; }           # always runs (it IS the check)
run_verify() {
  if_dry "pytest -q tests calibration/tests in $(env_name)" && return 0
  ( cd "$REPO" && "$(env_py)" -m pytest -q -p no:cacheprovider tests calibration/tests 2>&1 | tail -1 )
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
}

# ---------- stage: systemd (opt-in) ----------
probe_systemd() { [ -e /etc/systemd/system/isi-backbone.service ] && [ -e /etc/systemd/system/isistream.service ]; }
run_systemd() {
  if_dry "write /etc/systemd/system/isi-backbone.service + isistream.service (sudo) and enable them" && return 0
  local py; py="$(env_py)"; local envline=""
  [ "$VARIANT" = cpu ] && envline="Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
  sudo tee /etc/systemd/system/isi-backbone.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D metric engine ($VARIANT)
After=network-online.target docker.service
[Service]
User=$USER
WorkingDirectory=$REPO
$envline
ExecStart=$py -m backbone.runtime --config $REPO/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
  sudo tee /etc/systemd/system/isistream.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D perception producer ($VARIANT)
After=isi-backbone.service
[Service]
User=$USER
WorkingDirectory=$REPO
$envline
ExecStart=$py -m isistream --config $REPO/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload && sudo systemctl enable isi-backbone isistream >/dev/null && probe_systemd
  warn "units enabled but not started; 'sudo systemctl start isi-backbone isistream' when the dashboard is not running"
}

# ---------- main ----------
usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

main() {
  local a
  while [ $# -gt 0 ]; do
    case "$1" in
      gpu|cpu) VARIANT=$1 ;;
      --dry-run) DRY=1 ;;
      --skip) shift; SKIP_LIST+=("$1") ;;
      --only) shift; ONLY_STAGE=$1 ;;
      --systemd) WANT_SYSTEMD=1 ;;
      --list) LIST_ONLY=1 ;;
      -h|--help) usage; return 0 ;;
      *) err "unknown argument: $1"; usage; return 1 ;;
    esac; shift
  done
  if [ -z "$VARIANT" ]; then
    VARIANT="$(variant_from_branch "$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)")"
  fi
  select_stages "$VARIANT"
  if [ "$LIST_ONLY" = 1 ]; then
    local i; for i in "${!STAGE_IDS[@]}"; do printf '  %-10s %s\n' "${STAGE_IDS[$i]}" "${STAGE_TITLES[$i]}"; done; return 0
  fi

  say "${C_HEAD}ISI Monitor 3D installer — variant: $VARIANT — repo: $REPO${C_RESET}"
  [ "$DRY" = 1 ] && say "${C_DIM}dry run: nothing will be changed${C_RESET}"
  local total=${#STAGE_IDS[@]} i id title status note rc
  local -a RESULTS=() NOTES=()
  local failed=0
  for i in "${!STAGE_IDS[@]}"; do
    id=${STAGE_IDS[$i]}; title=${STAGE_TITLES[$i]}; note=""
    printf '\n%s %s%s%s\n' "$(render_bar $((i+1)) "$total")" "$C_HEAD" "$title" "$C_RESET"
    if should_skip "$id"; then status=SKIPPED; info "skipped (--skip/--only)"
    elif "probe_$id"; then status=DONE; ok "already done"
    else
      "run_$id"; rc=$?
      case $rc in
        0) status=DONE; [ "$DRY" = 1 ] && status=PENDING; [ "$DRY" = 1 ] || ok "done" ;;
        2) status="NEEDS YOU"; warn "needs a manual step (see above)" ;;
        *) status=FAILED; failed=1; err "failed (rc=$rc)" ;;
      esac
    fi
    RESULTS+=("$status")
  done

  printf '\n%s%s%s\n' "$C_HEAD" "Summary" "$C_RESET"
  for i in "${!STAGE_IDS[@]}"; do
    case "${RESULTS[$i]}" in
      DONE) c=$C_OK ;; "NEEDS YOU") c=$C_WARN ;; FAILED) c=$C_ERR ;; *) c=$C_DIM ;;
    esac
    printf '  %-10s %s%-9s%s %s\n' "${STAGE_IDS[$i]}" "$c" "${RESULTS[$i]}" "$C_RESET" "${STAGE_TITLES[$i]}"
  done
  if printf '%s\n' "${RESULTS[@]}" | grep -q "NEEDS YOU"; then
    say; say "Finish the NEEDS YOU items, then run ./install.sh $VARIANT again; done stages are skipped."
  fi
  say; say "Start: open a new shell, run '$(alias_name)' from $REPO, press START in the dashboard."
  [ "$failed" = 0 ]
}

if [ "${INSTALL_SH_LIB:-0}" = 1 ]; then
  select_stages "${VARIANT:-gpu}"
  return 0 2>/dev/null || true
else
  main "$@"
fi
