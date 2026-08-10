"""Black-box telemetry logger for unattended overnight runs.

Appends one line per sample (default every 60 s) — GPU util/VRAM/temp/power via
nvidia-smi, host RAM/swap from /proc/meminfo, loadavg — to a log file, fsync'd so
the trailing lines survive a hard host freeze. Standalone by design: no backbone
imports, no CUDA context (nvidia-smi is a query, not a compute client).

Usage:
    python tools/blackbox_logger.py                      # foreground, logs/blackbox.log
    python tools/blackbox_logger.py --once               # single sample to stdout+log
    nohup python tools/blackbox_logger.py >/dev/null 2>&1 &   # overnight companion

Post-mortem after a freeze: `tail logs/blackbox.log` — the last line is the state
of the machine within one interval of death.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024  # rotate to .1 beyond this; one previous file kept

_SMI_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def _gpu_sample() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return f"gpu=ERR({out.returncode})"
        util, used, total, temp, power = [f.strip() for f in out.stdout.splitlines()[0].split(",")]
        return f"gpu_util={util}% vram={used}/{total}MiB temp={temp}C pwr={power}W"
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        # A hung/absent driver is itself a finding — log it rather than die.
        return f"gpu=ERR({type(exc).__name__})"


def _mem_sample() -> str:
    kv = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":", 1)
            kv[key] = int(val.split()[0])  # kB
    ram_used = (kv["MemTotal"] - kv["MemAvailable"]) // 1024
    swap_used = (kv["SwapTotal"] - kv["SwapFree"]) // 1024
    return (
        f"ram={ram_used}/{kv['MemTotal'] // 1024}MiB "
        f"swap={swap_used}/{kv['SwapTotal'] // 1024}MiB"
    )


def _load_sample() -> str:
    with open("/proc/loadavg") as f:
        one, five, fifteen = f.read().split()[:3]
    return f"load={one},{five},{fifteen}"


def sample_line() -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{ts} {_gpu_sample()} {_mem_sample()} {_load_sample()}"


def _rotate(path: Path) -> None:
    if path.exists() and path.stat().st_size > MAX_BYTES:
        path.replace(path.with_suffix(path.suffix + ".1"))


def append(path: Path, line: str) -> None:
    _rotate(path)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between samples")
    parser.add_argument(
        "--log", type=Path,
        default=Path(__file__).resolve().parent.parent / "logs" / "blackbox.log",
    )
    parser.add_argument("--once", action="store_true", help="one sample, print it, exit")
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.once:
        line = sample_line()
        append(args.log, line)
        print(line)
        return

    print(f"blackbox: logging every {args.interval:g}s to {args.log}")
    while True:
        append(args.log, sample_line())
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
