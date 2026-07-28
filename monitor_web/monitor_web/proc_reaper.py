"""Instance-scoped process identity + safe stray reaping (shared).

Every dashboard instance stamps the children it spawns (the Backbone and the
isistream producer) with ``ISI3D_INSTANCE_ID`` in their environment and puts
each child in its own session (``start_new_session=True``). That buys two
guarantees the old cmdline-substring reapers could not give:

1. **Sibling safety.** The operator's dashboard (:8000) and a dev instance
   (:8100) coexist on one host; each reaper now kills only processes carrying
   *its own* instance id (read from ``/proc/<pid>/environ`` — the exec-time
   snapshot, readable for same-UID processes). A stray marked by another
   instance is left alone even when orphaned: its owner adopts it at its next
   boot (the id is port-derived, hence stable across crashes).
2. **Whole-tree stops.** Children are session leaders, so STOP escalates via
   ``os.killpg`` and takes grandchildren (a stuck ``ffprobe``) with them.

Legacy arm: processes spawned by pre-identity code carry no marker. They are
reaped only when *genuinely orphaned* (``ppid == 1``) — a markerless child
whose parent is alive belongs to someone and is never touched.

``ISI3D_DISABLE_REAP=1`` disables the kill paths only (the finders stay
read-only-safe); the test suite exports it as a belt on top of patching the
finder, so ``pytest`` can never again SIGKILL a live production process
(observed 2026-07-06 with the Backbone; the same hole existed for the
producer until this module unified the two reapers).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)

MARKER_ENV = "ISI3D_INSTANCE_ID"
DISABLE_ENV = "ISI3D_DISABLE_REAP"

# cmdline tokens, matched against the NUL-separated /proc/<pid>/cmdline bytes.
BACKBONE_TOKEN = b"backbone.runtime"          # unambiguous as a bare substring
ISISTREAM_TOKEN = b"-m\x00isistream\x00"      # NUL-aware: argv ["-m","isistream",...]


def reap_disabled() -> bool:
    """True when the kill switch is set — kill paths become no-ops."""
    return os.environ.get(DISABLE_ENV, "") not in ("", "0")


def fallback_instance_id() -> str:
    """Identity for hosts constructed without an explicit id (tests, scripts).

    PID-qualified so a bare-constructed object can never match — and therefore
    never kill — anything it did not spawn itself.
    """
    return os.environ.get(MARKER_ENV) or f"monitor-web:pid{os.getpid()}"


# ---- read-only /proc probes -------------------------------------------------

def _read_cmdline(pid: int) -> bytes | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _read_environ(pid: int) -> bytes | None:
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return fh.read()
    except OSError:
        return None      # vanished, or not ours — caller must NOT kill


def _ppid(pid: int) -> int | None:
    """Parent pid from /proc/<pid>/stat (field 4, after the ')' of comm)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read()
        return int(stat[stat.rindex(b") ") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _same_uid(pid: int) -> bool:
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def _is_mine(pid: int, token: bytes, instance_id: str) -> bool:
    """The reap rule. True only for processes this instance may kill."""
    cmd = _read_cmdline(pid)
    if not cmd or token not in cmd:
        return False                      # zombies have empty cmdline → excluded
    if not _same_uid(pid):
        return False
    env = _read_environ(pid)
    if env is None:
        return False                      # unreadable → never kill
    marker = f"{MARKER_ENV}=".encode()
    if marker in env:
        return f"{MARKER_ENV}={instance_id}".encode() + b"\x00" in env + b"\x00"
    # Legacy (pre-identity) process: only when genuinely orphaned. A markerless
    # child with a live parent belongs to someone — hands off.
    return _ppid(pid) == 1


def find_strays(token: bytes, instance_id: str, *,
                exclude: set[int] | None = None) -> list[int]:
    """Scan /proc for processes this instance owns (or legacy orphans).

    Read-only by design — no disable gate here, so tests can exercise the
    matcher against live decoys without arming anything.
    """
    skip = {os.getpid()}
    if exclude:
        skip |= set(exclude)
    found: list[int] = []
    try:
        names = os.listdir("/proc")
    except OSError as exc:               # not Linux / no procfs
        logger.debug("proc_reaper: /proc scan unavailable: %s", exc)
        return found
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in skip:
            continue
        if _is_mine(pid, token, instance_id):
            found.append(pid)
    return found


def kill_stray(pid: int, sig: int = signal.SIGKILL) -> bool:
    """Signal one stray; the whole group when the pid leads its own group.

    Our own spawns are session leaders (``start_new_session=True``), so a
    group kill takes their grandchildren too. A legacy stray that is NOT a
    leader gets a plain kill — we never signal a group we don't own.
    """
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False                     # won the race; already dead
    except PermissionError:
        logger.warning("proc_reaper: cannot signal pid %s (not ours)", pid)
        return False


def kill_strays(token: bytes, instance_id: str, *, why: str,
                on_kill=None, exclude: set[int] | None = None) -> int:
    """SIGKILL every stray matching the rule; returns how many were signalled.

    Honors the ``ISI3D_DISABLE_REAP`` kill switch. ``on_kill(pid)`` lets the
    caller mirror each kill into its own log surface (ring buffer, …).
    """
    if reap_disabled():
        logger.info("proc_reaper: %s skipped (%s set)", why, DISABLE_ENV)
        return 0
    killed = 0
    for pid in find_strays(token, instance_id, exclude=exclude):
        if not kill_stray(pid):
            continue
        killed += 1
        logger.warning("proc_reaper: %s pid %s", why, pid)
        if on_kill is not None:
            try:
                on_kill(pid)
            except Exception:            # a log hook must never abort the sweep
                logger.debug("proc_reaper: on_kill hook failed", exc_info=True)
    return killed


def terminate_tree(proc: subprocess.Popen, *, term_grace_s: float) -> str:
    """Stop a tracked child AND its whole process group. Returns the method
    that ended the direct child: ``"sigterm"`` | ``"sigkill"`` | ``"gone"``.

    Requires the child to have been spawned with ``start_new_session=True``
    (child pid == its pgid); otherwise falls back to the classic
    terminate/wait/kill on the single pid — never killpg a shared group.
    The final group SIGKILL sweep is what catches a grandchild (e.g. a stuck
    ``ffprobe``) that outlived the child itself.
    """
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        proc.poll()
        return "gone"
    own_group = pgid == pid and pgid != os.getpgid(0)
    method = "sigterm"
    try:
        if own_group:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=term_grace_s)
        except subprocess.TimeoutExpired:
            method = "sigkill"
            if own_group:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=2.0)
    except ProcessLookupError:
        method = "gone"
        proc.poll()
    if own_group:
        # Grandchildren may have survived the child (different signal timing,
        # or the child died before forwarding anything). One final sweep.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Give the kernel a beat to reparent/reap before callers re-scan /proc.
        time.sleep(0.05)
    return method
