from __future__ import annotations

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.shared.etagere_state import EtagereStateTracker


def _msg(states: list[str], ts: float, conf: float = 0.9) -> EtagereStateMessage:
    cells = tuple(EtagereCellState(r=i // 3 + 1, c=i % 3 + 1, state=s,
                                   confidence=(conf if s != "unknown" else 0.0))
                  for i, s in enumerate(states))
    return EtagereStateMessage(ts=ts, camera_id="cam_a", zone_id="et_1", name="A",
                               cells=cells, seq=int(ts * 10))


def _grid(msg) -> list[str]:
    return [c.state for c in msg.cells]


def test_first_observation_emits_immediately() -> None:
    tr = EtagereStateTracker()
    out = tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    assert out is not None and out.stabilized is True
    assert _grid(out) == ["filled"] * 9 and out.zone_id == "et_1"


def test_held_state_needs_supermajority_to_flip() -> None:
    tr = EtagereStateTracker(window=10, flip_ratio=0.7, heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    # 5 challenger votes in a window of 10 → 50% < 70% → hold
    for k in range(1, 6):
        assert tr.update(_msg(["empty"] * 9, float(k)), now=float(k)) is None
    # 7th challenger vote → 7/10 ≥ 70% → flip, emitted once
    tr.update(_msg(["empty"] * 9, 6.0), now=6.0)
    out = tr.update(_msg(["empty"] * 9, 7.0), now=7.0)
    assert out is not None and _grid(out) == ["empty"] * 9


def test_unknown_does_not_vote_but_decays_after_timeout() -> None:
    tr = EtagereStateTracker(unknown_after_s=5.0, heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    assert tr.update(_msg(["unknown"] * 9, 1.0), now=1.0) is None      # still filled (held)
    out = tr.update(_msg(["unknown"] * 9, 6.0), now=6.0)             # >5 s without a vote
    assert out is not None and _grid(out) == ["unknown"] * 9


def test_heartbeat_re_emits_unchanged_state() -> None:
    tr = EtagereStateTracker(heartbeat_s=5.0)
    tr.update(_msg(["empty"] * 9, 0.0), now=0.0)
    assert tr.update(_msg(["empty"] * 9, 1.0), now=1.0) is None
    out = tr.update(_msg(["empty"] * 9, 5.5), now=5.5)
    assert out is not None and _grid(out) == ["empty"] * 9


def test_per_cell_independence_and_confidence_carried() -> None:
    tr = EtagereStateTracker(heartbeat_s=1e9)
    states = ["filled"] * 9
    tr.update(_msg(states, 0.0, conf=0.8), now=0.0)
    states[4] = "empty"
    out = None
    for k in range(1, 15):
        out = tr.update(_msg(states, float(k), conf=0.8), now=float(k)) or out
    assert out is not None
    g = _grid(out)
    assert g[4] == "empty" and g[0] == "filled" and g[8] == "filled"
    assert out.cells[4].confidence == 0.8


def test_forget_zone_resets_history() -> None:
    tr = EtagereStateTracker(heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    tr.forget_zone("et_1")
    out = tr.update(_msg(["empty"] * 9, 1.0), now=1.0)
    assert out is not None and _grid(out) == ["empty"] * 9
