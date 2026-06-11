import csv
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


class EventLogger:
    """Per-event CSV logger with daily rotation and rolling retention.

    Each line-crossing event is appended as one row to
    ``events_YYYY-MM-DD.csv``::

        ts,class,id
        2026-04-23T14:23:45.312847,carton,42
        2026-04-23T14:23:46.118201,polybag,43

    - Thread-safe :meth:`log` via an internal lock; every write is
      flushed before the file handle closes so a crash does not lose
      the latest event.
    - Midnight rollover creates a new day's file on the first write
      past 00:00.
    - Files older than ``retention_days`` are pruned on init and on
      every midnight rollover.

    Replaces the previous :class:`DailyLogger` hourly-snapshot model.
    Event-level storage keeps the chart accurate down to the second
    and survives session restarts without cumulative/delta accounting.

    Args:
        log_dir: Directory for event CSVs. Created if missing.
            Defaults to ``"logs/events"``.
        retention_days: Days of history to keep. Older files are
            deleted. Defaults to 30.
    """

    def __init__(self, log_dir: str = "logs/events", retention_days: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(retention_days)
        self._lock = threading.Lock()
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        self._prune()

    def _filepath(self, date_str: str) -> Path:
        return self.log_dir / f"events_{date_str}.csv"

    def _prune(self) -> None:
        cutoff = datetime.now().date() - timedelta(days=self.retention_days)
        for fp in self.log_dir.glob("events_*.csv"):
            try:
                d = datetime.strptime(fp.stem[len("events_"):], "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff:
                try:
                    fp.unlink()
                    logger.info(f"🗑️  Pruned stale event log: {fp.name}")
                except OSError as exc:
                    logger.warning(f"Could not prune {fp}: {exc}")

    def log(self, class_name: str, event_id: Optional[int] = None) -> None:
        """Append one event row. Safe to call from any thread.

        Args:
            class_name: Class label string (e.g. ``"carton"``).
            event_id: Optional ByteTrack tracker ID. Stored so downstream
                tools can dedupe or trace individual objects.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        with self._lock:
            if date_str != self._current_date:
                self._current_date = date_str
                self._prune()
            fp = self._filepath(date_str)
            new_file = not fp.exists()
            try:
                with open(fp, mode="a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(["ts", "class", "id"])
                    w.writerow([
                        now.isoformat(),
                        str(class_name),
                        "" if event_id is None else int(event_id),
                    ])
            except OSError as exc:
                logger.error(f"⚠️ Failed to write event log: {exc}")

    @classmethod
    def read_events(
        cls,
        log_dir: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> Iterator[Tuple[datetime, str, Optional[int]]]:
        """Yield ``(ts, class_name, event_id_or_none)`` for events in
        ``[from_dt, to_dt)``.

        Reads only the files whose filename date is within the range,
        then filters rows by exact timestamp. Safe to call without
        instantiating an ``EventLogger``; it is a pure read against
        on-disk state and performs no pruning.
        """
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return
        start_date = from_dt.date()
        end_date = to_dt.date()
        day = start_date
        one_day = timedelta(days=1)
        while day <= end_date:
            fp = log_path / f"events_{day.strftime('%Y-%m-%d')}.csv"
            day += one_day
            if not fp.exists():
                continue
            try:
                with open(fp, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # header
                    for row in reader:
                        if len(row) < 2:
                            continue
                        try:
                            ts = datetime.fromisoformat(row[0])
                        except ValueError:
                            continue
                        if ts < from_dt or ts >= to_dt:
                            continue
                        eid: Optional[int] = None
                        if len(row) >= 3 and row[2]:
                            try:
                                eid = int(row[2])
                            except ValueError:
                                eid = None
                        yield ts, row[1], eid
            except OSError:
                continue
