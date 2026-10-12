"""Non-blocking queue and atomic status helpers."""
from __future__ import annotations

import json
import os
import queue


def _offer_latest(work_queue, item):
    """Enqueue without blocking, replacing the oldest queued diagnostic.

    Diagnostics must never delay a motor update.  When storage falls behind,
    retaining the newest observation is also more useful than allowing an old
    backlog to hide the vehicle's current state.
    """
    try:
        work_queue.put_nowait(item)
        return True, 0
    except queue.Full:
        pass

    try:
        work_queue.get_nowait()
        work_queue.task_done()
    except queue.Empty:
        # The consumer won the race after ``Full``; retrying below is safe.
        pass
    try:
        work_queue.put_nowait(item)
        return True, 1
    except queue.Full:
        # A consumer cannot cause this in the single-producer design, but do
        # not turn a diagnostic race back into a control-loop wait.
        return False, 1


def write_status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)

