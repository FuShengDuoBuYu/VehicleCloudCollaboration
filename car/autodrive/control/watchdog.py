"""Independent control-heartbeat watchdog."""
from __future__ import annotations

from autodrive.control.wheel_driver import SafeWheelDriver
from typing import Optional
import threading
import time


class CommandWatchdog:
    """Stop the vehicle when fresh perception commands stop arriving."""

    def __init__(
        self,
        wheel_driver: SafeWheelDriver,
        timeout: float = 2.5,
        check_interval: float = 0.05,
    ):
        if timeout <= 0 or check_interval <= 0:
            raise ValueError("watchdog timings must be positive")
        self.wheel_driver = wheel_driver
        self.timeout = float(timeout)
        self.check_interval = float(check_interval)
        self._lock = threading.Lock()
        self._last_heartbeat: Optional[float] = None
        self._armed = False
        self._tripped = False
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def arm(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._armed = True
            self._tripped = False

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._tripped = False

    def disarm(self, stop: bool = True) -> None:
        with self._lock:
            self._armed = False
            self._last_heartbeat = None
        if stop:
            self.wheel_driver.stop("watchdog disarmed")

    def get_state(self) -> dict:
        with self._lock:
            age = (
                None
                if self._last_heartbeat is None
                else time.monotonic() - self._last_heartbeat
            )
            return {
                "armed": self._armed,
                "tripped": self._tripped,
                "heartbeat_age": age,
                "timeout": self.timeout,
            }

    def _run(self) -> None:
        while not self._closed.wait(self.check_interval):
            should_stop = False
            with self._lock:
                if (
                    self._armed
                    and not self._tripped
                    and self._last_heartbeat is not None
                    and time.monotonic() - self._last_heartbeat > self.timeout
                ):
                    self._tripped = True
                    should_stop = True
            if should_stop:
                self.wheel_driver.stop("perception watchdog timeout")

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=max(1.0, self.check_interval * 3))
        self.disarm(stop=True)

