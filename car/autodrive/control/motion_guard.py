"""Local step deadlines and panel cancellation independent of inference."""
import json
import math
from pathlib import Path
import threading
import time


class MotionGuard:
    def __init__(self, driver, session_directory=None):
        self.driver=driver
        self.directory=None if session_directory is None else Path(session_directory)
        self.cancelled=threading.Event();self._closed=threading.Event();self.reason=''
        self._thread=threading.Thread(target=self._run,name='motion-step-guard',daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._closed.wait(.01):
                if self.directory is not None:
                    if (self.directory/'stop.json').exists():
                        self.reason='operator ended this session'
                    else:
                        try:
                            with (self.directory/'lease.json').open() as stream:
                                lease=json.loads(stream.read(4096))
                            expiry=lease.get('expires_monotonic')
                            now=time.monotonic()
                            if (type(expiry) not in (int,float) or not math.isfinite(expiry)
                                    or not now<=expiry<=now+4.):
                                self.reason='panel control lease expired'
                        except (OSError,ValueError,TypeError):self.reason='panel control lease unavailable'
                    if self.reason:
                        self.driver.inhibit(self.reason);self.cancelled.set();return
                self.driver.check_motion_deadline()
        except Exception as exc:
            self.reason='motion guard failed: '+type(exc).__name__
            self.cancelled.set()
            self.driver.inhibit(self.reason)

    def close(self):
        self._closed.set();self._thread.join(timeout=1)

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
