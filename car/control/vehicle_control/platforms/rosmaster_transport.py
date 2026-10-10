"""Bounded, exclusive Rosmaster UART transport without constructor commands.

Wire format and scales are audited against local Rosmaster_Lib 3.3.1.
A completed serial write is not an actuator acknowledgement. Call poll()
explicitly to receive feedback; telemetry() never refreshes old samples.
"""
import fcntl
import struct
import termios
import threading
import time

from ..base import checked_integer, checked_number


class RosmasterCommunicationError(OSError):
    """A device open, read, or complete-frame write failed."""


_GROUPS = {0x0a: ("motion", 7), 0x0b: ("imu", 18),
           0x0c: ("attitude", 6), 0x0d: ("encoder", 16),
           0x0e: ("imu", 18), 0x51: ("version", 2),
           0x15: ("car_type", (1, 2))}


def _open_serial(**options):
    import serial
    port = options.pop("port")
    device = serial.Serial(port=None, **options)
    device.dtr = False
    device.rts = False
    device.port = port
    try:
        device.open()
        fcntl.ioctl(device.fileno(), termios.TIOCEXCL)
    except Exception:
        device.close()
        raise
    return device


class RosmasterSerialSession:
    def __init__(self, serial_port="/dev/myserial", timeout=0.1,
                 write_timeout=0.2, delay=0.002, serial_factory=None):
        if not isinstance(serial_port, str) or not serial_port.strip():
            raise ValueError("serial_port must be a nonempty path")
        for name, value in (("timeout", timeout), ("write_timeout", write_timeout)):
            if not 0 < checked_number(name, value) <= 1:
                raise ValueError(f"{name} must be inside (0, 1] seconds")
        if not 0 <= checked_number("delay", delay) <= 1:
            raise ValueError("delay must be inside [0, 1] seconds")
        self.serial_port = serial_port
        self.delay = float(delay)
        self._lock = threading.RLock()
        self._closed = False
        self._fault = None
        self._buffer = bytearray()
        self._samples = {}
        self._stats = {"rx_bytes": 0, "tx_bytes": 0, "frames": 0,
                       "checksum_errors": 0, "length_errors": 0,
                       "unknown_frames": 0, "noise_bytes": 0}
        self._last_tx = None
        # Optional nonblocking run-archive sink; it never owns/closes the UART.
        self.recording_sink = None
        try:
            self.ser = (serial_factory or _open_serial)(
                port=serial_port, baudrate=115200, timeout=float(timeout),
                write_timeout=float(write_timeout), exclusive=True)
        except Exception as exc:
            raise RosmasterCommunicationError(f"open {serial_port}: {exc}") from exc

    def _error(self, operation, exc):
        self._fault = f"{operation}: {exc}"
        return RosmasterCommunicationError(f"{self.serial_port} {self._fault}")

    def _send(self, kind, payload, allow_fault=False):
        body = bytes([len(payload) + 3, kind]) + payload
        packet = b"\xff\xfc" + body + bytes([sum(body) & 0xff])
        with self._lock:
            if self._closed:
                raise RosmasterCommunicationError("session is closed")
            if self._fault and not allow_fault:
                raise RosmasterCommunicationError(f"session fault is latched: {self._fault}")
            try:
                count = self.ser.write(packet)
                if isinstance(count, int):
                    self._stats["tx_bytes"] += max(0, count)
                if count != len(packet):
                    raise OSError(f"short write {count}/{len(packet)} bytes")
                self._last_tx = {"hex": packet.hex(), "sent_monotonic": time.monotonic(),
                                 "acknowledged": False}
                if self.recording_sink is not None:
                    self.recording_sink('uart_tx',dict(self._last_tx))
            except Exception as exc:
                raise self._error("write", exc) from exc
            if self.delay:
                time.sleep(self.delay)
        return len(packet)

    def set_motor(self, *commands):
        if len(commands) != 4:
            raise ValueError("exactly four native motor commands are required")
        values = tuple(checked_integer("native motor command", x) for x in commands)
        if any(x < -100 or x > 100 for x in values):
            raise ValueError("native motor commands must be inside [-100, 100]")
        return self._send(0x10, struct.pack("<4b", *values), allow_fault=not any(values))

    def set_pwm_servo(self, servo_id, angle):
        servo_id = checked_integer("servo_id", servo_id)
        angle = checked_integer("angle", angle)
        if not 1 <= servo_id <= 4 or not 0 <= angle <= 180:
            raise ValueError("PWM servo ID must be 1..4 and angle 0..180")
        return self._send(0x03, bytes([servo_id, angle]))

    def poll(self):
        """Read at most 4096 bytes, with the configured finite serial timeout."""
        with self._lock:
            if self._closed:
                raise RosmasterCommunicationError("session is closed")
            try:
                waiting = self.ser.in_waiting
                data = self.ser.read(min(4096, max(1, waiting)))
            except Exception as exc:
                raise self._error("read", exc) from exc
            self._stats["rx_bytes"] += len(data)
            if data and self.recording_sink is not None:
                self.recording_sink('uart_rx', dict(received_monotonic=time.monotonic(),
                    data_hex=bytes(data).hex()))
            self._buffer.extend(data)
            self._parse()
            return bytes(data)

    def poll_available(self):
        """Read buffered telemetry without waiting for the control lock or bytes."""
        if not self._lock.acquire(blocking=False):
            return b""
        try:
            if self._closed:
                return b""
            try:
                if not self.ser.in_waiting:
                    return b""
            except Exception as exc:
                raise self._error("read", exc) from exc
            return self.poll()
        finally:
            self._lock.release()

    def _parse(self):
        while self._buffer:
            start = self._buffer.find(b"\xff\xfb")
            if start < 0:
                keep = 1 if self._buffer[-1] == 0xff else 0
                self._stats["noise_bytes"] += len(self._buffer) - keep
                self._buffer[:] = self._buffer[-1:] if keep else b""
                return
            if start:
                self._stats["noise_bytes"] += start
                del self._buffer[:start]
            if len(self._buffer) < 4:
                return
            length, kind = self._buffer[2:4]
            sizes = _GROUPS[kind][1] if kind in _GROUPS else None
            if isinstance(sizes, int):
                sizes = (sizes,)
            if not 3 <= length <= 64 or (sizes is not None and length - 3 not in sizes):
                self._stats["length_errors"] += 1
                del self._buffer[0]
                continue
            total = length + 2
            if len(self._buffer) < total:
                return
            packet = self._buffer[:total]
            if sum(packet[2:-1]) & 255 != packet[-1]:
                self._stats["checksum_errors"] += 1
                del self._buffer[0]
                continue
            del self._buffer[:total]
            self._stats["frames"] += 1
            if kind not in _GROUPS:
                self._stats["unknown_frames"] += 1
                continue
            payload = bytes(packet[4:-1])
            group = _GROUPS[kind][0]
            prior = self._samples.get(group, {})
            self._samples[group] = {
                "frame_type": kind, "received_monotonic": time.monotonic(),
                "sequence": prior.get("sequence", 0) + 1,
                "value": self._decode(kind, payload)}

    @staticmethod
    def _decode(kind, payload):
        if kind == 0x0a:
            vx, vy, vz, battery = struct.unpack("<hhhB", payload)
            return {"velocity": [vx / 1000, vy / 1000, vz / 1000], "battery_v": battery / 10}
        if kind == 0x0d:
            return {"native_ticks": list(struct.unpack("<4i", payload))}
        if kind == 0x0c:
            return {"radians": [x / 10000 for x in struct.unpack("<3h", payload)]}
        if kind in (0x0b, 0x0e):
            raw = list(struct.unpack("<9h", payload))
            if kind == 0x0b:
                gyro = [raw[0]/3754.9, -raw[1]/3754.9, -raw[2]/3754.9]
                accel = [x/1671.84 for x in raw[3:6]]
                mag = raw[6:9]
            else:
                gyro = [x/1000 for x in raw[:3]]
                accel = [x/1000 for x in raw[3:6]]
                mag = [x/1000 for x in raw[6:9]]
            return {"protocol": "MPU" if kind == 0x0b else "ICM", "raw_int16": raw,
                    "gyro_rad_s": gyro, "accel_m_s2": accel,
                    "mag_sdk_units": mag}
        if kind == 0x51:
            return {"major": payload[0], "minor": payload[1]}
        return {"car_type": payload[0], "reserved": payload[1] if len(payload) == 2 else None}

    def telemetry(self, max_age=0.5):
        from copy import deepcopy
        max_age = checked_number("max_age", max_age)
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        with self._lock:
            now = time.monotonic()
            result = {}
            for group in ("motion", "imu", "attitude", "encoder", "version", "car_type"):
                sample = deepcopy(self._samples.get(group))
                if sample is None:
                    result[group] = {"value": None, "valid": False, "stale": True,
                                     "age_s": None, "received_monotonic": None, "sequence": 0,
                                     "frame_type": None}
                else:
                    sample["age_s"] = max(0, now - sample["received_monotonic"])
                    sample["stale"] = sample["age_s"] > max_age
                    sample["valid"] = not self._closed and not self._fault and not sample["stale"]
                    result[group] = sample
            return result

    def get_status(self):
        with self._lock:
            return {**self._stats, "device": self.serial_port, "baudrate": 115200,
                    "closed": self._closed, "healthy": not self._closed and self._fault is None,
                    "error": self._fault, "last_tx": dict(self._last_tx) if self._last_tx else None}

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                try:
                    fcntl.ioctl(self.ser.fileno(), termios.TIOCNXCL)
                except (AttributeError, OSError):
                    pass
                self.ser.close()
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
