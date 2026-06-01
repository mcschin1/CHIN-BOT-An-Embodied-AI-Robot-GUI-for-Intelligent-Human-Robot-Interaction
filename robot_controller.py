"""
robot_controller.py
───────────────────
CHIN-BOT Submodule 6 — Robot Controller

Manages locomotion commands, dead-reckoning odometry,
and sensor polling. Interfaces with hardware via serial
(Arduino/ROS) or simulates when hardware is absent.

Dependencies:
    pip install pyserial

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("pyserial not found. Install with: pip install pyserial")

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_PORT     = "/dev/ttyUSB0"   # Linux; use "COM3" on Windows
DEFAULT_BAUD     = 115200
WHEEL_BASE       = 0.30             # metres between left/right wheels
MAX_SPEED        = 0.5              # m/s
SENSOR_HZ        = 20               # sensor polling rate
ODOMETRY_HZ      = 50               # dead-reckoning update rate
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotState:
    """Full snapshot of the robot's physical state."""
    x:         float = 0.0    # metres from origin
    y:         float = 0.0
    heading:   float = 0.0    # degrees, 0 = north
    speed:     float = 0.0    # m/s
    battery:   float = 100.0  # %
    cpu:       float = 0.0    # %
    temp:      float = 35.0   # °C
    signal:    float = 100.0  # %
    sonar_fwd: float = 5.0    # metres to nearest obstacle (forward)
    sonar:     dict  = field(default_factory=lambda: {
        "L2": 5.0, "L1": 5.0, "FWD": 5.0, "R1": 5.0, "R2": 5.0
    })
    timestamp: float = field(default_factory=time.time)

    def to_context_string(self) -> str:
        return (
            f"Position: ({self.x:.1f}m, {self.y:.1f}m), heading {self.heading:.0f}°. "
            f"Speed: {self.speed:.1f} m/s. "
            f"Nearest obstacle (fwd): {self.sonar['FWD']:.1f}m. "
            f"Battery: {self.battery:.0f}%. CPU: {self.cpu:.0f}%."
        )


class RobotController:
    """
    High-level locomotion and telemetry controller.

    In hardware mode, sends velocity commands over serial to a
    microcontroller (Arduino / ROS serial bridge).

    In simulation mode (default when no serial port is found),
    updates a dead-reckoning state model.

    Parameters
    ----------
    port        : serial port string  (e.g. "/dev/ttyUSB0")
    on_state    : callback(RobotState) called at SENSOR_HZ
    simulate    : force simulation mode even if serial is available

    Usage
    -----
        ctrl = RobotController(on_state=lambda s: print(s.to_context_string()))
        ctrl.start()
        ctrl.move_forward(speed=0.3, duration=2.0)
        ctrl.stop()
    """

    def __init__(
        self,
        port:     str  = DEFAULT_PORT,
        on_state: Callable[[RobotState], None] = None,
        simulate: bool = False,
    ):
        self.state     = RobotState()
        self.on_state  = on_state
        self._running  = False
        self._cmd_lock = threading.Lock()
        self._vl = 0.0  # left wheel velocity
        self._vr = 0.0  # right wheel velocity
        self._serial: Optional[serial.Serial] = None

        if not simulate and SERIAL_AVAILABLE:
            try:
                self._serial = serial.Serial(port, DEFAULT_BAUD, timeout=0.1)
                logger.info("Robot controller: serial connected on %s", port)
            except serial.SerialException as exc:
                logger.warning("Serial unavailable (%s) — simulation mode.", exc)

        if self._serial is None:
            logger.info("Robot controller: simulation mode active.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start background threads for odometry and sensor polling."""
        self._running = True
        threading.Thread(target=self._odometry_loop, daemon=True).start()
        threading.Thread(target=self._sensor_loop,   daemon=True).start()
        logger.info("Robot controller started.")

    def stop(self):
        self._running = False
        self.halt()
        if self._serial:
            self._serial.close()
        logger.info("Robot controller stopped.")

    # ── Locomotion commands ───────────────────────────────────────────────────

    def halt(self):
        self._set_velocity(0.0, 0.0)

    def move_forward(self, speed: float = 0.3, duration: float = None):
        s = min(abs(speed), MAX_SPEED)
        self._set_velocity(s, s)
        if duration:
            time.sleep(duration)
            self.halt()

    def move_backward(self, speed: float = 0.3, duration: float = None):
        s = min(abs(speed), MAX_SPEED)
        self._set_velocity(-s, -s)
        if duration:
            time.sleep(duration)
            self.halt()

    def turn_left(self, speed: float = 0.2, duration: float = None):
        s = min(abs(speed), MAX_SPEED)
        self._set_velocity(-s, s)
        if duration:
            time.sleep(duration)
            self.halt()

    def turn_right(self, speed: float = 0.2, duration: float = None):
        s = min(abs(speed), MAX_SPEED)
        self._set_velocity(s, -s)
        if duration:
            time.sleep(duration)
            self.halt()

    def set_heading(self, target_deg: float, speed: float = 0.2):
        """Rotate until heading matches target_deg (±5°)."""
        while self._running:
            diff = (target_deg - self.state.heading + 360) % 360
            if diff > 180:
                diff -= 360
            if abs(diff) < 5:
                self.halt()
                break
            direction = 1 if diff > 0 else -1
            self._set_velocity(direction * speed, -direction * speed)
            time.sleep(0.05)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_velocity(self, vl: float, vr: float):
        with self._cmd_lock:
            self._vl = max(-MAX_SPEED, min(MAX_SPEED, vl))
            self._vr = max(-MAX_SPEED, min(MAX_SPEED, vr))
        if self._serial:
            cmd = f"V {self._vl:.3f} {self._vr:.3f}\n"
            try:
                self._serial.write(cmd.encode())
            except serial.SerialException as exc:
                logger.error("Serial write error: %s", exc)

    def _odometry_loop(self):
        """Dead-reckoning position update at ODOMETRY_HZ."""
        dt = 1.0 / ODOMETRY_HZ
        while self._running:
            with self._cmd_lock:
                vl, vr = self._vl, self._vr

            v_centre = (vl + vr) / 2.0
            omega    = (vr - vl) / WHEEL_BASE   # rad/s

            h_rad = math.radians(self.state.heading)
            self.state.x       += v_centre * math.cos(h_rad) * dt
            self.state.y       += v_centre * math.sin(h_rad) * dt
            self.state.heading  = (self.state.heading + math.degrees(omega) * dt) % 360
            self.state.speed    = abs(v_centre)
            self.state.timestamp = time.time()
            time.sleep(dt)

    def _sensor_loop(self):
        """Poll sensors (simulated) and fire on_state callback."""
        import random
        dt = 1.0 / SENSOR_HZ
        while self._running:
            if self._serial:
                self._poll_serial_sensors()
            else:
                # Simulate realistic sensor noise
                self.state.battery   = max(5.0,  self.state.battery - 0.001)
                self.state.cpu       = max(10.0, min(95.0, self.state.cpu + random.uniform(-1, 1)))
                self.state.temp      = max(30.0, min(80.0, self.state.temp + random.uniform(-0.1, 0.1)))
                self.state.signal    = max(20.0, min(100.0, self.state.signal + random.uniform(-0.5, 0.5)))
                self.state.sonar["FWD"] = max(0.15, min(8.0,
                    self.state.sonar["FWD"] + random.uniform(-0.05, 0.05)))
                for k in self.state.sonar:
                    self.state.sonar[k] = max(0.1, min(8.0,
                        self.state.sonar[k] + random.uniform(-0.02, 0.02)))

            if self.on_state:
                self.on_state(self.state)
            time.sleep(dt)

    def _poll_serial_sensors(self):
        """Read one line of JSON telemetry from the microcontroller."""
        import json
        try:
            line = self._serial.readline().decode().strip()
            if line:
                data = json.loads(line)
                self.state.battery = data.get("bat", self.state.battery)
                self.state.temp    = data.get("temp", self.state.temp)
                sonar = data.get("sonar", {})
                self.state.sonar.update(sonar)
                self.state.sonar_fwd = self.state.sonar.get("FWD", self.state.sonar_fwd)
        except Exception:
            pass


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    def print_state(s: RobotState):
        print(f"\r{s.to_context_string()}", end="", flush=True)

    ctrl = RobotController(simulate=True, on_state=print_state)
    ctrl.start()

    print("Moving forward 1s…")
    ctrl.move_forward(speed=0.3, duration=1.0)
    print("\nTurning left 1s…")
    ctrl.turn_left(speed=0.2, duration=1.0)
    print("\nHalting.")
    ctrl.stop()
