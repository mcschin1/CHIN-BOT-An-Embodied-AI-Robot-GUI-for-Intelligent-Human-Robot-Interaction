"""
telemetry_server.py
───────────────────
CHIN-BOT Submodule 7 — WebSocket Telemetry Server

Broadcasts robot state, detection frames, and chat events
over WebSocket so the HTML GUI can receive live data.

The GUI connects to ws://localhost:8765 and receives JSON
messages categorised by "type": "telemetry" | "detection" | "chat".

Dependencies:
    pip install websockets asyncio

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import json
import time
import asyncio
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    logging.warning("websockets not found. Install with: pip install websockets")

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765
BROADCAST_HZ = 20     # how often telemetry is pushed to all clients
# ──────────────────────────────────────────────────────────────────────────────


class TelemetryServer:
    """
    Async WebSocket server. Broadcasts three message types:

    { "type": "telemetry", "data": { x, y, heading, speed, battery, ... } }
    { "type": "detection", "data": [ {cls, confidence, bbox_norm, center}, ... ] }
    { "type": "chat",      "data": { role: "assistant"|"user", text: "..." } }

    Parameters
    ----------
    host     : bind address
    port     : WebSocket port (default 8765)

    Usage
    -----
        server = TelemetryServer()
        server.start_background()         # non-blocking thread

        # Push data from robot modules:
        server.push_telemetry(state_dict)
        server.push_detections(det_list)
        server.push_chat("assistant", "I can see a person ahead.")
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._queue: asyncio.Queue = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_background(self):
        """Start the server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        # Give the loop a moment to initialise
        time.sleep(0.3)
        logger.info("Telemetry server running on ws://%s:%d", self.host, self.port)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        if not WS_AVAILABLE:
            logger.error("websockets library not available — server disabled.")
            return
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()   # run forever

    async def _handler(self, ws: "WebSocketServerProtocol", path: str = None):
        self._clients.add(ws)
        addr = ws.remote_address
        logger.info("GUI client connected: %s", addr)
        try:
            async for msg in ws:
                # Accept commands from the GUI (e.g. locomotion, chat)
                try:
                    data = json.loads(msg)
                    logger.debug("Received from GUI: %s", data)
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("GUI client disconnected: %s", addr)

    # ── Push methods (thread-safe) ────────────────────────────────────────────

    def push_telemetry(self, state_dict: dict):
        self._enqueue({"type": "telemetry", "data": state_dict})

    def push_detections(self, detections: list[dict]):
        self._enqueue({"type": "detection", "data": detections})

    def push_chat(self, role: str, text: str):
        self._enqueue({"type": "chat", "data": {"role": role, "text": text}})

    def push_log(self, source: str, level: str, message: str):
        self._enqueue({"type": "log", "data": {
            "source": source, "level": level, "message": message,
            "timestamp": time.time(),
        }})

    def _enqueue(self, msg: dict):
        if self._loop and self._queue:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    async def _broadcast(self, msg: dict):
        if not self._clients:
            return
        payload = json.dumps(msg)
        dead = set()
        for ws in self._clients.copy():
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ── Client command receiver ───────────────────────────────────────────────

    def on_command(self, handler: callable):
        """
        Register a handler for commands arriving from the GUI.
        Handler signature: handler(cmd: dict)

        Expected GUI commands:
          { "type": "move",  "direction": "fwd"|"bwd"|"left"|"right"|"stop" }
          { "type": "speed", "value": 1..10 }
          { "type": "chat",  "text": "..." }
        """
        self._cmd_handler = handler


# ── HTTP endpoint for file downloads (serves the Python files) ────────────────

class FileServer:
    """
    Minimal HTTP server to serve Python source files for download
    from the GUI's 'Download Source' buttons.

    Usage
    -----
        fs = FileServer(directory="/path/to/chinbot", port=8766)
        fs.start_background()
    """

    def __init__(self, directory: str = ".", port: int = 8766):
        self.directory = directory
        self.port      = port

    def start_background(self):
        import http.server
        import functools
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=self.directory,
        )
        server = http.server.HTTPServer(("", self.port), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("File server on http://localhost:%d", self.port)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    server = TelemetryServer()
    server.start_background()

    print("WebSocket server running on ws://localhost:8765")
    print("Connect your HTML GUI or test with: wscat -c ws://localhost:8765")

    t = 0
    try:
        while True:
            server.push_telemetry({
                "x": math.sin(t) * 2, "y": math.cos(t) * 2,
                "heading": (t * 10) % 360, "speed": abs(math.sin(t)) * 0.5,
                "battery": max(20, 100 - t * 0.1), "cpu": 30 + random.uniform(-5, 5),
                "temp": 45 + random.uniform(-2, 2), "signal": 85 + random.uniform(-3, 3),
                "sonar": {"L2": 4.0, "L1": 3.0, "FWD": 1.5+random.uniform(0,.5),
                           "R1": 2.5, "R2": 4.5},
            })
            server.push_log("TEST", "info", f"Simulation tick {int(t)}")
            t += 0.1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")
