"""
main.py
───────
CHIN-BOT Orchestrator — Wires all submodules together

Starts all services in the correct order, connects their
callbacks, and keeps the system running until interrupted.

Run:
    python main.py                    # full hardware mode
    python main.py --simulate         # simulation (no hardware needed)
    python main.py --provider openai  # use GPT-4o instead of Claude
    python main.py --gui-only         # start servers only (use HTML GUI)

Dependencies:
    pip install anthropic openai pyaudio ultralytics
    pip install opencv-python pyserial websockets pyttsx3

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import os
import sys
import time
import logging
import argparse
import threading

# ── Submodule imports ──────────────────────────────────────────────────────────
from wake_word_detector import WakeWordDetector
from voice_assistant    import VoiceAssistant, ConversationTurn
from chat_service       import ChatService
from tts_service        import TTSService
from object_detector    import ObjectDetector, DetectionFrame
from robot_controller   import RobotController, RobotState
from telemetry_server   import TelemetryServer, FileServer

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CHIN-BOT Embodied AI Robot")
    p.add_argument("--simulate",   action="store_true", help="Simulation mode (no hardware)")
    p.add_argument("--gui-only",   action="store_true", help="Skip voice loop; serve GUI only")
    p.add_argument("--provider",   default="anthropic", choices=["anthropic","openai"],
                   help="LLM provider")
    p.add_argument("--model",      default=None,        help="Override model name")
    p.add_argument("--wake-word",  default="chin",      help="Wake word (default: chin)")
    p.add_argument("--camera",     default=0,           help="Camera index or RTSP URL")
    p.add_argument("--serial",     default="/dev/ttyUSB0", help="Serial port for robot hardware")
    p.add_argument("--ws-port",    default=8765, type=int,  help="WebSocket port for GUI")
    p.add_argument("--http-port",  default=8766, type=int,  help="HTTP port for file downloads")
    return p.parse_args()


# ─── System orchestrator ──────────────────────────────────────────────────────
class CHINBot:
    def __init__(self, args):
        self.args = args
        self._running   = False
        self._in_session = False

        # Resolve API keys
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key    = os.environ.get("OPENAI_API_KEY",    "")

        # Initialise subsystems
        logger.info("═══════════════════════════════════")
        logger.info("  CHIN-BOT initialising")
        logger.info("  Provider : %s", args.provider)
        logger.info("  Simulate : %s", args.simulate)
        logger.info("═══════════════════════════════════")

        self._tts    = self._init_tts()
        self._chat   = self._init_chat()
        self._robot  = self._init_robot()
        self._vision = self._init_vision()
        self._server = self._init_server()
        self._wake   = self._init_wake_word()
        self._va     = self._init_voice_assistant()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_tts(self) -> TTSService:
        provider = "openai" if self._openai_key else "local"
        return TTSService(provider=provider, api_key=self._openai_key, blocking=False)

    def _init_chat(self) -> ChatService:
        key = self._anthropic_key if self.args.provider == "anthropic" else self._openai_key
        return ChatService(
            provider=self.args.provider,
            api_key=key,
            model=self.args.model,
        )

    def _init_robot(self) -> RobotController:
        ctrl = RobotController(
            port=self.args.serial,
            simulate=self.args.simulate,
            on_state=self._on_robot_state,
        )
        ctrl.start()
        return ctrl

    def _init_vision(self) -> ObjectDetector:
        try:
            cam = int(self.args.camera)
        except ValueError:
            cam = self.args.camera
        return ObjectDetector(
            camera_index=cam,
            on_frame=self._on_detection_frame,
        )

    def _init_server(self):
        srv = TelemetryServer(port=self.args.ws_port)
        srv.start_background()
        fs = FileServer(directory=os.path.dirname(os.path.abspath(__file__)),
                        port=self.args.http_port)
        fs.start_background()
        return srv

    def _init_wake_word(self) -> WakeWordDetector:
        return WakeWordDetector(
            api_key=self._openai_key,
            wake_word=self.args.wake_word,
            on_detected=self._on_wake_detected,
        )

    def _init_voice_assistant(self) -> VoiceAssistant:
        return VoiceAssistant(
            api_key=self._openai_key,
            on_user_turn=self._on_user_turn,
            on_goodbye=self._on_goodbye,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_robot_state(self, state: RobotState):
        """Fires at SENSOR_HZ — push to GUI via WebSocket."""
        self._server.push_telemetry({
            "x":        round(state.x, 2),
            "y":        round(state.y, 2),
            "heading":  round(state.heading, 1),
            "speed":    round(state.speed, 2),
            "battery":  round(state.battery, 1),
            "cpu":      round(state.cpu, 1),
            "temp":     round(state.temp, 1),
            "signal":   round(state.signal, 1),
            "sonar":    {k: round(v, 2) for k, v in state.sonar.items()},
        })

    def _on_detection_frame(self, frame: DetectionFrame):
        """Fires at TARGET_FPS — push detections to GUI."""
        self._server.push_detections([d.to_dict() for d in frame.detections])
        self._server.push_log("VISION", "detect",
            f"{frame.count} objects | {frame.latency_ms:.1f}ms")

    def _on_wake_detected(self, text: str):
        """Wake word confirmed — start a voice session."""
        if self._in_session:
            return
        logger.info("Wake word detected — starting session.")
        self._tts.speak_greeting()
        self._server.push_log("VOICE", "voice", "Wake word detected — session started")
        self._in_session = True
        threading.Thread(target=self._va.run_session, daemon=True).start()

    def _on_user_turn(self, turn: ConversationTurn):
        """Transcribed user utterance — send to LLM and speak reply."""
        self._server.push_chat("user", turn.clean_text)
        self._server.push_log("VOICE", "voice", f"User: {turn.clean_text[:60]}")

        # Build scene context from latest sensor + detection data
        scene = self._build_scene_context()
        reply = self._chat.send(turn.clean_text, scene_context=scene)

        self._server.push_chat("assistant", reply)
        self._server.push_log("LLM", "llm", f"CHIN: {reply[:60]}")
        self._tts.speak(reply)

    def _on_goodbye(self):
        self._in_session = False
        self._tts.speak_goodbye()
        self._chat.clear()
        self._server.push_log("VOICE", "voice", "Session ended by user.")
        logger.info("Session ended — returning to wake-word detection.")

    # ── Scene context builder ─────────────────────────────────────────────────

    def _build_scene_context(self) -> str:
        parts = []
        parts.append(self._robot.state.to_context_string())
        if self._vision.latest_frame:
            parts.append(self._vision.latest_frame.to_context_string())
        return "\n".join(parts)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        logger.info("CHIN-BOT ready.")
        self._server.push_log("SYSTEM", "info", "CHIN-BOT fully initialised and ready.")

        # Start vision in background thread
        vision_thread = threading.Thread(target=self._vision.start, daemon=True)
        vision_thread.start()

        if not self.args.gui_only:
            self._tts.speak("CHIN online. Say the wake word to begin.")
            logger.info("Listening for wake word: '%s'", self.args.wake_word)
            self._wake.start()     # blocks here until stop()
        else:
            logger.info("GUI-only mode — WebSocket ready, voice loop inactive.")
            print(f"\n  GUI:       open EmbodiedAI_Robot_GUI.html in your browser")
            print(f"  WebSocket: ws://localhost:{self.args.ws_port}")
            print(f"  Downloads: http://localhost:{self.args.http_port}")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    def shutdown(self):
        logger.info("Shutting down CHIN-BOT…")
        self._wake.stop()
        self._va.stop()
        self._vision.stop()
        self._robot.stop()
        self._running = False
        logger.info("Shutdown complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import math   # needed by telemetry_server standalone test
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  No API keys found in environment.")
        print("   Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY before running.\n")

    bot = CHINBot(args)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nInterrupt received.")
    finally:
        bot.shutdown()
