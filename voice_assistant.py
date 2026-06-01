"""
voice_assistant.py
──────────────────
CHIN-BOT Submodule 2 — Voice Assistant Service

Records sentences after wake-word activation, filters gibberish,
detects end-of-conversation, and forwards clean transcripts to
the chat service.

Dependencies:
    pip install pyaudio openai numpy

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import os
import re
import time
import wave
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
CHANNELS          = 1
CHUNK_SIZE        = 1024
SENTENCE_SECONDS  = 5.0     # How long each conversation turn is recorded
FORMAT            = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

GOODBYE_PHRASES   = [
    "goodbye", "bye", "exit", "quit", "stop", "shut down",
    "power off", "end session", "that's all", "thats all",
]
GIBBERISH_MIN_CHARS = 3     # Fewer clean alphabetic chars → gibberish
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single user utterance and its metadata."""
    raw_text:   str
    clean_text: str
    timestamp:  float = field(default_factory=time.time)
    is_gibberish: bool = False
    is_goodbye:   bool = False


class VoiceAssistant:
    """
    Manages the listen → transcribe → filter → dispatch loop
    that runs after the wake word has been detected.

    Parameters
    ----------
    api_key          : OpenAI API key (or set OPENAI_API_KEY env var)
    on_user_turn     : called with ConversationTurn for each clean utterance
    on_goodbye       : called when end-of-conversation is detected
    sentence_seconds : recording length per turn (default 5 s)

    Usage
    -----
        va = VoiceAssistant(
            api_key="sk-...",
            on_user_turn=lambda t: chat_service.send(t.clean_text),
            on_goodbye=lambda: print("Goodbye!"),
        )
        va.run_session()   # blocks until goodbye or stop()
    """

    def __init__(
        self,
        api_key: str = None,
        on_user_turn: Callable[[ConversationTurn], None] = None,
        on_goodbye: Callable[[], None] = None,
        sentence_seconds: float = SENTENCE_SECONDS,
    ):
        self._api_key         = api_key or os.environ.get("OPENAI_API_KEY")
        self.on_user_turn     = on_user_turn
        self.on_goodbye       = on_goodbye
        self.sentence_seconds = sentence_seconds
        self._running         = False

        self._client = OpenAI(api_key=self._api_key) if OPENAI_AVAILABLE else None
        self._pa     = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None

    # ── Session control ───────────────────────────────────────────────────────

    def run_session(self):
        """
        Blocking conversation loop.  Records turns until:
        - A goodbye phrase is detected, OR
        - stop() is called from another thread.
        """
        self._running = True
        logger.info("Voice assistant session started.")

        while self._running:
            turn = self._record_and_transcribe()
            if turn is None:
                continue

            if turn.is_gibberish:
                logger.debug("Gibberish detected — re-recording.")
                continue

            if turn.is_goodbye:
                logger.info("End-of-conversation detected: '%s'", turn.clean_text)
                if self.on_goodbye:
                    self.on_goodbye()
                self._running = False
                break

            logger.info("User said: '%s'", turn.clean_text)
            if self.on_user_turn:
                self.on_user_turn(turn)

    def stop(self):
        self._running = False

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _record_and_transcribe(self) -> Optional[ConversationTurn]:
        """Record one sentence clip and return a ConversationTurn."""
        pcm = self._record_clip(self.sentence_seconds)
        if pcm is None:
            return None

        text = self._transcribe(pcm)
        return self._classify(text)

    def _record_clip(self, duration: float) -> Optional[bytes]:
        if not self._pa:
            logger.error("PyAudio not available.")
            return None

        stream = self._pa.open(
            format=FORMAT, channels=CHANNELS,
            rate=SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        frames = []
        n = int(SAMPLE_RATE / CHUNK_SIZE * duration)
        logger.debug("Recording %gs clip…", duration)
        for _ in range(n):
            frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        return b"".join(frames)

    def _transcribe(self, pcm: bytes) -> str:
        if not self._client:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
            _write_wav(path, pcm)
        try:
            with open(path, "rb") as f:
                res = self._client.audio.transcriptions.create(
                    model="whisper-1", file=f, language="en"
                )
            return res.text.strip()
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return ""
        finally:
            os.unlink(path)

    # ── Classification helpers ────────────────────────────────────────────────

    def _classify(self, raw: str) -> ConversationTurn:
        clean = raw.strip().lower()
        turn = ConversationTurn(raw_text=raw, clean_text=clean)
        turn.is_gibberish = self._is_gibberish(clean)
        if not turn.is_gibberish:
            turn.is_goodbye = self._is_goodbye(clean)
        return turn

    @staticmethod
    def _is_gibberish(text: str) -> bool:
        """Return True if text has too few real alphabetic characters."""
        clean = re.sub(r"[^a-zA-Z ]", "", text.strip())
        return len(clean) < GIBBERISH_MIN_CHARS

    @staticmethod
    def _is_goodbye(text: str) -> bool:
        """Return True if text contains any goodbye phrase."""
        t = text.lower()
        return any(phrase in t for phrase in GOODBYE_PHRASES)

    def __del__(self):
        if self._pa:
            self._pa.terminate()


# ── Shared utility ────────────────────────────────────────────────────────────

def _write_wav(path: str, pcm: bytes):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    def handle_turn(turn: ConversationTurn):
        print(f"  → User: {turn.clean_text}")

    def handle_goodbye():
        print("  → Session ended by user.")

    va = VoiceAssistant(
        api_key=os.environ.get("OPENAI_API_KEY"),
        on_user_turn=handle_turn,
        on_goodbye=handle_goodbye,
        sentence_seconds=5.0,
    )
    print("Voice session active. Speak. Say 'goodbye' to end. Ctrl-C to force stop.")
    try:
        va.run_session()
    except KeyboardInterrupt:
        va.stop()
