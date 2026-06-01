"""
tts_service.py
──────────────
CHIN-BOT Submodule 4 — Text-to-Speech Service

Converts LLM text replies to audio and plays them back.
Supports OpenAI TTS-1 (cloud) and pyttsx3 (offline fallback).

Dependencies:
    pip install openai pyttsx3 playsound

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import io
import os
import logging
import tempfile
import threading
from typing import Literal

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_VOICE   = "alloy"      # OpenAI TTS voice: alloy|echo|fable|onyx|nova|shimmer
DEFAULT_MODEL   = "tts-1"      # tts-1 or tts-1-hd
DEFAULT_SPEED   = 1.05         # Slightly faster sounds more robotic
DEFAULT_PITCH   = 0.85         # pyttsx3 pitch multiplier
# ──────────────────────────────────────────────────────────────────────────────


class TTSService:
    """
    Text-to-speech playback service.

    Tries OpenAI TTS first; falls back to pyttsx3 if unavailable.
    Playback is non-blocking by default (runs in a daemon thread).

    Parameters
    ----------
    provider  : "openai" or "local"
    api_key   : OpenAI API key
    voice     : OpenAI voice name
    speed     : speech rate multiplier
    blocking  : if True, wait for playback to finish before returning

    Usage
    -----
        tts = TTSService(provider="openai", api_key="sk-...")
        tts.speak("I can see a person directly ahead.")
    """

    def __init__(
        self,
        provider: Literal["openai", "local"] = "openai",
        api_key:  str  = None,
        voice:    str  = DEFAULT_VOICE,
        speed:    float = DEFAULT_SPEED,
        blocking: bool  = False,
    ):
        self.provider = provider
        self.voice    = voice
        self.speed    = speed
        self.blocking = blocking
        self._lock    = threading.Lock()   # prevent overlapping playback

        if provider == "openai" and OPENAI_AVAILABLE:
            self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            logger.info("TTS: using OpenAI TTS-1 voice '%s'", voice)
        elif PYTTSX3_AVAILABLE:
            self.provider = "local"
            self._engine  = pyttsx3.init()
            self._engine.setProperty("rate",  int(180 * speed))
            self._engine.setProperty("volume", 0.9)
            logger.info("TTS: using pyttsx3 (offline)")
        else:
            logger.warning("TTS: no backend available — speech disabled.")
            self._client = None
            self._engine = None

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Speak text. Non-blocking unless self.blocking=True."""
        if not text.strip():
            return
        if self.blocking:
            self._speak_sync(text)
        else:
            t = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
            t.start()

    def speak_greeting(self):
        self.speak("CHIN online. Vision system active. How can I assist you?")

    def speak_goodbye(self):
        self.speak("Session ended. Goodbye.")

    # ── Backends ──────────────────────────────────────────────────────────────

    def _speak_sync(self, text: str):
        with self._lock:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                self._speak_openai(text)
            elif self.provider == "local" and PYTTSX3_AVAILABLE:
                self._speak_local(text)
            else:
                logger.warning("TTS: no backend — would say: %s", text)

    def _speak_openai(self, text: str):
        try:
            response = self._client.audio.speech.create(
                model=DEFAULT_MODEL,
                voice=self.voice,
                input=text,
                speed=self.speed,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(response.content)

            if PLAYSOUND_AVAILABLE:
                playsound(tmp_path)
            else:
                # Fallback: use system player
                os.system(f"{'afplay' if os.name != 'nt' else 'start'} \"{tmp_path}\"")
            os.unlink(tmp_path)
        except Exception as exc:
            logger.error("OpenAI TTS error: %s", exc)

    def _speak_local(self, text: str):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            logger.error("pyttsx3 error: %s", exc)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    tts = TTSService(
        provider="openai",
        api_key=os.environ.get("OPENAI_API_KEY"),
        voice="onyx",
        blocking=True,
    )
    test_lines = [
        "CHIN online. Vision system active.",
        "I can see a person directly ahead at two point four metres.",
        "Battery level seventy eight percent. All systems nominal.",
    ]
    for line in test_lines:
        print(f"Speaking: {line}")
        tts.speak(line)
