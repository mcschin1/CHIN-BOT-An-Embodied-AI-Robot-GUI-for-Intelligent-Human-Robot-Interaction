"""
wake_word_detector.py
─────────────────────
CHIN-BOT Submodule 1 — Wake-Word Detection Service

Records short audio clips and uses OpenAI Whisper to check
whether the configured wake word was spoken.

Dependencies:
    pip install pyaudio openai numpy

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import io
import os
import time
import wave
import tempfile
import logging
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("pyaudio not found — microphone input disabled. Install with: pip install pyaudio")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not found — transcription disabled. Install with: pip install openai")

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000   # Hz — Whisper prefers 16 kHz
CHANNELS         = 1       # Mono
CHUNK_SIZE       = 1024    # Frames per buffer
RECORD_SECONDS   = 2.0     # How long each wake-word clip is
FORMAT           = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
DEFAULT_WAKE     = "chin"  # Wake word (case-insensitive)
ENERGY_THRESHOLD = 300     # RMS below this → silence, skip transcription
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Continuously records SHORT audio clips and checks whether
    the wake word appears in the Whisper transcription.

    Usage
    -----
        detector = WakeWordDetector(api_key="sk-...", wake_word="chin")
        detector.start()           # blocking loop
        # or
        detected = detector.listen_once()   # single clip check → bool
    """

    def __init__(
        self,
        api_key: str = None,
        wake_word: str = DEFAULT_WAKE,
        record_seconds: float = RECORD_SECONDS,
        energy_threshold: int = ENERGY_THRESHOLD,
        on_detected=None,
    ):
        self.wake_word       = wake_word.lower()
        self.record_seconds  = record_seconds
        self.energy_threshold = energy_threshold
        self.on_detected     = on_detected  # callback(text: str) when triggered
        self._running        = False

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY")) \
                       if OPENAI_AVAILABLE else None

        if PYAUDIO_AVAILABLE:
            self._pa = pyaudio.PyAudio()
        else:
            self._pa = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Blocking wake-word detection loop. Call stop() from another thread."""
        self._running = True
        logger.info("Wake-word detector started. Listening for '%s'…", self.wake_word)
        while self._running:
            detected, text = self.listen_once()
            if detected:
                logger.info("Wake word detected in: '%s'", text)
                if self.on_detected:
                    self.on_detected(text)
            else:
                time.sleep(0.05)   # tiny pause before next clip

    def stop(self):
        self._running = False
        logger.info("Wake-word detector stopped.")

    def listen_once(self) -> tuple[bool, str]:
        """
        Record one clip, transcribe, check for wake word.
        Returns (detected: bool, transcription: str).
        """
        audio_bytes = self._record_clip()
        if audio_bytes is None:
            return False, ""

        rms = self._rms(audio_bytes)
        if rms < self.energy_threshold:
            logger.debug("Clip too quiet (RMS=%d) — skipping transcription.", rms)
            return False, ""

        text = self._transcribe(audio_bytes)
        detected = self._contains_wake_word(text)
        return detected, text

    # ── Audio recording ───────────────────────────────────────────────────────

    def _record_clip(self) -> bytes | None:
        """Record RECORD_SECONDS of audio and return raw PCM bytes."""
        if not self._pa:
            logger.error("PyAudio unavailable — cannot record.")
            return None

        stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        frames = []
        n_chunks = int(SAMPLE_RATE / CHUNK_SIZE * self.record_seconds)
        for _ in range(n_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        return b"".join(frames)

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, pcm_bytes: bytes) -> str:
        """Send PCM audio to OpenAI Whisper and return transcript."""
        if not self._client:
            logger.error("OpenAI client unavailable — returning empty transcript.")
            return ""

        # Write to a temp WAV file (Whisper needs a container format)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            self._write_wav(tmp_path, pcm_bytes)

        try:
            with open(tmp_path, "rb") as f:
                result = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                )
            text = result.text.strip().lower()
            logger.debug("Whisper transcript: '%s'", text)
            return text
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return ""
        finally:
            os.unlink(tmp_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_wav(self, path: str, pcm_bytes: bytes):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)          # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)

    def _rms(self, pcm_bytes: bytes) -> float:
        """Root-mean-square energy of the audio clip."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0

    def _contains_wake_word(self, text: str) -> bool:
        """Case-insensitive substring check for the wake word."""
        return self.wake_word in text.lower()

    def __del__(self):
        if self._pa:
            self._pa.terminate()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    def on_wake(text):
        print(f"\n✅  Wake word detected! Full text: '{text}'\n")

    detector = WakeWordDetector(
        api_key=os.environ.get("OPENAI_API_KEY"),
        wake_word="chin",
        on_detected=on_wake,
    )
    print("Listening… say 'chin' to trigger. Ctrl-C to stop.")
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()
        print("Stopped.")
