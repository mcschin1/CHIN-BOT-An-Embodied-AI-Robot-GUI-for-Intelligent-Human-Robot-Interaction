"""
chat_service.py
───────────────
CHIN-BOT Submodule 3 — Chat Service

Maintains conversation history and sends turns to either the
Anthropic Claude API or OpenAI GPT-4o. Returns text replies
for downstream TTS playback.

Dependencies:
    pip install anthropic openai

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic not found. Install with: pip install anthropic")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_GPT_MODEL    = "gpt-4o"
DEFAULT_MAX_TOKENS   = 300
DEFAULT_SYSTEM       = (
    "You are CHIN, a helpful and precise embodied AI robot assistant. "
    "You respond concisely (2–4 sentences), in plain English. "
    "You are aware of your sensor state and what your camera currently detects. "
    "When asked to perform an action, acknowledge it and describe what you would do."
)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role:      Literal["user", "assistant"]
    content:   str
    timestamp: float = field(default_factory=time.time)
    tokens:    int   = 0


class ChatService:
    """
    Stateful chat service that wraps Anthropic Claude or OpenAI GPT.

    Maintains a rolling conversation history so the robot remembers
    context across turns within a session.

    Parameters
    ----------
    provider      : "anthropic" or "openai"
    api_key       : API key (falls back to env vars)
    model         : model string override
    system_prompt : base system prompt (will be extended with scene context)
    max_tokens    : maximum tokens per response
    history_limit : max past turns kept in context (older ones are pruned)

    Usage
    -----
        chat = ChatService(provider="anthropic", api_key="sk-ant-...")
        reply = chat.send("What do you see in front of you?",
                          scene_context="PERSON detected at 2m, 94% confidence.")
        print(reply)
    """

    def __init__(
        self,
        provider:      Literal["anthropic", "openai"] = "anthropic",
        api_key:       str  = None,
        model:         str  = None,
        system_prompt: str  = DEFAULT_SYSTEM,
        max_tokens:    int  = DEFAULT_MAX_TOKENS,
        history_limit: int  = 20,
    ):
        self.provider      = provider
        self.system_prompt = system_prompt
        self.max_tokens    = max_tokens
        self.history_limit = history_limit
        self.history:  list[ChatMessage] = []
        self._latencies: list[float]     = []

        if provider == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key) if ANTHROPIC_AVAILABLE else None
            self._model  = model or DEFAULT_CLAUDE_MODEL
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if OPENAI_AVAILABLE else None
            self._model  = model or DEFAULT_GPT_MODEL

        logger.info("ChatService initialised: provider=%s model=%s", provider, self._model)

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, user_text: str, scene_context: str = "") -> str:
        """
        Send a user turn and return the assistant reply.

        Parameters
        ----------
        user_text     : transcribed user utterance
        scene_context : real-time sensor / detection context to inject
        """
        if not self._client:
            logger.error("No LLM client available.")
            return "LLM client not initialised. Check API key and dependencies."

        self.history.append(ChatMessage(role="user", content=user_text))
        system = self._build_system(scene_context)

        t0 = time.time()
        try:
            reply = self._call_api(system)
        except Exception as exc:
            logger.error("LLM API error: %s", exc)
            reply = f"I encountered an error: {exc}"

        latency = time.time() - t0
        self._latencies.append(latency)
        logger.info("LLM reply in %.2fs: '%s'", latency, reply[:80])

        self.history.append(ChatMessage(role="assistant", content=reply, tokens=len(reply.split())))
        self._prune_history()
        return reply

    def clear(self):
        """Reset conversation history (keeps system prompt)."""
        self.history.clear()
        logger.info("Conversation history cleared.")

    @property
    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.history if m.role == "assistant")

    # ── API dispatch ──────────────────────────────────────────────────────────

    def _call_api(self, system: str) -> str:
        messages = [{"role": m.role, "content": m.content}
                    for m in self.history[:-1]]   # exclude latest user turn — sent separately

        if self.provider == "anthropic":
            return self._call_anthropic(system, messages)
        else:
            return self._call_openai(system, messages)

    def _call_anthropic(self, system: str, history: list[dict]) -> str:
        messages = history + [{"role": "user", "content": self.history[-1].content}]
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text.strip()

    def _call_openai(self, system: str, history: list[dict]) -> str:
        messages = [{"role": "system", "content": system}] + \
                   history + \
                   [{"role": "user", "content": self.history[-1].content}]
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system(self, scene_context: str) -> str:
        if not scene_context:
            return self.system_prompt
        return f"{self.system_prompt}\n\nCurrent scene context:\n{scene_context}"

    def _prune_history(self):
        """Keep only the most recent history_limit turns."""
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    chat = ChatService(
        provider="anthropic",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    scene = (
        "Detected: PERSON (94%, left foreground), TABLE (91%, centre), "
        "LAPTOP (89%, on table). Battery: 78%. Speed: 0 m/s. Distance to nearest obstacle: 2.4m."
    )

    turns = [
        "What do you see in front of you?",
        "Is it safe to move forward?",
        "How is your battery?",
    ]

    for t in turns:
        print(f"\nUser : {t}")
        reply = chat.send(t, scene_context=scene)
        print(f"CHIN : {reply}")

    print(f"\nAvg latency : {chat.avg_latency:.2f}s")
    print(f"Total turns : {len(chat.history)}")
