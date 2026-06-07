"""
llm_client.py — Groq LLM wrapper with multi-key rotation and retry logic.
Reads GROQ_API_KEYS from .env (comma-separated list).
"""

import os
import time
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Parse comma-separated keys
_raw_keys = os.getenv("GROQ_API_KEYS", "")
_API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

if not _API_KEYS:
    raise EnvironmentError(
        "No GROQ_API_KEYS found in .env. "
        "Set GROQ_API_KEYS=key1,key2,... in your .env file."
    )

DEFAULT_MODEL   = "llama-3.3-70b-versatile"
DEFAULT_TEMP    = 0.2
MAX_TOKENS_REPORT = 1500
MAX_TOKENS_QUERY  = 900
MAX_RETRIES     = len(_API_KEYS)   # one attempt per key


class LLMClient:
    """
    Groq LLM wrapper.
    Automatically rotates through all available API keys on rate-limit errors.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model   = model
        self.keys    = list(_API_KEYS)
        self._idx    = 0   # current key index

    def _get_client(self) -> Groq:
        return Groq(api_key=self.keys[self._idx % len(self.keys)])

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS_QUERY,
        temperature: float = DEFAULT_TEMP,
    ) -> str:
        """
        Call the LLM with key rotation on rate-limit (429) errors.
        Returns the response text, or raises RuntimeError if all keys exhausted.
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e).lower()
                last_error = e

                if "rate_limit" in err_str or "429" in err_str:
                    # Rotate to next key
                    self._idx = (self._idx + 1) % len(self.keys)
                    time.sleep(1.0)
                    continue
                else:
                    # Non-rate-limit error — re-raise immediately
                    raise

        raise RuntimeError(
            f"All {len(self.keys)} Groq API keys exhausted. Last error: {last_error}"
        )
