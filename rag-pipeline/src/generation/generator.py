import os

import requests


class OllamaGenerator:
    def __init__(self, model="mistral", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class ClaudeGenerator:
    """
    Optional upgrade for production quality.
    Set ANTHROPIC_API_KEY in your environment or Streamlit secrets.
    """

    def __init__(self, model="claude-sonnet-4-20250514"):
        self.model = model
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Run: pip install anthropic") from exc

        api_key = os.getenv("ANTHROPIC_API_KEY") or self._streamlit_secret(
            "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required for the Claude backend. "
                "Set it as an environment variable or Streamlit secret."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    @staticmethod
    def _streamlit_secret(name):
        try:
            import streamlit as st

            return st.secrets.get(name)
        except Exception:
            return None


def get_generator(backend="ollama"):
    if backend == "ollama":
        try:
            requests.get("http://localhost:11434", timeout=2).raise_for_status()
            return OllamaGenerator()
        except Exception:
            print("Ollama not available - switching to Claude API")
            return ClaudeGenerator()
    if backend == "claude":
        return ClaudeGenerator()

    raise ValueError(f"Unknown backend: {backend}")
