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
    Set ANTHROPIC_API_KEY in your environment to use this.
    """

    def __init__(self, model="claude-sonnet-4-20250514"):
        self.model = model
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Run: pip install anthropic") from exc

        self.client = anthropic.Anthropic()

    def generate(self, prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


def get_generator(backend="ollama"):
    if backend == "ollama":
        return OllamaGenerator()
    if backend == "claude":
        return ClaudeGenerator()

    raise ValueError(f"Unknown backend: {backend}")
