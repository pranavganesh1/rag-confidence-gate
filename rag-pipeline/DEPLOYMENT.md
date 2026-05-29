# Deployment

This app is ready for Streamlit Cloud.

## Streamlit Cloud

Use these settings when creating the app:

```txt
Repository:  your-username/rag-confidence-gate
Branch:      main
Main file:   rag-pipeline/ui/app.py
```

If the repository root is `rag-pipeline`, use `ui/app.py` as the main file instead.

## Secrets

Set the Claude API key in Streamlit Cloud under:

```txt
App -> Settings -> Secrets
```

Use this secret:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Locally, the same key can be set as an environment variable:

```cmd
set ANTHROPIC_API_KEY=your-key-here
```

The app uses local Ollama when it is reachable. On Streamlit Cloud, where Ollama is not available, it falls back to Claude.
