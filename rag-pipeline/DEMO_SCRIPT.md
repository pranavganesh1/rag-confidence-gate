# Demo Script

Target length: under 2 minutes.

## Recording Flow

| Time | Action | Voiceover |
|------|--------|-----------|
| 0:00-0:15 | Load the app and upload a PDF. | This is a RAG pipeline with a confidence gate that refuses to hallucinate rather than making things up. |
| 0:15-0:45 | Ask an in-scope question and show the PASS badge, answer, and citations. | When retrieval quality is high, the gate passes and generates an answer with citations. |
| 0:45-1:15 | Ask an out-of-scope question and show the REFUSE badge. | When retrieval quality is too low, the system refuses entirely rather than fabricating an answer. The LLM is never even called. |
| 1:15-1:45 | Open the source chunks expander and point to similarity scores. | Every response is fully auditable. You can see exactly which chunks were used and their similarity scores. |
| 1:45-2:00 | Show the GitHub repo and test files. | The gate logic is independently unit tested. |

## Before Publishing

- Replace the README live app placeholder with the Streamlit Cloud URL.
- Replace the README video placeholder with the unlisted YouTube URL.
- Replace placeholder screenshots with real captures from the deployed app.
- Confirm `ANTHROPIC_API_KEY` is set in Streamlit Cloud secrets.
- Run `python -m pytest`.
