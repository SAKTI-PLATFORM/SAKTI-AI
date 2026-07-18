## SAKTI-AI

FastAPI service for first-stage BI Hackathon job seeker onboarding.

Current focus:

- `POST /ml/cv/parse`
- `POST /ml/cv/extract`

Both endpoints receive CV plain text and return structured education,
experience, project, certification, and preliminary skill extraction.

LLM configuration uses the OpenAI-compatible OpenRouter API:

```env
LLM_API_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<your-openrouter-key>
LLM_MODEL=minimax/minimax-m2.5:free
```

If `LLM_API_KEY` is empty, the service falls back to deterministic local
heuristics so local development can still run.
