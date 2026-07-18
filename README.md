## SAKTI-AI

FastAPI service for first-stage BI Hackathon job seeker onboarding.

Current focus:

- `POST /ml/cv/parse-file` accepts a `multipart/form-data` PDF upload, extracts
  its text on the server, and returns structured CV data.
- `POST /ml/cv/parse` remains available for internal plain-text parsing.

The PDF upload is limited to 10 MB. Education, experience, project,
certification, and preliminary skill extraction are returned as arrays, so a
CV can contain multiple records of each type.

LLM configuration uses the OpenAI-compatible DeepSeek API:

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=<your-deepseek-api-key>
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_TOKENS=8000
DEEPSEEK_TIMEOUT_SECONDS=120
```

Use `deepseek-reasoner` as `DEEPSEEK_MODEL` when reasoning mode is needed. If
`DEEPSEEK_API_KEY` is empty, the service falls back to deterministic local
heuristics so local development can still run.
