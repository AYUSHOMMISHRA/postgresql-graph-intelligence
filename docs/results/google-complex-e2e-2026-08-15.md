# Complex Real Google Gemini E2E — 2026-08-15

## Configuration

- Generation model: `gemini-3.1-flash-lite`
- Embedding model: `gemini-embedding-001`
- Embedding dimension: `3072`
- Documents: 6
- Questions: 5
- API key: loaded locally; not printed or stored

## Results

| Question | Grounded | Citations | Result |
|---|---:|---:|---|
| Which team owns the service that checkout depends on? | Yes | 2 | Identity Team; checkout and ownership citations |
| Which runbook is connected to the checkout deployment? | No | 0 | Safe abstention; indirect runbook path was not retrieved sufficiently |
| What caused the checkout login failures? | Yes | 2 | Token validation failures caused by auth-client-v4/deploy-417 |
| Which team coordinated the checkout incident response? | Yes | 1 | Identity Team |
| Which team owns the payment service? | Yes | 1 | Finance Team |

## Score

```text
Grounded answers: 4/5
Citation-valid answers: 4/5
Safe abstention: 1/5
```

## Conclusion

The system handled multi-document ownership, causal reasoning, incident
response, and an independent payment domain. The runbook question exposed a
retrieval gap for an indirectly connected fact, which should be addressed with
better graph-seed selection or retrieval expansion before claiming universal
multi-hop coverage.
