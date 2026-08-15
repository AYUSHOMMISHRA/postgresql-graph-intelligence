# Real OpenAI Four-Question E2E — 2026-08-15

## Configuration

- Provider: OpenAI
- Embedding dimension: `1536`
- Database: isolated PostgreSQL test database
- Namespace: `openai-four-question-e2e`
- API key: loaded from the local environment; not printed or stored

## Results

| Question | Grounded | Citations | Result |
|---|---:|---:|---|
| Which team owns the dependency of checkout-service? | Yes | 2 | Identity Team; citations from `ownership-auth#0` and `incident-2026-041#0` |
| Which runbook is connected to the checkout deployment? | No | 0 | Safe abstention: insufficient evidence |
| What deployment caused the checkout incident? | Yes | 1 | `deploy-417`; citation `incident-2026-041#0` |
| Which service does checkout-service depend on? | Yes | 1 | `auth-service`; citation `incident-2026-041#0` |

## Score

```text
Grounded answers: 3/4
Original multi-hop question: passed
Citation validation: passed for 3 questions
Safe abstention: passed for 1 question
```

The runbook question abstained because the real-provider retrieval/answer path
did not produce a grounded citation-valid answer on this run. This result should
not be reported as a 4/4 pass without a separate reproducible rerun.
