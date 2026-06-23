## feat: add structured failure learning for agent capabilities

- added persistent `failure_reflection` memory records with `trigger`, `error`, `consequence`, `defense`, `severity`, `source`, and `timestamp`
- exposed guardrail metadata through capability snapshots, agent state/context/toolbelt payloads, and a new read-only `GET /api/agent/guardrails` route
- added read-only MCP access via `get_agent_guardrails`
- recorded conservative metadata-only failure reflections when `POST /api/agent/act` fails for an active session
- documented the guardrail contract in `docs/API-CONTRACT.md`
