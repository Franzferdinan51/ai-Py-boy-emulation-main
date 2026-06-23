# task-run-ledger

- Added `backend.agent_features.run_ledger` with JSON-safe `v1` builders for game actions, observations, and run events.
- Extended `POST /api/agent/act` to attach an `event` field only after a successful emulator step.
- Added `GET /api/agent/runs/events` with bounded `limit` handling and safe no-ROM behavior.
- Ledger writes append to session JSONL only when an active session exists; otherwise events stay in bounded memory.

## Verification

- Red: `uv run --with pytest --with-requirements ai-game-server/requirements.txt python -m pytest tests/test_run_ledger.py -q`
- Green: `uv run --with pytest --with-requirements ai-game-server/requirements.txt python -m pytest tests/test_run_ledger.py tests/test_agent_contracts.py tests/test_blueprint_smoke.py -q`
