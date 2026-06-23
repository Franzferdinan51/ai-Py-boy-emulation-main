# task-hermes-capabilities

- added `backend.agent_features.agent_capabilities` as an adapter-first Hermes-inspired capability layer backed by existing session and memory state
- extended `GET /api/agent/state` and `GET /api/agent/context` with additive capability metadata: `active_session_id`, `active_routine`, `available_tools`, `memory_summary`, and `next_recommended_action`
- added read-only capability endpoints `GET /api/agent/toolbelt` and `GET /api/agent/routines`
- added `POST /api/agent/routines` for session-scoped playbook metadata only; no emulator authority changed
- exposed the new read-only capability routes through `generic_mcp_server.py` as `get_agent_toolbelt` and `get_agent_routines`
- documented the new contract in `docs/API-CONTRACT.md`
- verification:
  - `uv run --with pytest --with-requirements ai-game-server/requirements.txt python -m pytest tests/test_agent_capabilities.py tests/test_agent_contracts.py tests/test_generic_mcp_contracts.py tests/test_run_ledger.py -q`
