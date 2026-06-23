# task-real-save-load

- Replaced placeholder save/load route behavior with a shared real implementation for `/api/save_state`, `/api/load_state`, `/save_state`, and `/load_state`.
- Added in-memory named slots per active emulator with validation and `quick_save` as the default slot.
- Preserved compatibility response fields while returning `name` and `bytes`.
- Added contract tests for named slots, alias coverage, invalid-name rejection, and real emulator save/load calls.
- Verification:
  - `uv run --with pytest --with-requirements ai-game-server/requirements.txt python -m pytest tests/test_save_load_contract.py -q`
  - `uv run --with pytest --with-requirements ai-game-server/requirements.txt python -m pytest tests/test_save_load_contract.py tests/test_agent_contracts.py tests/test_blueprint_smoke.py -q`
