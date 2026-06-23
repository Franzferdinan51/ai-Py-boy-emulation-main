# Agent Platform Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a stable OpenClaw-first game agent platform with real save/load,
versioned action traces, MCP parity, and a modern operator surface.

**Architecture:** PyBoy remains the only frame and state authority. Small backend
contract modules use the existing Flask route-registration pattern, while React
consumes those contracts through the shared API client.

**Tech Stack:** Flask, PyBoy, pytest, MCP Python SDK, React, TypeScript, Vite,
Vitest, SSE.

---

### Task 1: Establish the Canonical Contract Matrix

**Files:**
- Create: `tests/test_agent_contracts.py`
- Modify: `tests/test_blueprint_smoke.py`
- Create: `docs/API-CONTRACT.md`

- [ ] **Step 1: Write failing canonical-route tests.**

```python
def test_canonical_game_contracts_are_registered(client):
    for method, path in [
        ("get", "/api/game/state"),
        ("get", "/api/agent/context"),
        ("post", "/api/agent/act"),
        ("post", "/api/save_state"),
        ("post", "/api/load_state"),
    ]:
        response = getattr(client, method)(path, json={} if method == "post" else None)
        assert response.status_code != 404
```

- [ ] **Step 2: Run `pytest tests/test_agent_contracts.py -q` and record failures.**
- [ ] **Step 3: Document canonical routes, aliases, stream fields, named slots, and MCP mappings.**
- [ ] **Step 4: Run `pytest tests/test_agent_contracts.py tests/test_blueprint_smoke.py -q`.**
- [ ] **Step 5: Commit `test: define canonical game agent contracts`.**

### Task 2: Make Every Save/Load Route Real and Named

**Files:**
- Modify: `ai-game-server/src/backend/routes/save_load.py`
- Modify: `ai-game-server/src/backend/routes/__init__.py`
- Create: `tests/test_save_load_contract.py`

- [ ] **Step 1: Write a failing fake-emulator test.**

```python
def test_legacy_and_api_save_load_share_named_real_state(app_client, fake_emulator):
    assert app_client.post("/api/save_state", json={"name": "checkpoint"}).status_code == 200
    assert app_client.post("/load_state", json={"name": "checkpoint"}).status_code == 200
    assert fake_emulator.saved == 1
    assert fake_emulator.loaded == 1
```

- [ ] **Step 2: Run `pytest tests/test_save_load_contract.py -q`; it must fail before implementation.**
- [ ] **Step 3: Implement one validated named-slot helper used by canonical and legacy routes.**
- [ ] **Step 4: Run `pytest tests/test_save_load_contract.py tests/test_blueprint_smoke.py -q`.**
- [ ] **Step 5: Commit `fix: make all save state routes restore real slots`.**

### Task 3: Add Versioned Observation and Run Events

**Files:**
- Create: `ai-game-server/src/backend/agent_features/run_ledger.py`
- Modify: `ai-game-server/src/backend/routes/agent.py`
- Modify: `ai-game-server/src/backend/agent_features/__init__.py`
- Create: `tests/test_run_ledger.py`

- [ ] **Step 1: Write a failing action-event test.**

```python
def test_agent_act_records_versioned_event(client, fake_emulator):
    response = client.post("/api/agent/act", json={"action": "A", "frames": 1, "source": "mcp"})
    event = response.get_json()["event"]
    assert event["version"] == "v1"
    assert event["action"]["action"] == "A"
    assert event["observation"]["version"] == "v1"
```

- [ ] **Step 2: Run `pytest tests/test_run_ledger.py -q`; it must fail before implementation.**
- [ ] **Step 3: Add bounded in-memory plus JSONL append-only events and `GET /api/agent/runs/events`.**
- [ ] **Step 4: Run `pytest tests/test_run_ledger.py tests/test_agent_contracts.py -q`.**
- [ ] **Step 5: Commit `feat: record versioned game agent run events`.**

### Task 4: Align Generic MCP Tools With Canonical APIs

**Files:**
- Modify: `ai-game-server/generic_mcp_server.py`
- Modify: `tests/test_mcp.py`
- Modify: `docs/API-CONTRACT.md`

- [ ] **Step 1: Write a failing route-mapping test.**

```python
def test_quick_save_uses_canonical_route(monkeypatch):
    calls = []
    monkeypatch.setattr(generic_mcp_server, "api_post", lambda path, data: calls.append((path, data)) or {})
    asyncio.run(generic_mcp_server.call_tool("quick_save", {}))
    assert calls == [("/api/save_state", {"name": "quick_save"})]
```

- [ ] **Step 2: Run `pytest tests/test_mcp.py -q`; it must expose mismatches.**
- [ ] **Step 3: Route buttons, actions, screen, save/load, and context through canonical APIs only.**
- [ ] **Step 4: Run `pytest tests/test_mcp.py -q`.**
- [ ] **Step 5: Commit `fix: align MCP core tools with backend contracts`.**

### Task 5: Repair the Live Stream and Shared Frontend Client

**Files:**
- Modify: `ai-game-assistant/src/WebUiApp.tsx`
- Modify: `ai-game-assistant/src/components/GameCanvas.tsx`
- Modify: `ai-game-assistant/src/services/apiService.ts`
- Create: `ai-game-assistant/src/components/GameCanvas.test.tsx`
- Modify: `ai-game-assistant/package.json`

- [ ] **Step 1: Write a failing SSE decoding test.**

```tsx
it("renders a server frame without a synthetic type field", async () => {
  emitMessage({ image: "ZmFrZQ==", frame: 12, fps: 60, timestamp: "2026-06-22T00:00:00Z" });
  expect(drawImage).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run `npm test -- --run GameCanvas.test.tsx`; it must fail before implementation.**
- [ ] **Step 3: Normalize frame parsing, reconnect state, cleanup, and the default backend to `http://localhost:5002`.**
- [ ] **Step 4: Run `npm test -- --run GameCanvas.test.tsx && npm run build`.**
- [ ] **Step 5: Commit `fix: stabilize live game stream contract`.**

### Task 6: Add the Compact Agent Run Panel

**Files:**
- Create: `ai-game-assistant/src/components/AgentRunPanel.tsx`
- Modify: `ai-game-assistant/src/components/index.ts`
- Modify: `ai-game-assistant/src/WebUiApp.tsx`
- Modify: `ai-game-assistant/src/services/apiService.ts`
- Create: `ai-game-assistant/src/components/AgentRunPanel.test.tsx`

- [ ] **Step 1: Write a failing run-panel test.**

```tsx
it("shows the active goal and the most recent action", () => {
  render(<AgentRunPanel state={state} events={[event]} onApprove={vi.fn()} />);
  expect(screen.getByText("Reach Pewter City")).toBeVisible();
  expect(screen.getByText("A")).toBeVisible();
});
```

- [ ] **Step 2: Run `npm test -- --run AgentRunPanel.test.tsx`; it must fail before implementation.**
- [ ] **Step 3: Implement a responsive null-safe panel using existing visual tokens and the current polling path.**
- [ ] **Step 4: Run `npm test -- --run AgentRunPanel.test.tsx && npm run build`.**
- [ ] **Step 5: Commit `feat: add agent run operator panel`.**

### Task 7: Integrate and Verify the Running Stack

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Document ports `5002`, `5003`, `5173`, canonical save/load, and MCP smoke commands.**
- [ ] **Step 2: Run `pytest tests/test_blueprint_smoke.py tests/test_agent_contracts.py tests/test_save_load_contract.py tests/test_run_ledger.py tests/test_mcp.py -q`.**
- [ ] **Step 3: Run `npm run build` in `ai-game-assistant`.**
- [ ] **Step 4: With a legal ROM, verify load, multiple stream events, named-state restore, and identical MCP calls.**
- [ ] **Step 5: Commit docs and push `master`.**

## Self-Review

- The plan covers contracts, real save/load, action traces, MCP parity, streaming,
  the operator UI, docs, and live verification.
- Tasks 2 through 6 have disjoint primary write scopes after Task 1.
- No task creates a second tick loop or direct MCP emulator access.
