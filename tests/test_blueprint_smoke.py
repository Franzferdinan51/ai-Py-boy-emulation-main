"""Flask test_client smoke test for the ai-game-server blueprint refactor.

Verifies that the refactored blueprints register cleanly, no routes shadow
each other, and the public endpoint surface responds with the expected
status codes + key shape fields.
"""
from __future__ import annotations

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))
os.environ.setdefault("BACKEND_PORT", "5002")
os.environ.setdefault("FLASK_ENV", "development")

import backend.server as srv  # noqa: E402  (path setup must come first)

client = srv.app.test_client()


def _hit(method: str, path: str, **kwargs):
    fn = getattr(client, method.lower())
    return fn(path, **kwargs)


results = {"ok": 0, "warn": 0, "fail": 0, "details": []}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        results["ok"] += 1
        print(f"  ✅ {name}")
    else:
        results["fail"] += 1
        print(f"  ❌ {name}  {detail}")
        results["details"].append(f"{name}: {detail}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# Each check is recorded as a pytest assertion so `pytest tests/test_blueprint_smoke.py`
# also exits cleanly. The function returns True when all checks passed; pytest sees
# the assert statements and reports per-test pass/fail automatically.
def test_blueprint_refactor():
    # All work happens at module import; we re-import the assertions here as
    # discrete checks for the pytest reporter.
    assert results["fail"] == 0, (
        f"{results['fail']} smoke checks failed:\n  " + "\n  ".join(results["details"])
    )


# --- 1. No duplicate rules (shadowed routes) ---
section("Route shadowing check")
seen = {}
shadows = []
for rule in srv.app.url_map.iter_rules():
    methods = tuple(sorted(rule.methods - {"HEAD", "OPTIONS"}))
    key = (rule.rule, methods)
    if key in seen:
        shadows.append((rule.rule, methods, seen[key].endpoint, rule.endpoint))
    else:
        seen[key] = rule
check("no duplicate (rule, method) pairs", not shadows,
      detail=f"shadows={shadows}" if shadows else "")


# --- 2. Health blueprint ---
section("Health blueprint")
r = _hit("get", "/health")
check("/health 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/health has status", data.get("status") in {"healthy", "degraded", "unhealthy"},
      detail=f"got {data.get('status')}")
check("/health has checks.flask", "flask" in (data.get("checks") or {}),
      detail=str(data.get("checks")))
check("/health has checks.pyboy", "pyboy" in (data.get("checks") or {}))
check("/health has checks.mcp", "mcp" in (data.get("checks") or {}))

for ep in ("/api/health", "/api/health/runtime", "/api/health/emulator", "/api/health/stream"):
    r = _hit("get", ep)
    check(f"{ep} 200", r.status_code == 200, f"got {r.status_code}")
    d = r.get_json() or {}
    check(f"{ep} has 'status'", "status" in d, detail=str(list(d.keys())[:3]))


# --- 3. AI models blueprint ---
section("AI models blueprint")
for ep in ("/api/providers", "/api/providers/status", "/api/models"):
    r = _hit("get", ep)
    check(f"{ep} 200", r.status_code == 200, f"got {r.status_code} body={r.data[:200]!r}")
r = _hit("get", "/api/models?provider=mock")
check("/api/models?provider=mock 200", r.status_code == 200, f"got {r.status_code}")
providers_data = (_hit("get", "/api/providers").get_json() or {})
check("/api/providers has 'providers' key", "providers" in providers_data)
check("/api/providers has 'default_provider'", "default_provider" in providers_data)


# --- 4. AI runtime blueprint ---
section("AI runtime blueprint")
r = _hit("get", "/api/ai/runtime")
check("/api/ai/runtime GET 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/ai/runtime has 'state'", "state" in data)
check("/api/ai/runtime has 'available_providers'", "available_providers" in data)

r = _hit("get", "/api/ai/settings")
check("/api/ai/settings GET 200", r.status_code == 200, f"got {r.status_code}")

r = _hit("get", "/api/openclaw/health")
check("/api/openclaw/health 200", r.status_code == 200, f"got {r.status_code}")

# --- 5. Screen blueprint ---
section("Screen blueprint")
r = _hit("get", "/api/emulator/mode")
check("/api/emulator/mode 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/emulator/mode has 'multi_process_mode'", "multi_process_mode" in data)

r = _hit("get", "/api/screen")
check("/api/screen returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code} body={r.data[:200]!r}")

r = _hit("get", "/api/stream")
check("/api/stream returns SSE without ROM", r.status_code == 200,
      f"got {r.status_code}")
check("/api/stream is text/event-stream", "text/event-stream" in r.content_type,
      detail=r.content_type)
check("/api/stream emits no-ROM error", '"error": "No ROM loaded"' in r.get_data(as_text=True))

r = _hit("get", "/api/screen/debug")
check("/api/screen/debug 503 without ROM", r.status_code in {400, 503},
      f"got {r.status_code}")

r = _hit("get", "/api/performance")
check("/api/performance 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/performance has 'server_performance'", "server_performance" in data)
check("/api/performance has 'system_info'", "system_info" in data)


# --- 6. Input blueprint ---
section("Input blueprint")
r = _hit("post", "/api/action", json={"action": "A"})
check("/api/action returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code} body={r.data[:200]!r}")

r = _hit("post", "/api/game/button", json={"button": "A"})
check("/api/game/button returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code}")

r = _hit("post", "/api/game/action", json={"action": "A"})
check("/api/game/action returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code}")

r = _hit("post", "/api/action", json={"action": "A"})
check("/api/action returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code}")

# --- 7. Already-known good blueprints ---
section("Legacy-extracted blueprints (no regressions)")
# /api/config and /api/config/validate return 503 when secure_config is not
# available (test env has no SecureConfig). The 503 + body shape matches
# baseline behavior verified by stashing the refactor and re-running.
r = _hit("get", "/api/config")
check("/api/config 503 when secure_config unavailable (baseline)",
      r.status_code == 503, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/config 503 body has 'basic_config'", "basic_config" in data,
      detail=str(list(data.keys())))

r = _hit("get", "/api/config/validate")
check("/api/config/validate 503 when secure_config unavailable (baseline)",
      r.status_code == 503, f"got {r.status_code}")

r = _hit("get", "/api/ws/status")
check("/api/ws/status 200", r.status_code == 200, f"got {r.status_code}")

r = _hit("get", "/api/tetris/status")
check("/api/tetris/status 200", r.status_code == 200, f"got {r.status_code}")


# --- 8. Agent blueprint (Stage 3 cont. round 2) ---
section("Agent blueprint")
for ep in (
    "/api/agent/state",
    "/api/agent/status",
    "/api/agent/errors",
    "/api/agent/actions",
    "/api/agent/context",
    "/api/agent/mode",
    "/api/agent/dialogue",
    "/api/agent/menu",
    "/api/agent/strategy",
):
    r = _hit("get", ep)
    check(f"{ep} 200", r.status_code == 200, f"got {r.status_code}")

r = _hit("get", "/api/agent/goal")
check("/api/agent/goal GET 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/agent/goal has 'goal' key", "goal" in data)

# POST without ROM should 400 (chat needs ROM); POST without message should 400 too
r = _hit("post", "/api/agent/chat", json={"message": "test"})
check("/api/agent/chat returns 400 without ROM", r.status_code in {400, 503},
      f"got {r.status_code} body={r.data[:200]!r}")

r = _hit("post", "/api/agent/goal", json={"goal": "test goal"})
check("/api/agent/goal POST 200", r.status_code == 200, f"got {r.status_code}")

r = _hit("post", "/api/agent/mode", json={"mode": "manual"})
check("/api/agent/mode POST 200", r.status_code == 200, f"got {r.status_code}")

r = _hit("post", "/api/agent/act", json={"action": "A"})
check("/api/agent/act returns 400 without ROM", r.status_code == 400,
      f"got {r.status_code}")


# --- 9. Spatial blueprint (Stage 3 cont. round 2) ---
section("Spatial blueprint")
for ep in (
    "/api/spatial/position",
    "/api/spatial/minimap",
    "/api/spatial/npcs",
    "/api/spatial/strategy",
):
    r = _hit("get", ep)
    check(f"{ep} 200 (empty without ROM)", r.status_code == 200,
          f"got {r.status_code} body={r.data[:200]!r}")

r = _hit("get", "/api/spatial/position")
data = r.get_json() or {}
check("/api/spatial/position has 'loaded' key", "loaded" in data)
check("/api/spatial/position has empty 'loaded=False'", data.get("loaded") is False,
      detail=f"loaded={data.get('loaded')}")


# --- 10. ROM blueprint (Stage 3 cont. round 2) ---
section("ROM blueprint")
# Read-only endpoints — work without ROM
r = _hit("get", "/api/game/state")
check("/api/game/state 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/game/state has 'rom_loaded'", "rom_loaded" in data)
check("/api/game/state has 'active_emulator'", "active_emulator" in data)

r = _hit("get", "/api/party")
check("/api/party 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/party has 'party' key", "party" in data)
check("/api/party has 'party_count'", "party_count" in data)

r = _hit("get", "/api/inventory")
check("/api/inventory 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/inventory has 'money'", "money" in data)
check("/api/inventory has 'items'", "items" in data)

r = _hit("get", "/api/memory/watch")
check("/api/memory/watch 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/memory/watch has 'addresses'", "addresses" in data)

# ROM upload without file → 400
r = _hit("post", "/api/upload-rom")
check("/api/upload-rom 400 without file", r.status_code == 400, f"got {r.status_code}")

# Legacy load_rom without body → 400
r = _hit("post", "/api/load_rom", json={})
check("/api/load_rom 400 without path or file", r.status_code == 400,
      f"got {r.status_code} body={r.data[:200]!r}")

r = _hit("post", "/save_state", json={})
check("/save_state 400 without ROM", r.status_code == 400, f"got {r.status_code}")

r = _hit("post", "/load_state", json={})
check("/load_state 400 without ROM", r.status_code == 400, f"got {r.status_code}")

# Modern /api/rom/load alias → same handler
r = _hit("post", "/api/rom/load", json={})
check("/api/rom/load 400 without path or file", r.status_code == 400,
      f"got {r.status_code}")


# --- 11. Original /api/status (still in server.py) ---
section("Server.py-resident routes")
r = _hit("get", "/api/status")
check("/api/status 200", r.status_code == 200, f"got {r.status_code}")
data = r.get_json() or {}
check("/api/status has 'ai_providers'", "ai_providers" in data)
check("/api/status has 'rom_loaded'", "rom_loaded" in data)
check("/api/status has 'active_emulator'", "active_emulator" in data)


# --- Summary ---
print("\n" + "=" * 60)
total = results["ok"] + results["fail"]
print(f"RESULTS: {results['ok']}/{total} passed")
if results["fail"]:
    print("FAILURES:")
    for d in results["details"]:
        print(f"  - {d}")
print("=" * 60)

# When run directly via `python tests/test_blueprint_smoke.py`, exit non-zero on
# failure. When imported via pytest, the `test_blueprint_refactor` function above
# is the entry point and sys.exit() would conflict with the runner.
if __name__ == "__main__":
    sys.exit(0 if results["fail"] == 0 else 1)
