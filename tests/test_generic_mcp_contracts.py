import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "ai-game-server" / "generic_mcp_server.py"


class _StubTool:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class _StubTextContent:
    def __init__(self, type, text):
        self.type = type
        self.text = text


class _StubImageContent:
    def __init__(self, type, data, mimeType):
        self.type = type
        self.data = data
        self.mimeType = mimeType


class _StubServer:
    def __init__(self, name):
        self.name = name
        self._tools = []
        self._call_tool = None

    def list_tools(self):
        def decorator(fn):
            self._tools = fn
            return fn

        return decorator

    def call_tool(self):
        def decorator(fn):
            self._call_tool = fn
            return fn

        return decorator


def _load_generic_mcp_module():
    module_name = "generic_mcp_server_contract_test"
    for key in [module_name, "mcp", "mcp.server", "mcp.server.stdio", "mcp.types"]:
        sys.modules.pop(key, None)

    server_mod = types.ModuleType("mcp.server")
    server_mod.Server = _StubServer
    server_mod.__path__ = []

    stdio_mod = types.ModuleType("mcp.server.stdio")
    stdio_mod.stdio_server = lambda: None

    types_mod = types.ModuleType("mcp.types")
    types_mod.Tool = _StubTool
    types_mod.TextContent = _StubTextContent
    types_mod.ImageContent = _StubImageContent

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.__path__ = []
    mcp_mod.server = server_mod
    mcp_mod.types = types_mod

    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.stdio"] = stdio_mod
    sys.modules["mcp.types"] = types_mod

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def generic_mcp_module():
    return _load_generic_mcp_module()


def test_core_tools_use_canonical_backend_routes(monkeypatch, generic_mcp_module):
    calls = []

    class Response:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    def fake_get(url, timeout=10):
        calls.append(("GET", url, None, timeout))
        if url.endswith("/api/screen"):
            return Response({"image": "img", "pyboy_frame": 7, "shape": [144, 160, 3]})
        return Response({"ok": True})

    def fake_post(url, json=None, timeout=10):
        calls.append(("POST", url, json, timeout))
        return Response({"ok": True})

    monkeypatch.setattr(generic_mcp_module.requests, "get", fake_get)
    monkeypatch.setattr(generic_mcp_module.requests, "post", fake_post)
    monkeypatch.setattr(generic_mcp_module, "DEFAULT_BACKEND_URL", "http://backend.test")

    async def invoke(tool_name, arguments):
        return await generic_mcp_module.call_tool(tool_name, arguments)

    asyncio.run(invoke("load_rom", {"rom_path": "/roms/test.gb"}))
    asyncio.run(invoke("get_state", {}))
    asyncio.run(invoke("get_agent_context", {}))
    asyncio.run(invoke("get_agent_toolbelt", {}))
    asyncio.run(invoke("get_agent_routines", {}))
    asyncio.run(invoke("act_and_observe", {"action": "A", "frames": 3}))
    asyncio.run(invoke("get_screen", {}))
    asyncio.run(invoke("tick", {"frames": 5}))
    asyncio.run(invoke("save_state", {"name": "slot-1"}))
    asyncio.run(invoke("load_state", {"name": "slot-1"}))
    asyncio.run(invoke("quick_save", {}))
    asyncio.run(invoke("quick_load", {}))

    assert calls == [
        ("POST", "http://backend.test/api/load_rom", {"path": "/roms/test.gb"}, 10),
        ("GET", "http://backend.test/api/game/state", None, 10),
        ("GET", "http://backend.test/api/agent/context", None, 10),
        ("GET", "http://backend.test/api/agent/toolbelt", None, 10),
        ("GET", "http://backend.test/api/agent/routines", None, 10),
        ("POST", "http://backend.test/api/agent/act", {"action": "A", "frames": 3}, 10),
        ("GET", "http://backend.test/api/screen", None, 10),
        ("POST", "http://backend.test/api/game/button", {"button": "NOOP", "frames": 5}, 10),
        ("POST", "http://backend.test/api/save_state", {"name": "slot-1"}, 10),
        ("POST", "http://backend.test/api/load_state", {"name": "slot-1"}, 10),
        ("POST", "http://backend.test/api/save_state", {"name": "quick_save"}, 10),
        ("POST", "http://backend.test/api/load_state", {"name": "quick_save"}, 10),
    ]


def test_tool_descriptions_mark_access_level(generic_mcp_module):
    tools = {tool.name: tool for tool in asyncio.run(generic_mcp_module.list_tools())}

    assert "[read-only]" in tools["get_state"].description
    assert "[read-only]" in tools["get_screen"].description
    assert "[read-only]" in tools["get_agent_toolbelt"].description
    assert "[read-only]" in tools["get_agent_routines"].description
    assert "[mutating]" in tools["load_rom"].description
    assert "[mutating]" in tools["save_state"].description
    assert "[mutating]" in tools["quick_save"].description


def test_backend_error_text_is_returned(monkeypatch, generic_mcp_module):
    class Response:
        status_code = 500

        def json(self):
            return {"error": "backend failed"}

    monkeypatch.setattr(generic_mcp_module.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(generic_mcp_module, "DEFAULT_BACKEND_URL", "http://backend.test")

    result = asyncio.run(generic_mcp_module.call_tool("save_state", {"name": "slot-1"}))
    assert json.loads(result[0].text) == {
        "error": "backend failed",
        "status_code": 500,
    }
