#!/bin/bash
# Launcher for Generic Game Boy MCP Server with Pokemon Red auto-loaded

ROM_PATH="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
PYTHON="/opt/homebrew/opt/python@3.14/bin/python3.14"
SERVER_DIR="/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server"

cd "$SERVER_DIR"

# The MCP server needs to load ROM via tool call, but we can pass it as env var
export GB_DEFAULT_ROM="$ROM_PATH"

exec "$PYTHON" "$SERVER_DIR/generic_mcp_server.py"
