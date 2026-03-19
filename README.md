# 🎮 AI GameBoy Emulator

OpenClaw Agent-powered emulation for Game Boy, Game Boy Color, and Game Boy Advance games. Now with **live streaming**, **tile-based rendering**, and **mobile support**!

## What This Is

An MCP server + web interface that lets AI agents control Game Boy emulation. Features:
- **Live Screen Streaming** - Real-time 60fps via SSE
- **Tile-Based Rendering** - Works in headless environments (no display required)
- **Mobile-Friendly** - Single proxy server serves both frontend and API
- **Memory Reading** - Position, inventory, HP, party data
- **Vision AI** - Screen analysis with dual-model routing
- **Autonomous Play** - AI makes intelligent decisions

## ✨ Latest Features

| Feature | Description |
|---------|-------------|
| **Live Streaming** | SSE-based 60fps screen streaming |
| **Tile Rendering** | Direct VRAM tile reading - works on headless servers |
| **Mobile Proxy** | Single server handles frontend + API for cross-device access |
| **Auto-Reconnect** | SSE automatically reconnects on connection loss |
| **Mobile Polling** | Fallback to polling on iOS/Android for compatibility |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Proxy Server :5173                     │
│  ┌─────────────────┐    ┌────────────────────────────┐  │
│  │  Static Files  │    │     API Proxy              │  │
│  │  (Frontend)     │───▶│  /api/* ──▶ :5002        │  │
│  │                 │    │  /health ──▶ :5002        │  │
│  └─────────────────┘    └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  PyBoy Backend :5002 │
                   │  ┌───────────────┐  │
                   │  │  Tile Renderer │  │
                   │  │  (VRAM tiles) │  │
                   │  └───────────────┘  │
                   │  ┌───────────────┐  │
                   │  │  SSE Stream   │  │
                   │  │  60fps        │  │
                   │  └───────────────┘  │
                   └─────────────────────┘
```

## Supported Systems

| System | Formats |
|--------|---------|
| Game Boy | .gb |
| Game Boy Color | .gbc |
| Game Boy Advance | .gba |

## Requirements

- Python 3.10+
- Node.js 18+ (for web UI)
- PyBoy: `pip install pyboy`
- MCP: `pip install mcp`

## Installation

```bash
# Clone repo
git clone https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
cd ai-Py-boy-emulation-main

# Install Python dependencies
cd ai-game-server
pip install -r requirements.txt

# Install Node dependencies (for web UI)
cd ../ai-game-assistant
npm install
```

## LM Studio / MCP Setup

The AI GameBoy Emulator includes an MCP server for integration with LM Studio and other AI tools.

### Add to LM Studio MCP Config

Add this to your `~/.lmstudio/mcp.json`:

```json
{
  "mcpServers": {
    "pyboy": {
      "command": "python3",
      "args": ["/path/to/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"],
      "cwd": "/path/to/ai-Py-boy-emulation-main/ai-game-server"
    }
  }
}
```

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `emulator_load_rom` | Load a ROM file |
| `emulator_press_button` | Press button (A/B/START/SELECT/UP/DOWN/LEFT/RIGHT) |
| `emulator_tick` | Advance N frames |
| `emulator_get_state` | Get emulator state |
| `get_screen_base64` | Get screen for vision AI |
| `get_memory_address` | Read memory address |
| `set_memory_address` | Write to memory address |
| `get_player_position` | Get player X,Y |
| `get_party_pokemon` | Get party stats |
| `get_inventory` | Get bag items |
| `get_map_location` | Get current map |
| `get_money` | Get player money |
| `save_game_state` | Save game |
| `load_game_state` | Load game |
| `auto_explore_mode` | Autonomous exploration |
| `auto_battle_mode` | AI battle assistant |

### Using with OpenClaw

The MCP server works with OpenClaw agents for autonomous gameplay. Configure in your OpenClaw MCP settings.

## Quick Start

### 1. Start Backend

```bash
cd ai-game-server/src
BACKEND_PORT=5002 python3 main.py
```

### 2. Start Proxy Server (serves both frontend + API)

```bash
cd ai-game-assistant
python3 proxy-server.py
```

### 3. Access the Web UI

**Desktop:** http://localhost:5173

**Mobile:** http://YOUR_MAC_IP:5173

Both use the same URL - the proxy handles routing automatically!

## Mobile Access Setup

On your phone, connect to your Mac's IP address:

```bash
# Find your Mac's IP
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Then open `http://192.168.x.x:5173` on your phone. The proxy server automatically forwards API requests to the backend.

## MCP Tools

### Core Controls

| Tool | Description |
|------|-------------|
| `emulator_load_rom` | Load a ROM file |
| `emulator_press_button` | Press GB button (A/B/START/SELECT/UP/DOWN/LEFT/RIGHT) |
| `emulator_tick` | Advance N frames |
| `emulator_get_state` | Get emulator state |

### Vision & Screen

| Tool | Description |
|------|-------------|
| `get_screen_base64` | Get screen as base64 for AI vision |

### Memory Reading

| Tool | Description |
|------|-------------|
| `get_memory_address` | Read memory at address with description |
| `set_memory_address` | Write to memory address (⚠️ can corrupt game state) |
| `get_player_position` | Get player X,Y coordinates |
| `get_party_pokemon` | Get all Pokemon in party with stats |
| `get_inventory` | Get bag items with quantities |
| `get_map_location` | Get current location |
| `get_money` | Get player money |

### Save States

| Tool | Description |
|------|-------------|
| `save_game_state` | Save game progress |
| `load_game_state` | Load saved game |

### Auto-Play Modes 🤖

| Tool | Description |
|------|-------------|
| `auto_battle_mode` | AI fights Pokemon automatically |
| `auto_explore_mode` | AI explores the world |

### Session Management

| Tool | Description |
|------|-------------|
| `session_start` | Create new gaming session |
| `session_get` | Get session data |
| `session_set` | Store session data |
| `session_list` | List all sessions |
| `session_delete` | Delete session |

## HTTP API Endpoints

### Core
- `GET /health` - Health check
- `GET /api/status` - Server status
- `GET /api/config` - Configuration

### Emulator
- `POST /api/load_rom` - Load ROM
- `POST /api/action` - Send button input
- `POST /api/ai-action` - Get AI action
- `GET /api/screen` - Get screen image (polling)
- `GET /api/stream` - SSE stream of screen updates (60fps)

### Save/Load
- `POST /api/save_state` - Save state
- `POST /api/load_state` - Load state

### Memory
- `POST /memory` - Read memory
- `GET /characters` - Get sprites
- `GET /tilemap` - Get tilemap
- `GET /sprites` - Get sprites

### AI
- `POST /api/chat` - Chat with AI
- `GET /api/providers/status` - AI provider status

## Web UI Features

- **Live Screen** - 60fps streaming via SSE (desktop) or polling (mobile)
- **Game Controls** - D-pad, A/B, Start/Select
- **Party View** - Pokemon stats, HP bars, moves
- **Inventory** - Bag contents
- **Memory Watch** - Debug watched addresses
- **OpenClaw Status** - Agent decision feed

## Tile-Based Rendering (Headless Mode)

PyBoy normally requires a display for SDL2 rendering. This project uses **direct VRAM tile reading** to render screens in headless environments:

- **Tile Map:** `0x9800-0x9BFF` (32x32 tiles, 20x18 visible)
- **Tile Data:** `0x8000-0x87FF` (128 tiles, 16 bytes each)
- **Palette:** Game Boy DMG green tones

This enables the backend to run on headless servers without X11/display.

## Troubleshooting

### Screen shows white/blank
- Backend needs display for SDL2 rendering
- Use tile-based rendering (enabled by default)
- Or use Xvfb: `xvfb-run python3 main.py`

### Mobile can't connect
- Ensure phone and Mac on same WiFi
- Check Mac's firewall allows port 5173
- Try hard refresh on phone browser

### SSE not working on iOS
- iOS Chrome has limited SSE support
- Fallback to polling (automatic on mobile)

### Backend won't start
- Check Python version (3.10+)
- Verify dependencies installed
- Check port not in use: `lsof -i :5002`

## Documentation

- [📖 API Reference](guides/API_REFERENCE.md) - Complete API docs
- [🚀 Quick Start](QUICKSTART.md) - New user guide
- [🧠 Decision Tree](guides/DECISION_TREE.md) - How AI makes decisions
- [👁️ Vision Guide](guides/VISION_GUIDE.md) - Vision AI integration
- [📝 Examples](EXAMPLES.md) - Example prompts and sessions
- [🔧 Troubleshooting](TROUBLESHOOTING.md) - Common issues

## File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/         # Python backend + MCP
│   ├── mcp_server.py       # MCP server
│   └── src/                # Flask API
│       └── backend/
│           └── emulators/
│               └── pyboy_emulator.py  # Tile-based renderer
├── ai-game-assistant/      # React web UI
│   ├── proxy-server.py    # Unified proxy (frontend + API)
│   └── services/
│       └── apiService.ts   # API client
├── skills/                 # OpenClaw skills
├── tools/                  # Agent utilities
└── guides/                 # Documentation
```

## License

MIT

---

**🦆 DuckBot is playing Pokemon Red!** - See SOUL.md for current progress
