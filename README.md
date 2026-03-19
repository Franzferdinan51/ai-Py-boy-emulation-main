# 🎮 AI GameBoy Emulator

**42 MCP tools** for autonomous Game Boy emulation. Works with **ANY** GB/GBC/GBA game!

## What This Is

An MCP server + web interface for AI agents to control Game Boy emulation. Features:
- **Live Streaming** - Real-time 60fps via SSE
- **42 MCP Tools** - Comprehensive controls for any game
- **Mobile Support** - Single URL works on desktop + mobile
- **Memory Access** - Read/write game RAM
- **Vision AI** - Screen analysis ready

## ⚡ Quick Start

```bash
# 1. Start backend
cd ai-game-server/src
BACKEND_PORT=5002 python3 main.py

# 2. Start proxy (serves frontend + API)
cd ../ai-game-assistant
python3 proxy-server.py

# 3. Open http://localhost:5173
```

## 🔧 LM Studio Setup

Add to `~/.lmstudio/mcp.json`:

```json
{
  "mcpServers": {
    "gameboy": {
      "command": "/opt/homebrew/opt/python@3.14/bin/python3.14",
      "args": ["/path/to/ai-Py-boy-emulation-main/ai-game-server/generic_mcp_server.py"]
    }
  }
}
```

## 📋 MCP Tools (42 Total)

### 🎮 Buttons (10)
| Tool | Description |
|------|-------------|
| `press_a` | Press A |
| `press_b` | Press B |
| `press_up` | Press UP |
| `press_down` | Press DOWN |
| `press_left` | Press LEFT |
| `press_right` | Press RIGHT |
| `press_start` | Press START |
| `press_select` | Press SELECT |
| `press_button_combo` | Combo (UP+A) |
| `hold_button` | Hold N frames |

### 📺 Screen (5)
| Tool | Description |
|------|-------------|
| `get_screen` | Current screen |
| `screenshot` | Screenshot |
| `tick` | Advance frames |
| `compare_screens` | Detect changes |
| `get_tile_data` | VRAM tiles |

### 🎯 Game State (7)
| Tool | Description |
|------|-------------|
| `get_state` | Emulator state |
| `get_game_info` | Game info |
| `get_system_info` | ROM header |
| `save_state` | Save |
| `load_state` | Load |
| `quick_save` | Quick save |
| `quick_load` | Quick load |

### 🐭 Pokemon (8)
| Tool | Description |
|------|-------------|
| `get_party` | Party Pokemon |
| `get_inventory` | Items |
| `get_position` | X,Y position |
| `get_map` | Current map |
| `get_money` | Money |
| `get_badges` | Badges |
| `get_wild_pokemon` | Wild Pokemon |
| `get_enemy_info` | Enemy info |

### 🎲 Generic - ANY Game (5)
| Tool | Description |
|------|-------------|
| `get_health` | HP |
| `get_score` | Score |
| `get_level` | Level/Area |
| `get_lives` | Lives |
| `get_game_time` | Timer |

### 💾 Memory (6)
| Tool | Description |
|------|-------------|
| `get_memory` | Read address |
| `read_ram` | Read RAM range |
| `write_ram` | Write RAM |
| `search_ram` | Search RAM |
| `list_save_slots` | List saves |
| `load_rom` | Load ROM |

## 🎮 Supported Games

**ANY Game Boy game works!**

| System | Formats |
|--------|---------|
| Game Boy | .gb |
| Game Boy Color | .gbc |
| Game Boy Advance | .gba |

### Examples
- ✅ Pokemon Red/Blue/Yellow
- ✅ Super Mario
- ✅ Legend of Zelda
- ✅ Tetris
- ✅ Any .gb, .gbc, .gba!

## 📱 Mobile Access

Connect from phone to same URL:
```
Desktop: http://localhost:5173
Mobile:  http://YOUR_IP:5173
```

Proxy server handles routing automatically.

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│  Proxy Server :5173            │
│  ┌─────────┐  ┌────────────┐  │
│  │Frontend │──│API Proxy   │  │
│  └─────────┘  └─────┬──────┘  │
└──────────────────────┼──────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Backend :5002  │
              │ ┌────────────┐ │
              │ │Tile Renderer│ │
              │ │SSE Stream  │ │
              │ └────────────┘ │
              └────────────────┘
```

## 📁 File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/
│   ├── src/
│   │   ├── backend/           # Flask API
│   │   │   └── emulators/
│   │   │       └── pyboy_emulator.py
│   │   └── server.py
│   └── generic_mcp_server.py   # MCP server (42 tools)
├── ai-game-assistant/
│   ├── proxy-server.py       # Unified proxy
│   ├── App.tsx               # React frontend
│   └── services/apiService.ts
└── README.md
```

## 🔧 Troubleshooting

### Screen shows white
- Use tile-based rendering (default)
- Or use Xvfb: `xvfb-run python3 main.py`

### Mobile can't connect
- Phone must be on same network
- Try: `ifconfig | grep "inet "` to get IP
- Check firewall allows port 5173

### MCP not working in LM Studio
- Restart LM Studio
- Check Python path: `/opt/homebrew/opt/python@3.14/bin/python3.14`

## 📚 Docs

- [API Reference](guides/API_REFERENCE.md)
- [Quick Start](QUICKSTART.md)
- [Vision Guide](guides/VISION_GUIDE.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## 🦆 DuckBot

DuckBot uses this for autonomous Pokemon Red gameplay via OpenClaw.

**GitHub:** https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
