# AI GameBoy Emulator

OpenClaw Agent-powered emulation for Game Boy, Game Boy Color, and Game Boy Advance games.

## What This Is

An MCP server + web interface that lets AI agents control Game Boy emulation. Agents can:
- Read game memory (position, inventory, HP, etc.)
- Press buttons
- Analyze screens via vision
- Make autonomous decisions

## Supported Systems

| System | Formats |
|--------|---------|
| Game Boy | .gb |
| Game Boy Color | .gbc |
| Game Boy Advance | .gba |

## Requirements

- Python 3.10+
- PyBoy 2.7.0+
- PyGBA 0.2.4+
- Node.js 18+ (for web UI)

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

## Quick Start

### 1. Start Backend

```bash
cd ai-game-server/src
BACKEND_PORT=5002 python3 main.py
```

### 2. Start Web UI (optional)

```bash
cd ../ai-game-assistant
npx vite
```

### 3. Register MCP Server

```bash
mcporter add gameboy --stdio "python3 /path/to/mcp_server.py"
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `load_rom` | Load a ROM file |
| `press_button` | Press GB button (A/B/START/SELECT/UP/DOWN/LEFT/RIGHT) |
| `press_sequence` | Press multiple buttons |
| `get_screen` | Get screen as base64 image |
| `get_state` | Get emulator state |
| `tick` | Advance N frames |
| `save_state` | Save emulator state |
| `load_state` | Load emulator state |

## Memory Reading

Game state is readable from emulator memory:

| Data | Address |
|------|---------|
| Player X | 0xD062 |
| Player Y | 0xD063 |
| Current Map | 0xD35E |
| Money | 0xD6F5 |
| Party Count | 0xD163 |

## Agent Integration

See [AGENTS.md](AGENTS.md) for how to set up autonomous agents.

## Web UI

The web UI provides:
- Manual game control
- Agent status monitoring  
- Screen viewing
- Settings configuration

Access at http://localhost:5173 (default)

## File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/      # Python backend + MCP
│   ├── mcp_server.py    # MCP server
│   └── src/              # Flask API
├── ai-game-assistant/    # React web UI
├── skills/               # OpenClaw skills
├── tools/                # Agent utilities
└── guides/               # Documentation
```

## Troubleshooting

### Backend won't start
- Check Python version (3.10+)
- Verify dependencies installed
- Check port not in use

### MCP tools not responding
- Verify backend running
- Check mcporter registration
- Try restarting backend

### ROM won't load
- Verify ROM file exists and is readable
- Check file format supported

## License

MIT
