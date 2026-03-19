# 🚀 QUICKSTART.md - Get Started with AI-PyBoy

Welcome! This guide will get you running with AI-powered Game Boy emulation in minutes.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| OS | macOS, Linux, or Windows (WSL) |

---

## Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
cd ai-Py-boy-emulation-main

# Install Python dependencies
cd ai-game-server
pip install -r requirements.txt

# Install Node dependencies (for web UI)
cd ../ai-game-assistant
npm install
```

---

## Step 2: Configure AI

Create `ai-game-server/.env`:

```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-your-key-here

# Option 2: Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Option 3: Local (LM Studio)
OPENAI_API_BASE=http://localhost:1234/v1
```

---

## Step 3: Start the Backend

```bash
cd ai-game-server/src
BACKEND_PORT=5002 python3 main.py
```

You should see:
```
✅ AI Game Server running on http://localhost:5002
```

---

## Step 4: Start Web UI (Optional)

```bash
cd ../ai-game-assistant
npx vite
```

Open http://localhost:5173 in your browser.

---

## Step 5: Register MCP Server

```bash
mcporter add gameboy --stdio "python3 /full/path/to/ai-game-server/mcp_server.py"
```

Verify it's working:
```bash
mcporter list | grep gameboy
```

---

## First Game: Pokemon Red

### Load a ROM

Place a Pokemon Red ROM at `roms/pokemon-red.gb`, then:

```bash
# Via MCP
echo '{"method":"tools/call","params":{"name":"emulator_load_rom","arguments":{"rom_path":"roms/pokemon-red.gb"}}}' | python3 ai-game-server/mcp_server.py

# Or via HTTP
curl -X POST http://localhost:5002/api/load_rom \
  -H "Content-Type: application/json" \
  -d '{"rom_path":"roms/pokemon-red.gb"}'
```

### Press Buttons

```bash
# Press A button
curl -X POST http://localhost:5002/api/action \
  -H "Content-Type: application/json" \
  -d '{"button":"A"}'
```

Valid buttons: `A`, `B`, `START`, `SELECT`, `UP`, `DOWN`, `LEFT`, `RIGHT`

### Get Game State

```bash
# Get screen
curl http://localhost:5002/api/screen

# Get party
curl http://localhost:5002/api/status
```

---

## Using AI to Play

### Start a Session

```json
{
  "method": "tools/call",
  "params": {
    "name": "session_start",
    "arguments": {"goal": "Beat the Elite 4"}
  }
}
```

### Let AI Play

```json
{
  "method": "tools/call",
  "params": {
    "name": "auto_battle",
    "arguments": {"max_moves": 10}
  }
}
```

### Other Auto-Play Modes

| Command | Use Case |
|---------|----------|
| `auto_battle` | Fight Pokemon automatically |
| `auto_explore` | Explore the world |
| `auto_grind` | Grind for XP |
| `auto_catch` | Catch wild Pokemon |
| `auto_item_use` | Use items wisely |

---

## Common Tasks

### How do I...


| Task | Solution |
|------|----------|
| See the screen? | `GET /api/screen` or open web UI |
| See my Pokemon? | `get_party_info` MCP tool |
| See my items? | `get_inventory` MCP tool |
| See my position? | `get_player_position` MCP tool |
| Save my game? | `save_game_state` with save name |
| Load a save? | `load_game_state` with save name |
| Make the AI play? | Use `auto_battle`, `auto_explore`, etc. |

---

## Troubleshooting

### "Port 5002 in use"
```bash
# Find and kill the process
lsof -i :5002
kill -9 <PID>
```

### "ROM not found"
- Make sure the ROM file exists
- Use absolute path, not relative

### "MCP tools not responding"
- Restart the backend: `cd ai-game-server/src && python3 main.py`
- Check mcporter: `mcporter list`

---

## Next Steps

1. 📖 Read [API_REFERENCE.md](guides/API_REFERENCE.md) for all endpoints
2. 🧠 Check [DECISION_TREE.md](guides/DECISION_TREE.md) for AI logic
3. 👁️ See [VISION_GUIDE.md](guides/VISION_GUIDE.md) for vision AI
4. 📝 Browse [EXAMPLES.md](EXAMPLES.md) for prompts

---

## Quick Reference

```bash
# Start backend
cd ai-game-server/src && python3 main.py

# Start web UI
cd ai-game-assistant && npx vite

# Test MCP server
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python3 ../ai-game-server/mcp_server.py

# Load ROM (HTTP)
curl -X POST http://localhost:5002/api/load_rom -d '{"rom_path":"path/to/rom.gb"}'

# Press button (HTTP)
curl -X POST http://localhost:5002/api/action -d '{"button":"A"}'

# Get screen (HTTP)
curl http://localhost:5002/api/screen -o screen.png
```

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.

**Happy gaming!** 🎮