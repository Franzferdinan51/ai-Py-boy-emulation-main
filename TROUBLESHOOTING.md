# 🔧 TROUBLESHOOTING.md - Common Issues and Solutions

**Fixes for common problems with AI GameBoy Emulator**

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| MCP not registered | `mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"` |
| Emulator won't start | Check PyBoy installed: `pip install pyboy` |
| Black screen | ROM not loaded - call `emulator_load_rom` first |
| Buttons not working | Check emulator running: `emulator_get_state` |

---

## Installation Issues

### PyBoy Not Installed

**Symptom:**
```
ModuleNotFoundError: No module named 'pyboy'
```

**Fix:**
```bash
# Install PyBoy
pip install pyboy

# Or install with Cython for better performance
cd PyBoy
pip install -r requirements.txt
make build_python
pip install -e .
```

### MCP Library Not Installed

**Symptom:**
```
ModuleNotFoundError: No module named 'mcp'
```

**Fix:**
```bash
# Install MCP library
pip install mcp

# Or install from requirements
cd ai-game-server
pip install -r requirements.txt
```

### Missing Dependencies

**Symptom:**
```
ImportError: missing dependency
```

**Fix:**
```bash
# Install all dependencies
cd ai-game-server
pip install -r requirements.txt

# Common dependencies:
# - flask
# - pillow
# - numpy
# - pyboy
# - mcp
```

---

## MCP Server Issues

### MCP Not Registered

**Symptom:**
```
Tool not found: emulator_press_button
```

**Fix:**
```bash
# Check current registration
mcporter list | grep gameboy

# Remove and re-register
mcporter remove gameboy
mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"
```

### Server Won't Start

**Symptom:**
```
Address already in use: 5002
```

**Fix:**
```bash
# Check what's using the port
lsof -i :5002

# Kill the process or use different port
python3 ai-game-server/mcp_server.py --port 5003
```

### Server Crashes on Start

**Symptom:**
```
Server starts then immediately exits
```

**Debug:**
```bash
# Run with debug output
python3 -v ai-game-server/mcp_server.py

# Check logs
tail -f ai-game-server/ai_game_server.log
```

---

## Emulator Issues

### No ROM Loaded

**Symptom:**
```
{"success": false, "error": "No ROM loaded"}
```

**Fix:**
```json
{
  "tool": "emulator_load_rom",
  "args": {"rom_path": "/path/to/rom.gb"}
}
```

### Invalid ROM File

**Symptom:**
```
Invalid ROM file or unsupported format
```

**Fix:**
- Check file exists: `ls -la rom.gb`
- Verify ROM format: Must be .gb, .gbc, or .gba
- Check ROM is valid: Try loading in different emulator
- Ensure file is not corrupted: Re-download ROM

### Emulator Freezes

**Symptom:**
- Screen doesn't update
- Buttons have no effect

**Fix:**
```json
{
  "tool": "emulator_get_state",
  "args": {}
}
```
If stuck:
1. Load last save: `load_game_state("last-good")`
2. Restart server: Kill and restart `mcp_server.py`
3. Check game file: Try different ROM

---

## Memory Reading Issues

### Wrong Position Data

**Symptom:**
- Position always returns same values
- Incorrect map ID

**Cause:** Memory addresses vary by game

**Fix:** 
- Verify game version (Red/Blue/Yellow have different addresses)
- Check game compatibility
- Use vision as fallback

### Party Info Empty

**Symptom:**
```
{"party": []}
```

**Cause:** Not in party screen, or game-specific memory layout

**Fix:**
- Navigate to Pokemon menu in game
- Verify Pokemon game (only Pokemon games have party)
- Check game state: `emulator_get_state()`

### Memory Read Errors

**Symptom:**
```
Memory read failed at address 0xD000
```

**Fix:**
- Address may be invalid for current game
- Use higher-level functions instead: `get_party_info()`, `get_money()`
- Check game is loaded and running

---

## Vision Issues

### Screen Capture Returns Empty

**Symptom:**
```
{"screen": ""}
```

**Fix:**
1. Verify ROM loaded: `emulator_get_state()`
2. Check emulator running
3. Try: `emulator_tick(1)` then retry

### Base64 Decoding Fails

**Symptom:**
```
Failed to decode base64 image
```

**Fix:**
- Server may be starting up - wait and retry
- Check disk space
- Restart server

---

## Save/Load Issues

### Save File Not Found

**Symptom:**
```
Save not found: my-save
```

**Fix:**
```json
{
  "tool": "emulator_list_saves",
  "args": {}
}
```
Shows all available saves.

### Save Created But Can't Load

**Symptom:**
```
Error loading save file
```

**Causes:**
- File corrupted
- Different ROM version
- Permissions issue

**Fix:**
- Check file exists: `ls -la saves/`
- Create new save
- Check disk space

---

## Session Issues

### Session Not Found

**Symptom:**
```
Session not found: session_123
```

**Fix:**
```json
{
  "tool": "session_list",
  "args": {}
}
```
Shows all sessions.

### Session Data Lost

**Symptom:**
- Session variables empty
- Progress not saved

**Fix:**
- Check session_id is correct
- Session may have expired - create new one
- Save important data to file

---

## Button Input Issues

### Invalid Button

**Symptom:**
```
Invalid button: X
```

**Fix:** Use valid buttons: `A`, `B`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `START`, `SELECT`

### Button Sequence Too Long

**Symptom:**
```
Sequence too long
```

**Fix:** Break into smaller sequences:
```json
{"sequence": "DOWN DOWN DOWN"}
```
becomes:
```json
{"sequence": "DOWN"}
```
repeated 3 times

---

## Performance Issues

### Slow Response

**Symptom:**
- `timing_ms` > 500ms
- Lag between action and result

**Fix:**
- Use memory reading instead of vision when possible
- Reduce screenshot quality
- Close other applications
- Use SSD for saves

### High CPU Usage

**Symptom:**
- Computer runs hot
- Slow performance

**Fix:**
- Reduce emulator speed: `emulator_set_speed(0.5)`
- Don't run vision on every frame
- Use frame skipping

---

## Debug Tips

### Enable Debug Logging

```bash
# Start server with debug
python3 ai-game-server/mcp_server.py --log-level DEBUG

# Or set environment
export LOG_LEVEL=DEBUG
python3 ai-game-server/mcp_server.py
```

### Check Server Status

```json
{
  "tool": "emulator_get_state",
  "args": {}
}
```

Returns:
- ROM loaded
- Emulator running
- Frame count
- Speed

### Manual Emulator Control

```bash
# Open Python REPL
cd ai-game-server/src
python3

# Test manually
from backend.server import app
# ... manual testing
```

### View Server Logs

```bash
# Real-time logs
tail -f ai-game-server/ai_game_server.log

# Last 50 lines
tail -50 ai-game-server/ai_game_server.log

# Search for errors
grep -i error ai-game-server/ai_game_server.log
```

---

## Frequently Asked Questions

### Q: How do I restart the emulator?

**A:**
```json
{
  "tool": "emulator_reset",
  "args": {}
}
```
Then reload ROM.

### Q: Can I play GBA games?

**A:** Yes! The emulator supports .gb, .gbc, and .gba formats.

### Q: How do I save my progress?

**A:**
```json
{
  "tool": "save_game_state",
  "args": {"save_name": "my-save"}
}
```

### Q: The game runs too fast/slow

**A:**
```json
{
  "tool": "emulator_set_speed",
  "args": {"speed": 1.0}
}
```
1.0 = normal, 2.0 = double speed, 0.5 = half speed

### Q: How do I get the Pokemon Red memory addresses?

**A:** Memory addresses are in `AGENTS.md`. They differ for each game version (Red vs Blue vs Yellow).

### Q: Can I use my own ROM file?

**A:** Yes! Just provide the full path:
```json
{
  "tool": "emulator_load_rom",
  "args": {"rom_path": "/Users/you/roms/pokemon-red.gb"}
}
```

### Q: Why isn't auto_battle working?

**A:** Ensure:
1. You're in a battle (check vision)
2. Party has Pokemon
3. Emulator is running

### Q: How do I stop the emulator?

**A:**
```json
{
  "tool": "emulator_stop",
  "args": {}
}
```
Or just stop the MCP server process.

---

## Getting More Help

### Check All Tools Available

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python3 ai-game-server/mcp_server.py
```

### Verify Setup

```bash
# Run verification script
python3 ai-game-server/verify_complete_setup.py
```

### Check System Requirements

```bash
python3 ai-game-server/system_diagnostic.py
```

---

## Related Documentation

- [API_REFERENCE.md](guides/API_REFERENCE.md) - All MCP endpoints
- [DECISION_TREE.md](guides/DECISION_TREE.md) - Decision making
- [EXAMPLES.md](EXAMPLES.md) - Example prompts and sessions
- [README.md](README.md) - Main documentation