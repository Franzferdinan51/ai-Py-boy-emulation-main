# OpenClaw Agent - GameBoy Emulator

Control Game Boy, Game Boy Color, and Game Boy Advance games via MCP.

## Tools

- `emulator_load_rom` - Load ROM
- `emulator_press_button` - Press button  
- `emulator_press_sequence` - Press multiple buttons
- `emulator_get_frame` - Get screen image
- `emulator_get_state` - Get emulator state
- `emulator_save_state` / `emulator_load_state` - Save/load state
- Memory reading available

## Setup

```bash
# Use generic_mcp_server.py for LM Studio + OpenClaw compatibility
mcporter add pyboy-emulator --stdio "python3 $REPO/ai-game-server/generic_mcp_server.py"
```

Where `$REPO` is `/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main`

## Usage

See `AGENTS.md` and `skills/pyboy-platform/SKILL.md` for full documentation.
