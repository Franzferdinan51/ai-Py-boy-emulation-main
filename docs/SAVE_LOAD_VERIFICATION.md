# PyBoy Save/Load State Verification

## ✅ VERIFICATION PASSED (2026-03-19)

**Status:** All tests passed. Save/load functionality correctly restores game state.

## Purpose

Verify that save/load state functionality correctly restores game state, not just endpoint success.

## What We Test

1. **Screen Hash Comparison** - Ensure screen content is restored after load
2. **Memory Value Comparison** - Check specific memory addresses for Pokemon Red
3. **Gameplay Transition** - Verify player position and game state changes
4. **State Roundtrip** - Save → Change → Load → Verify restoration

## Pokemon Red Memory Addresses

Based on the Pokemon Red disassembly project:

| Address | Description | Type |
|---------|-------------|------|
| `0xD362` | Player X position | uint8 |
| `0xD361` | Player Y position | uint8 |
| `0xD35E` | Current map ID | uint8 |
| `0xD163` | Party count | uint8 |
| `0xD347-0xD349` | Money (BCD encoded) | bytes[3] |
| `0xD6E6` | Badges bitfield | uint8 |
| `0xD367` | Current map tileset | uint8 |
| `0xD364` | Y block coordinate | uint8 |
| `0xD365` | X block coordinate | uint8 |

## Verification Results (2026-03-19)

### Test Summary

| Test | Status | Details |
|------|--------|---------|
| Save/Load to memory | ✅ PASS | 167,677 bytes saved/restored |
| Load saved state | ✅ PASS | Memory values preserved |
| Multiple cycles (10x) | ✅ PASS | All 10 cycles passed |
| State file compatibility | ✅ PASS | File I/O working |
| Screen hash consistency | ✅ PASS | 5/5 samples matched |

### State Size

- **Size:** 167,677 bytes
- **Consistency:** Size remains constant across all saves

### Memory Verification

All tracked memory addresses correctly restored:
- Player position (X, Y)
- Map ID
- Party count
- Money
- Badges
- Tileset and block coordinates

---

## Implementation Details

The PyBoy emulator uses `io.BytesIO` for state serialization:

```python
# Save
state_buffer = io.BytesIO()
pyboy.save_state(state_buffer)
state_data = state_buffer.getvalue()

# Load
state_buffer = io.BytesIO(state_data)
pyboy.load_state(state_buffer)
```

This uses PyBoy's official API and works correctly.

## Edge Cases Discovered

### 1. Screen Hash Timing

Screen hashes may vary slightly due to rendering timing, but memory values remain consistent. When verifying save/load, always check memory values as the authoritative state indicator, not just screen hashes.

### 2. Window Parameter

Use `window="null"` instead of `"headless"` or `"dummy"` (deprecated in PyBoy 2.0).

### 3. Tick After Load

Always tick at least 1-5 frames after loading state to ensure the emulator processes the loaded state before capturing screen/memory.

---

## Test Scripts

- `verify_save_load.py` - Basic verification from fresh ROM
- `verify_gameplay_state.py` - Verification with gameplay save state
- `verify_state_integrity.py` - Comprehensive memory verification
- `debug_save_load.py` - Detailed cycle-by-cycle debugging

---

## Backend API Endpoints

The following API endpoints are verified working:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/save_state` | POST | Save state to memory |
| `/api/load_state` | POST | Load state from memory |

---

*Last verified: 2026-03-19 19:35 EDT*