# OpenClaw Provider Setup Guide

**Date:** 2026-03-19  
**Status:** ✅ Complete and Pushed to GitHub

---

## Overview

The PyBoy Emulation project now uses **OpenClaw as the default AI provider**, enabling seamless integration with your OpenClaw Gateway for unified model access (Bailian, LM Studio, local models, etc.).

---

## What Changed

### Before:
- Default provider: `mock` (fake responses)
- No indication which provider was active
- OpenClaw integration not visible in UI
- Confusing placeholder states

### After:
- Default provider: `openclaw` (real AI via OpenClaw Gateway)
- Clear provider status in UI header
- OpenClaw connection status prominently displayed
- Accurate runtime information

---

## Quick Start

### 1. Start OpenClaw Gateway

Make sure OpenClaw Gateway is running:

```bash
openclaw gateway status
# If not running:
openclaw gateway start
```

Gateway should be at: `http://localhost:18789`

### 2. Start Backend Server

```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server
python3 start_server.py
```

You should see in logs:
```
INFO: Using OpenClaw provider - local MCP integration
INFO: Successfully initialized openclaw provider
```

### 3. Start Frontend

```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-assistant
npm run dev
```

Open http://localhost:5173

### 4. Verify Provider Status

In the UI header, you should see:
- ✅ **Backend** (green pill)
- ✅ **OpenClaw** (green pill)
- ✅ **AI Provider (openclaw)** (green pill)

In the OpenClaw Runtime panel:
- **AI Provider:** OpenClaw (Native)
- **Vision Model:** kimi-k2.5 (or your configured model)

---

## How It Works

### Provider Priority Order

The backend tries providers in this order:

1. **openclaw** (priority 1) ← NEW DEFAULT
2. gemini (priority 2) - requires API key
3. openrouter (priority 3) - requires API key
4. openai-compatible (priority 4) - requires API key
5. nvidia (priority 5) - requires API key
6. tetris-genetic (priority 10) - specialized
7. mock (priority 99) ← LAST RESORT ONLY

### OpenClaw Provider Features

- **No API Key Required** - Uses local MCP
- **Auto-Detection** - Connects to `http://localhost:18789` by default
- **Model Agnostic** - Works with any OpenClaw model (Bailian, LM Studio, etc.)
- **Vision Support** - Can analyze game screens via OpenClaw vision models
- **Automatic Fallback** - If OpenClaw unavailable, tries other providers

---

## Configuration

### Environment Variables (Optional)

```bash
# Change default provider (usually not needed)
export DEFAULT_AI_PROVIDER=openclaw

# Custom OpenClaw endpoint (if not on default port)
export OPENCLAW_MCP_ENDPOINT=http://localhost:18789
```

### Frontend Settings

In the WebUI Settings modal:
- **Backend URL:** `http://localhost:5002` (default)
- **OpenClaw Endpoint:** `http://localhost:18789` (default)
- **Vision Model:** `kimi-k2.5` or `qwen-vl-plus`

These settings are saved to localStorage and persist across sessions.

---

## Testing

### Test Script

```bash
./tools/test-provider-fixes.sh
```

### Manual Tests

1. **Check Provider Status API:**
```bash
curl http://localhost:5002/api/providers/status
```

Expected response:
```json
{
  "openclaw": {
    "status": "available",
    "priority": 1,
    "error": null,
    "available": true
  },
  "mock": {
    "status": "available",
    "priority": 99,
    "error": null,
    "available": true
  }
}
```

2. **Check Agent Status:**
```bash
curl http://localhost:5002/api/agent/status
```

Look for:
```json
{
  "provider": "openclaw",
  "vision_model": "kimi-k2.5",
  "enabled": true,
  ...
}
```

3. **Test AI Action:**
```bash
curl -X POST http://localhost:5002/api/ai-action \
  -H "Content-Type: application/json" \
  -d '{"goal": "Explore the map", "action_history": []}'
```

Check backend logs to see which provider was used.

---

## Troubleshooting

### Issue: UI Shows "AI Provider (mock)"

**Cause:** OpenClaw Gateway not running or not reachable.

**Fix:**
1. Start OpenClaw Gateway: `openclaw gateway start`
2. Check endpoint: `curl http://localhost:18789/session/status`
3. Restart backend server
4. Refresh frontend

### Issue: Backend Logs Show "OpenClaw not available"

**Cause:** Wrong endpoint or OpenClaw not configured.

**Fix:**
```bash
# Check OpenClaw status
openclaw gateway status

# Set correct endpoint
export OPENCLAW_MCP_ENDPOINT=http://localhost:18789

# Restart backend
```

### Issue: AI Actions Not Working

**Cause:** No provider available or OpenClaw models not configured.

**Fix:**
1. Check OpenClaw models: `openclaw session status`
2. Configure models in OpenClaw (Bailian, LM Studio, etc.)
3. Check backend logs for provider errors
4. Verify `/api/providers/status` shows at least one available provider

---

## Benefits

### 1. Unified Billing
All AI usage goes through OpenClaw, which manages:
- Alibaba Bailian (FREE unlimited for MiniMax-M2.5, kimi-k2.5)
- LM Studio (FREE local)
- Other providers via OpenClaw MCP

### 2. Better UX
- Clear indication of which provider is active
- No confusing "mock" provider by default
- Real-time provider status in UI

### 3. Automatic Fallback
If OpenClaw is unavailable:
1. Tries Gemini (if API key configured)
2. Tries OpenRouter (if API key configured)
3. Tries other providers
4. Falls back to mock (last resort only)

### 4. OpenClaw-Native Features
- Uses OpenClaw's model orchestration
- Leverages OpenClaw's vision capabilities
- Integrates with OpenClaw session management
- Respects OpenClaw's model priorities

---

## Architecture

```
┌─────────────┐
│   WebUI     │
│  (React)    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Backend   │
│   (Flask)   │
└──────┬──────┘
       │ MCP
       ▼
┌─────────────┐
│ OpenClaw    │
│  Gateway    │
└──────┬──────┘
       │
       ├───▶ Bailian (kimi-k2.5, MiniMax-M2.5, etc.)
       ├───▶ LM Studio (local models)
       └───▶ Other MCP providers
```

---

## Files Modified

### Backend:
- `ai-game-server/src/backend/ai_apis/ai_provider_manager.py`
- `ai-game-server/src/backend/ai_apis/openclaw_ai_provider.py` (NEW)

### Frontend:
- `ai-game-assistant/App.tsx`
- `ai-game-assistant/services/apiService.ts`

### Documentation:
- `PROVIDER-FIXES-SUMMARY.md`
- `docs/OPENCLAW-PROVIDER-SETUP.md` (this file)
- `tools/test-provider-fixes.sh` (NEW)

---

## Next Steps

1. ✅ Backend provider manager updated
2. ✅ OpenClaw provider created
3. ✅ Frontend displays provider status
4. ✅ Pushed to GitHub
5. 🔄 Test with actual gameplay
6. 🔄 Monitor provider usage
7. 🔄 Adjust model priorities as needed

---

## Support

For issues or questions:
1. Check backend logs: `ai-game-server/logs/`
2. Check frontend console: Browser DevTools
3. Verify OpenClaw status: `openclaw gateway status`
4. Review provider docs: `PROVIDER-FIXES-SUMMARY.md`

---

**Last Updated:** 2026-03-19 12:15 EDT  
**Developer:** DuckBot (Subagent: Provider + OpenClaw Native Fixes)  
**GitHub:** https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
