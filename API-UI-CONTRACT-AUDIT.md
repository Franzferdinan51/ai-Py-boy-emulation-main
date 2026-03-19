# API/UI Contract Audit Report

**Date:** 2026-03-19  
**Auditor:** DuckBot Sub-Agent  
**Scope:** Backend server.py ↔ Frontend apiService.ts/App.tsx

---

## Summary

This audit identifies mismatches between backend endpoints and frontend expectations. The primary issues are:

1. **Port mismatch** between frontend service defaults
2. **Agent mode value mismatch** - frontend sends values backend doesn't recognize
3. **Inconsistent naming conventions** between API responses and frontend types

---

## Critical Issues

### 1. Port Mismatch 🔴 CRITICAL

| Component | Default Port |
|-----------|-------------|
| Backend (`server.py`) | `5002` (from `BACKEND_PORT` env or hardcoded) |
| Frontend `apiService.ts` | `5000` (`DEFAULT_BASE_URL = 'http://localhost:5000'`) |
| Frontend `App.tsx` | `5002` (`backendUrl: 'http://localhost:5002'`) |

**Impact:** Frontend `apiService.ts` will fail to connect if using its default.

**Fix:** Update `apiService.ts` to use port `5002` as default.

---

### 2. Agent Mode Value Mismatch 🔴 CRITICAL

**Frontend sends:**
```typescript
mode: 'auto' | 'manual'
```

**Backend expects (`VALID_AGENT_MODES`):**
```python
['idle', 'auto_explore', 'auto_battle', 'auto_train', 'auto_fish', 
 'auto_walk', 'auto_center', 'auto_shop', 'speedrun', 'manual']
```

**Impact:** Frontend sending `mode: 'auto'` will return 400 error with "Invalid mode".

**Fix:** Backend should accept 'auto' as an alias for 'auto_explore' or frontend should send valid mode.

---

### 3. Memory Watch Response Shape Mismatch 🟡 MODERATE

**Frontend expects (`MemoryWatch`):**
```typescript
interface MemoryWatch {
  addresses: MemoryAddress[];  // { address: number, name: string, size: number }[]
  values: MemoryValue[];       // { address: number, name: string, size: number, value: number | null, hex: string }[]
  timestamp: string;
}
```

**Backend returns:**
```python
{
  "addresses": [...],  # List of {address, name, size}
  "values": [...],     # List of {...addr_info, value, hex}
  "timestamp": "..."
}
```

**Issue:** The `values` array in backend includes all fields from `addresses` plus `value` and `hex`, but frontend `MemoryValue` expects `address` as a field. Backend returns `addr_info` spread which may not include `address` directly in each value object.

**Fix:** Ensure backend `values` array includes explicit `address` field.

---

## Moderate Issues

### 4. Agent Status Mode Field Inconsistency 🟡 MODERATE

**Frontend expects:**
```typescript
mode: 'auto' | 'manual'
```

**Backend returns:**
```python
agent_mode_state["mode"]  # Could be any of VALID_AGENT_MODES
```

**Fix:** Normalize backend response to return simplified mode for frontend.

---

### 5. Game Button Response Missing Timestamp 🟢 MINOR

**Frontend calls:** `POST /api/game/button`

**Backend returns:**
```python
{
  "success": success,
  "button": button,
  "timestamp": datetime.now().isoformat()
}
```

**Frontend expects:** `{ success: boolean; button: string }`

**Status:** ✅ Backend returns more than expected, which is fine.

---

### 6. Load ROM Endpoint Inconsistency 🟡 MODERATE

**Frontend uses:**
- `POST /api/upload-rom` (for file upload)
- `POST /api/rom/load` (for URL/path - defined but not used in App.tsx)

**Backend has:**
- `POST /api/upload-rom` - handles file upload
- `POST /api/rom/load` - handles path-based loading
- `POST /api/load_rom` - legacy endpoint

**Status:** ⚠️ Redundant endpoints. Keep `upload-rom` for files, `rom/load` for paths.

---

## Minor Issues

### 7. Health Check Endpoint Path 🟢 MINOR

**Frontend doesn't have a dedicated health check** but backend has `/health`.

**Recommendation:** Frontend should use `/health` for connection status instead of `/api/game/state`.

---

### 8. Screen Endpoint Response Shape 🟢 MINOR

**Backend `/api/screen` returns:**
```python
{
  "image": img_base64,
  "shape": [...],
  "timestamp": time.time(),
  "performance": {...},
  ...
}
```

**Frontend expects:** `Blob` (from `response.blob()`)

**Status:** ✅ Frontend handles this correctly by calling `response.blob()`.

---

## Endpoint Mapping

| Frontend Call | Backend Endpoint | Status |
|--------------|------------------|--------|
| `GET /api/game/state` | ✅ Line 3832 | OK |
| `GET /api/agent/status` | ✅ Line 3856 | OK |
| `POST /api/game/button` | ✅ Line 3880 | OK |
| `GET /api/memory/watch` | ✅ Line 3918 | NEEDS FIX |
| `GET /api/screen` | ✅ Line 2215 | OK |
| `POST /api/upload-rom` | ✅ Line 1078 | OK |
| `GET /api/save_state` | ✅ Line 2494 | OK |
| `POST /api/load_state` | ✅ Line 2519 | OK |
| `GET /api/party` | ✅ Line 4229 | OK |
| `GET /api/inventory` | ✅ Line 4388 | OK |
| `GET /api/agent/mode` | ✅ Line 4646 | NEEDS FIX |
| `POST /api/agent/mode` | ✅ Line 4570 | NEEDS FIX |
| `GET /api/memory/<address>` | ✅ Line 3975 | OK |
| `POST /api/memory/<address>` | ✅ Line 4043 | OK |
| `GET /api/screen/stream` | ✅ Line 4537 | OK |
| `POST /api/rom/load` | ✅ Line 4448 | OK |
| `POST /api/chat` | ✅ Line 1764 | OK |
| `POST /api/ai-action` | ✅ Line 1431 | OK |

---

## Recommended Fixes

### Fix 1: Update apiService.ts Default Port
```typescript
// Change from:
const DEFAULT_BASE_URL = 'http://localhost:5000';
// To:
const DEFAULT_BASE_URL = 'http://localhost:5002';
```

### Fix 2: Update Backend VALID_AGENT_MODES to Include 'auto'
```python
VALID_AGENT_MODES = [
    "idle",
    "auto",           # NEW: Alias for auto_explore (for frontend compatibility)
    "auto_explore",
    "auto_battle",
    ...
]
```

### Fix 3: Add Mode Normalization in Agent Status Response
```python
# In /api/agent/status endpoint:
def get_agent_status_api():
    # Normalize mode for frontend
    simplified_mode = "auto" if agent_mode_state["mode"] not in ["idle", "manual"] else agent_mode_state["mode"]
    return jsonify({
        ...
        "mode": simplified_mode,  # Frontend-friendly mode
        "actual_mode": agent_mode_state["mode"],  # Actual mode for advanced use
        ...
    })
```

### Fix 4: Ensure Memory Values Include Address Field
```python
# In /api/memory/watch endpoint:
for addr_info in watched_addresses:
    memory_data.append({
        "address": addr_info["address"],  # Explicit address field
        **addr_info,
        "value": value,
        "hex": hex(value)
    })
```

---

## Files Modified

1. `ai-game-assistant/services/apiService.ts` - Fix default port
2. `ai-game-server/src/backend/server.py` - Fix agent mode handling
3. `ai-game-assistant/types/api.ts` - Add type for agent mode response

---

## Testing Checklist

- [ ] Frontend connects to backend on port 5002
- [ ] Agent mode 'auto' is accepted by backend
- [ ] Agent status returns 'auto' or 'manual' in mode field
- [ ] Memory watch values include address field
- [ ] All endpoints return expected response shapes
- [ ] Error messages are helpful and consistent