# AI Game Assistant

React + Vite frontend for the PyBoy operator stack.

## Current frontend shape

- Active entrypoint: `ai-game-assistant/index.tsx` -> `ai-game-assistant/App.tsx`
- Compatibility surface: `ai-game-assistant/src/WebUiApp.tsx`
- Shared UI components: `ai-game-assistant/src/components/`
- Shared API client: `ai-game-assistant/services/apiService.ts`

The modern `App.tsx` shell is the primary OpenClaw operator UI. `WebUiApp.tsx`
is still kept working for compatibility and now shares the same read-only
`AgentRunPanel` component.

## Canonical local ports

- Frontend/proxy: `http://localhost:5173`
- Backend API: `http://localhost:5002`
- WebSocket stream: `ws://localhost:5003/`

## Quick start

### 1. Start the backend

```bash
cd ../ai-game-server
PYTHONPATH="$PWD/src" python3 -c "from backend.server import app; app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)"
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Run the frontend

```bash
npm run dev
```

### 4. Open the UI

```text
http://localhost:5173
```

## Operator surfaces

- Live game screen + stream status
- OpenClaw runtime settings and objective sync
- Shared `AgentRunPanel` backed by `GET /api/agent/runs/events`
- Party, inventory, memory, minimap, NPC, and strategy panels

## Useful commands

```bash
npm test -- --run AgentRunPanel.test.tsx GameCanvas.test.tsx
npm run build
```

## Notes

- Default backend URL is `http://localhost:5002`
- The run operator panel is intentionally read-only
- Frontend refresh cadence is shared; the panel does not create its own polling loop
