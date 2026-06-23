Added a shared read-only `AgentCapabilityPanel` to both frontend surfaces.

What changed:
- New compact panel renders toolbelt, routines, skill drafts, planner hints, and learned guardrails.
- Frontend API service now fetches `/api/agent/toolbelt` and `/api/agent/routines`.
- Both `App.tsx` and `src/WebUiApp.tsx` refresh the capability snapshots through existing status/live refreshes.

Verification:
- `npm test -- --run AgentCapabilityPanel.test.tsx AgentRunPanel.test.tsx GameCanvas.test.tsx`
- `npm run build`
