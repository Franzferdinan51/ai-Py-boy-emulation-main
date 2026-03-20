# Floating Chat Agent Instruction Flow

**Last Updated:** March 19, 2026 21:45 EST

## Overview

The floating chat in the AI-Py-Boy platform is designed to be **agent-first**, not cosmetic. It provides a direct interface for users to:

1. **Chat with the AI** about game state
2. **Set agent goals/tasks** via natural language instructions
3. **Query agent state** through the chat interface

## Instruction Patterns

The floating chat detects special prefixes to map instructions to agent state:

| Pattern | Action | Example |
|---------|--------|---------|
| `goal:` or `objective:` | Sets `current_goal` | `goal: Beat the Elite Four` |
| `task:` or `do:` | Sets `current_task` | `task: Go to the Pokemon Center` |
| `?status` or `?state` | Returns agent status | `?status` |
| Plain message | Regular AI chat | `What should I do next?` |

## Backend Agent State Mapping

### Existing Endpoints (used by floating chat)

| Endpoint | Purpose | Used By |
|----------|---------|---------|
| `GET /api/agent/state` | Full agent state snapshot | Chat status queries |
| `GET /api/agent/status` | Agent status summary | Quick status |
| `GET /api/agent/goal` | Get current goal/task | Status display |
| `POST /api/agent/goal` | Set goal/task | Instruction parsing |

### New Endpoint

**`POST /api/agent/chat`** — Agent-aware chat that can set goals

```json
// Request
{
  "message": "goal: Catch a legendary Pokemon",
  "api_name": "openclaw",
  "model": "bailian/kimi-k2.5"
}

// Response
{
  "ok": true,
  "intent_detected": "goal",
  "goal_updated": "Catch a legendary Pokemon",
  "task_updated": null,
  "chat_response": "I've updated your goal to: Catch a legendary Pokemon. I'll help you find and catch a legendary Pokemon. The available legendaries in this game are...",
  "agent_state": {
    "mode": "autonomous",
    "enabled": true,
    "current_goal": "Catch a legendary Pokemon",
    "current_task": "",
    "last_action": null
  },
  "timestamp": "2026-03-19T21:45:00Z"
}
```

## Frontend Integration

### Chat Message Types

```typescript
interface ChatMessage {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  intent?: 'goal' | 'task' | 'query' | 'chat';
  goal_updated?: string;
  task_updated?: string;
  timestamp: string;
}
```

### Detection Logic (Frontend)

```typescript
function detectIntent(message: string): { intent: string; value: string } {
  const lower = message.toLowerCase().trim();
  
  if (lower.startsWith('goal:') || lower.startsWith('objective:')) {
    return { intent: 'goal', value: message.substring(5).trim() };
  }
  if (lower.startsWith('task:') || lower.startsWith('do:')) {
    return { intent: 'task', value: message.substring(5).trim() };
  }
  if (lower === '?status' || lower === '?state' || lower === 'status' || lower === 'state') {
    return { intent: 'query', value: '' };
  }
  return { intent: 'chat', value: message };
}
```

### Backend Parsing Logic

The backend parses the message to detect intent:

```python
INTENT_PATTERNS = {
    'goal': ['goal:', 'objective:', 'my goal is ', 'i want to ', 'go for '],
    'task': ['task:', 'do: ', 'please ', 'can you '],
    'query': ['?status', '?state', 'status', 'state', 'what are you doing']
}

def parse_instruction(message: str) -> dict:
    """Parse message to detect instruction intent"""
    lower = message.lower().strip()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if lower.startswith(pattern):
                return {
                    'intent': intent,
                    'value': message[len(pattern):].strip()
                }
    
    return {'intent': 'chat', 'value': message}
```

## Agent State Flow

```
User Input: "goal: Beat the Elite Four"
        │
        ▼
┌───────────────────┐
│  /api/agent/chat │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Parse Intent    │ ──→ { intent: 'goal', value: 'Beat the Elite Four' }
└────────┬──────────┘
         │
    ┌────┴────┐
    │ intent  │
    │ detected│
    └────┬────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────────┐              ┌──────────────┐
│ goal/task   │              │   Regular    │
│ detected    │              │    chat      │
└─────┬───────┘              └──────┬───────┘
      │                             │
      ▼                             ▼
┌─────────────┐              ┌──────────────┐
│ POST        │              │ POST         │
│ /api/agent/ │              │ /api/chat    │
│ goal        │              │ (for AI     │
└─────┬───────┘              │ response)   │
      │                      └──────┬───────┘
      ▼                             │
┌─────────────┐                      │
│ Update      │                      │
│ agent_state │                      │
│ {goal, task}│                      │
└─────┬───────┘                      │
      │                              │
      └──────────┬───────────────────┘
                 ▼
        ┌────────────────┐
        │ Return combined│
        │ response       │
        └────────────────┘
```

## Response Format

### Success Response

```json
{
  "ok": true,
  "intent_detected": "goal",
  "goal_updated": "Beat the Elite Four",
  "task_updated": null,
  "chat_response": "I've updated your goal to: Beat the Elite Four. To beat the Elite Four, you'll need a strong team...",
  "agent_state": {
    "mode": "autonomous",
    "enabled": true,
    "current_goal": "Beat the Elite Four",
    "current_task": "",
    "last_action": "A",
    "last_action_time": "2026-03-19T21:44:55Z",
    "recent_actions": [
      {"action": "A", "timestamp": "2026-03-19T21:44:55Z"},
      {"action": "RIGHT", "timestamp": "2026-03-19T21:44:30Z"}
    ]
  },
  "timestamp": "2026-03-19T21:45:00Z"
}
```

### Query Response (status/state)

```json
{
  "ok": true,
  "intent_detected": "query",
  "chat_response": "Current agent status:\n- Mode: autonomous\n- Goal: Beat the Elite Four\n- Task: (none)\n- Last action: A (15 seconds ago)\n- Total actions: 42",
  "agent_state": {
    "mode": "autonomous",
    "enabled": true,
    "current_goal": "Beat the Elite Four",
    "current_task": "",
    "last_action": "A",
    "last_action_time": "2026-03-19T21:44:55Z"
  },
  "timestamp": "2026-03-19T21:45:00Z"
}
```

### Error Response

```json
{
  "ok": false,
  "error": "No ROM loaded",
  "chat_response": null,
  "timestamp": "2026-03-19T21:45:00Z"
}
```

## Integration with Existing Systems

### MCP Tools

When using LM Studio with `generic_mcp_server.py`, add this tool:

```python
# In generic_mcp_server.py
@tool
def agent_chat(message: str, api_name: str = "openclaw", model: str = None) -> dict:
    """
    Send a message to the agent chat that can set goals/tasks.
    
    Use prefixes to control agent behavior:
    - "goal: <text>" - Set the agent's current goal
    - "task: <text>" - Set the agent's current task
    - "?status" - Query agent status
    - Plain text - Regular chat with AI
    """
    response = requests.post(
        f"{BACKEND_URL}/api/agent/chat",
        json={"message": message, "api_name": api_name, "model": model}
    )
    return response.json()
```

### Agent Autonomy Levels

The floating chat respects the agent's autonomy level:

| Autonomy | Behavior |
|----------|----------|
| `passive` | Chat only, no goal changes take effect |
| `moderate` | Goals set but require confirmation |
| `aggressive` | Full goal/task control via chat |

## Testing the Flow

### Test 1: Set Goal

```bash
curl -X POST http://localhost:5002/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "goal: Catch a Mewtwo"}'
```

### Test 2: Query Status

```bash
curl -X POST http://localhost:5002/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "?status"}'
```

### Test 3: Regular Chat

```bash
curl -X POST http://localhost:5002/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What Pokemon should I use?"}'
```

## Migration Path

1. **Phase 1** (this PR): Add `/api/agent/chat` endpoint
2. **Phase 2**: Update frontend chat to use new endpoint
3. **Phase 3**: Add MCP tool wrapper
4. **Phase 4**: Add autonomy level filtering

## Backward Compatibility

- Existing `/api/chat` endpoint continues to work
- New `/api/agent/chat` is additive
- Frontend can gradually migrate
- No breaking changes to existing contracts