#!/bin/bash

# WebUI Integration Test Script
# Tests all the newly wired endpoints

BACKEND_URL="${BACKEND_URL:-http://localhost:5002}"

echo "🧪 Testing WebUI Integration"
echo "Backend URL: $BACKEND_URL"
echo "======================================"

# Test 1: Health check
echo -e "\n1️⃣  Testing health check..."
curl -s "$BACKEND_URL/health" | jq '.' || echo "❌ Health check failed"

# Test 2: Game state
echo -e "\n2️⃣  Testing game state..."
curl -s "$BACKEND_URL/api/game/state" | jq '.' || echo "❌ Game state failed"

# Test 3: Agent status
echo -e "\n3️⃣  Testing agent status..."
curl -s "$BACKEND_URL/api/agent/status" | jq '.' || echo "❌ Agent status failed"

# Test 4: Party endpoint
echo -e "\n4️⃣  Testing party endpoint..."
PARTY_RESPONSE=$(curl -s "$BACKEND_URL/api/party")
echo "$PARTY_RESPONSE" | jq '.' || echo "❌ Party endpoint failed"

# Test 5: Inventory endpoint
echo -e "\n5️⃣  Testing inventory endpoint..."
INVENTORY_RESPONSE=$(curl -s "$BACKEND_URL/api/inventory")
echo "$INVENTORY_RESPONSE" | jq '.' || echo "❌ Inventory endpoint failed"

# Test 6: Agent mode GET
echo -e "\n6️⃣  Testing agent mode (GET)..."
curl -s "$BACKEND_URL/api/agent/mode" | jq '.' || echo "❌ Agent mode GET failed"

# Test 7: Agent mode POST
echo -e "\n7️⃣  Testing agent mode (POST)..."
curl -s -X POST "$BACKEND_URL/api/agent/mode" \
  -H "Content-Type: application/json" \
  -d '{"mode":"manual","enabled":false,"autonomous_level":"moderate"}' | jq '.' || echo "❌ Agent mode POST failed"

# Test 8: Screen capture
echo -e "\n8️⃣  Testing screen capture..."
curl -s -o /tmp/test_screen.png "$BACKEND_URL/api/screen" && echo "✅ Screen captured to /tmp/test_screen.png" || echo "❌ Screen capture failed"

# Test 9: Memory watch
echo -e "\n9️⃣  Testing memory watch..."
curl -s "$BACKEND_URL/api/memory/watch" | jq '.' || echo "❌ Memory watch failed"

echo -e "\n======================================"
echo "✅ All tests completed!"
echo ""
echo "Next steps:"
echo "1. Start backend: cd ai-game-server/src && python3 main.py"
echo "2. Start frontend: cd ai-game-assistant && npm run dev"
echo "3. Open browser: http://localhost:5173"
echo "4. Load a ROM and test all panels"
