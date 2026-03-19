#!/bin/bash
# =============================================================================
# DuckBot Gaming Agent Spawn Script
# Spawns autonomous gaming agents for Game Boy emulation
# =============================================================================

set -e

# Configuration
REPO_DIR="/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main"
MCP_SERVER="$REPO_DIR/ai-game-server/mcp_server.py"
ROMS_DIR="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# DuckBot ASCII art
DUCKBOT_ASCII="
    ___ 
   ( _ \ 
  / /_\ \ 
  \____/ 
   🦆🦆  
   DuckBot
   Gaming Agent
"

echo -e "${CYAN}$DUCKBOT_ASCII${NC}"
echo ""

# Check if MCP server is registered
check_mcp() {
    echo -e "${YELLOW}Checking MCP server registration...${NC}"
    if mcporter list 2>/dev/null | grep -q "duckbot-emulator"; then
        echo -e "${GREEN}✓ MCP server 'duckbot-emulator' is registered${NC}"
        return 0
    else
        echo -e "${YELLOW}! MCP server not registered. Registering now...${NC}"
        register_mcp
    fi
}

# Register MCP server
register_mcp() {
    echo -e "${BLUE}Registering DuckBot MCP server...${NC}"
    mcporter add duckbot-emulator --stdio "python3 $MCP_SERVER"
    echo -e "${GREEN}✓ MCP server registered${NC}"
}

# List available ROMs
list_roms() {
    echo -e "\n${YELLOW}Available ROMs:${NC}"
    if [ -d "$ROMS_DIR" ]; then
        ls -1 "$ROMS_DIR" | grep -E "\.(gb|gbc|gba)$" | nl
    else
        echo "  No ROMs directory found at $ROMS_DIR"
    fi
}

# Spawn gaming agent using sessions_spawn
spawn_agent() {
    local game="${1:-pokemon-red.gb}"
    local model="${2:-bailian/kimi-k2.5}"
    local task="${3:-Play Pokemon Red autonomously. Your goal is to beat the Elite 4. Make strategic decisions, manage resources, and save progress frequently.}"
    
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  🦆 DUCKBOT GAMING AGENT SPAWN${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Game:${NC}     $game"
    echo -e "${CYAN}Model:${NC}    $model"
    echo -e "${CYAN}Task:${NC}     $task"
    echo ""
    
    # Check MCP registration
    check_mcp
    
    # Verify ROM exists
    if [ ! -f "$ROMS_DIR/$game" ]; then
        echo -e "${RED}Error: ROM not found: $ROMS_DIR/$game${NC}"
        list_roms
        exit 1
    fi
    
    echo -e "${GREEN}✓ Ready to spawn agent${NC}"
    echo ""
    echo -e "${YELLOW}To spawn the agent, use sessions_spawn:${NC}"
    echo ""
    echo "  sessions_spawn \\"
    echo "    --task \"$task\" \\"
    echo "    --model $model \\"
    echo "    --agentId duckbot-gaming"
    echo ""
    echo -e "${BLUE}Or use the MCP tools directly:${NC}"
    echo ""
    echo "  # Load ROM"
    echo "  mcporter call duckbot-emulator.emulator_load_rom rom_path=\"$ROMS_DIR/$game\""
    echo ""
    echo "  # Start playing"
    echo "  mcporter call duckbot-emulator.emulator_press_sequence sequence=\"W W W START\""
    echo ""
    echo "  # Get screen for vision"
    echo "  mcporter call duckbot-emulator.emulator_get_frame include_base64=true"
    echo ""
}

# Run autonomous agent loop
run_autonomous() {
    local game="${1:-pokemon-red.gb}"
    local model="${2:-bailian/kimi-k2.5}"
    local max_turns="${3:-50}"
    
    echo -e "${GREEN}Starting autonomous DuckBot gameplay...${NC}"
    echo "Game: $game"
    echo "Model: $model"
    echo "Max turns: $max_turns"
    echo ""
    
    # Verify MCP
    check_mcp
    
    # Verify ROM
    if [ ! -f "$ROMS_DIR/$game" ]; then
        echo -e "${RED}Error: ROM not found: $ROMS_DIR/$game${NC}"
        exit 1
    fi
    
    # Run the agent
    cd "$REPO_DIR/ai-game-server"
    python openclaw_agent.py \
        --rom "$ROMS_DIR/$game" \
        --model "$model" \
        --turns "$max_turns"
}

# Show help
show_help() {
    echo -e "${CYAN}DuckBot Gaming Agent Spawner${NC}"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  list                      List available ROMs"
    echo "  spawn [game] [model]     Show spawn command for game"
    echo "  auto [game] [model]      Run autonomous gameplay"
    echo "  register                  Register MCP server"
    echo "  help                      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 spawn pokemon-red.gb"
    echo "  $0 spawn pokemon-blue.gb bailian/kimi-k2.5"
    echo "  $0 auto pokemon-red.gb bailian/kimi-k2.5 100"
    echo "  $0 register"
    echo ""
    echo "Default game: pokemon-red.gb"
    echo "Default model: bailian/kimi-k2.5 (FREE unlimited vision!)"
    echo "Default turns: 50"
}

# Main
case "${1:-help}" in
    list)
        list_roms
        ;;
    spawn)
        spawn_agent "${2:-pokemon-red.gb}" "${3:-bailian/kimi-k2.5}"
        ;;
    auto)
        run_autonomous "${2:-pokemon-red.gb}" "${3:-bailian/kimi-k2.5}" "${4:-50}"
        ;;
    register)
        register_mcp
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac