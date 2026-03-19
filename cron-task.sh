#!/bin/bash
# DuckBot AI-PyBoy-Emulation Maintenance Cron
# Runs every hour - expires in 12 hours from setup
# Auto-expires at: $(date -r $(( $(date +%s) + 43200)))

LOG="/Users/duckets/.openclaw/workspace/logs/duckbot-cron.log"
REPO_DIR="/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main"
PYBOY_DIR="/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server"

echo "[$(date '+%Y-%m-%d %H:%M')] DuckBot maintenance..." >> $LOG

# Check/fix PyBoy version
echo "[$(date '+%Y-%m-%d %H:%M')] Checking PyBoy..." >> $LOG
/Users/duckets/.openclaw/workspace/.venv-pyboy/bin/pip install --upgrade pyboy --quiet 2>&1 | tail -1 >> $LOG

# Check if backend is running, restart if needed
if ! curl -s http://localhost:5002/api/health > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M')] Backend down, restarting..." >> $LOG
    cd $PYBOY_DIR/src
    pkill -f "main.py" 2>/dev/null
    BACKEND_PORT=5002 nohup /Users/duckets/.openclaw/workspace/.venv-pyboy/bin/python3 main.py >> $LOG 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M')] Backend restarted" >> $LOG
else
    echo "[$(date '+%Y-%m-%d %H:%M')] Backend OK" >> $LOG
fi

# Check if frontend is running, restart if needed
if ! curl -s http://localhost:5174 > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M')] Frontend down, restarting..." >> $LOG
    cd $REPO_DIR/ai-game-assistant
    pkill -f "vite" 2>/dev/null
    nohup npx vite --port 5174 --host >> $LOG 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M')] Frontend restarted" >> $LOG
else
    echo "[$(date '+%Y-%m-%d %H:%M')] Frontend OK" >> $LOG
fi

# Check MCP server
if ! curl -s http://localhost:5002/api/emulator/state > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M')] MCP server needs attention" >> $LOG
fi

echo "[$(date '+%Y-%m-%d %H:%M')] Maintenance complete" >> $LOG
