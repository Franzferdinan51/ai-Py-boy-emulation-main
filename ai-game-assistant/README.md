# AI Game Assistant - Web UI

**Status:** ✅ Production Ready  
**Last Updated:** March 19, 2026

## Features

### 🎮 Game Control
- **Real-time Screen Display** - 250ms refresh rate
- **Virtual Controller** - D-Pad, A/B buttons, Start/Select
- **Keyboard Shortcuts** - Arrow keys, Z, X, Enter, Shift
- **Save/Load States** - Quick save and load game states
- **ROM Loader** - Drag-and-drop or file picker

### 🤖 Agent System
- **Auto/Manual Mode** - Toggle AI control
- **Live Decision Logs** - Real-time agent thoughts and actions
- **Action History** - Track all agent actions
- **Agent Status** - Connection and mode indicators

### 📊 Game Data Panels

#### Party Panel
- View all 6 Pokemon in your party
- Real-time HP bars with color coding
- Status conditions (poison, burn, etc.)
- Move lists and types
- Auto-refresh every 5 seconds

#### Inventory Panel
- Money display (₽ currency)
- Item count and list
- 12 item categories with color coding
- Quantity tracking
- Auto-refresh every 10 seconds

#### Memory Inspector
- Watch key memory addresses
- Real-time value updates
- Hex and decimal display
- Custom address monitoring

#### Vision Analysis
- AI-powered screen analysis
- Auto-analyze mode (30s intervals)
- Confidence indicators
- Recommended actions
- Ready for vision model integration

### ⚙️ Settings
- Backend URL configuration
- Auto-connect on startup
- localStorage persistence
- API provider selection
- Vision model selection
- Agent personality and objectives

## Quick Start

### 1. Start Backend Server
```bash
cd ../ai-game-server
python start_server.py
# Server runs on http://localhost:5000
```

### 2. Build Frontend
```bash
cd ai-game-assistant
npm install
npm run build
```

### 3. Serve Frontend
```bash
npm run dev
# Or serve the build:
npx serve dist
```

### 4. Open Browser
Navigate to `http://localhost:5173` (dev) or `http://localhost:3000` (production)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| ↑↓←→ | D-Pad |
| Z | A Button |
| X | B Button |
| Enter | Start |
| Shift | Select |
| ? | Toggle keyboard help |

## Project Structure

```
ai-game-assistant/
├── App.tsx                      # Main application (rewritten)
├── services/
│   ├── apiService.ts           # API client (extended)
│   └── configService.ts        # Configuration
├── src/components/
│   ├── PartyPanel.tsx          # NEW: Pokemon party display
│   ├── InventoryPanel.tsx      # NEW: Item inventory
│   ├── VisionAnalysisPanel.tsx # NEW: AI vision analysis
│   ├── SettingsModal.tsx       # Settings dialog
│   └── ...                     # Other components
├── components/                  # DEAD CODE - to be removed
├── dist/                        # Build output
└── package.json
```

## API Integration

All backend endpoints are typed and callable via `apiService`:

```typescript
import apiService from './services/apiService';

// Game control
await apiService.pressButton('A');
await apiService.saveState();
await apiService.loadState();

// Data fetching
const party = await apiService.getParty();
const inventory = await apiService.getInventory();
const memory = await apiService.getMemoryWatch();

// Agent control
await apiService.setAgentMode({ mode: 'auto' });
const status = await apiService.getAgentStatus();
```

## Configuration

Settings are stored in localStorage:
- `aiGameAssistant_settings` - App settings
- `aiGameAssistant_lastRom` - Last loaded ROM name

Default backend URL: `http://localhost:5000`

## Performance

- **Build Time:** ~134ms
- **Bundle Size:** 397KB (123KB gzipped)
- **Screen Refresh:** 250ms (4 FPS)
- **State Refresh:** 2000ms
- **Memory/Party Refresh:** 5000ms

## Troubleshooting

### Can't Connect to Backend
1. Check backend server is running
2. Verify backend URL in settings
3. Check for CORS errors in console

### Screen Not Updating
1. Ensure ROM is loaded
2. Check connection status (green dot)
3. Try manual refresh button

### Panels Not Showing Data
1. Verify ROM is loaded
2. Check browser console for errors
3. Refresh the panel manually

## Development

### Add New Panel
1. Create component in `src/components/`
2. Add state to App.tsx
3. Add panel to main layout
4. Wire up API calls

### Add New API Endpoint
1. Add type to `apiService.ts`
2. Add method to ApiService class
3. Use in component

## Cleanup

Dead code identified in `/components/` directory:
```bash
# Safe to remove (not used)
rm -rf components/
```

**⚠️ DO NOT delete `/src/components/` - that's active!**

## License

MIT

## Credits

Built with:
- React + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Lucide React (icons)
- PyBoy (GameBoy emulator)

---

**Last Updated:** March 19, 2026  
**Version:** 2.0.0 (Feature Wiring Update)
