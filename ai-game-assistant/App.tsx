/**
 * AI Game Assistant - Main Application (Enhanced)
 * 
 * Features:
 * - Settings saved to localStorage
 * - Auto-connect on startup
 * - Real-time screen streaming (SSE/polling)
 * - Keyboard shortcuts for controls
 * - ROM file picker
 * - Save/Load state buttons
 * - Live decision logs in agent panel
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Settings, Play, Pause, Save, FolderOpen, RefreshCw, Activity, MemoryStick, Gamepad2, ChevronDown, ChevronUp, Keyboard, Upload } from 'lucide-react';
import SettingsModal from './src/components/SettingsModal';
import type { GameState, AgentStatus, MemoryWatch, LogEntry, GameButton, AppSettings } from './services/apiService';
import apiService from './services/apiService';

// Storage keys
const STORAGE_KEYS = {
  SETTINGS: 'aiGameAssistant_settings',
  LAST_ROM: 'aiGameAssistant_lastRom',
};

// Constants
const SCREEN_REFRESH_MS = 250;  // Faster refresh for real-time feel
const STATE_REFRESH_MS = 2000;
const MEMORY_REFRESH_MS = 5000;
const DECISION_LOG_MAX = 100;

// Default settings
const DEFAULT_SETTINGS: AppSettings = {
  aiActionInterval: 5000,
  backendUrl: 'http://localhost:5000',
  agentMode: true,
  openclawMcpEndpoint: 'http://localhost:3000/mcp',
  visionModel: 'kimi-k2.5',
  autonomousLevel: 'moderate',
  agentPersonality: 'strategic',
  agentObjectives: 'Complete Pokemon Red',
  apiProvider: 'openclaw',
  apiEndpoint: '',
  apiKey: '',
  model: '',
};

// Load settings from localStorage
const loadSettings = (): AppSettings => {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
    if (stored) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
  return DEFAULT_SETTINGS;
};

// Save settings to localStorage
const saveSettings = (settings: AppSettings): void => {
  try {
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
};

const App: React.FC = () => {
  // ============ STATE MANAGEMENT ============
  
  // Settings - loaded from localStorage on mount
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  // Game State
  const [gameState, setGameState] = useState<GameState>({
    running: false,
    rom_loaded: false,
    rom_name: '',
    screen_available: false,
    frame_count: 0,
    fps: 0,
    emulator: 'gb',
    timestamp: '',
  });

  // Agent State
  const [agentState, setAgentState] = useState<AgentStatus>({
    connected: false,
    agent_name: 'OpenClaw Agent',
    mode: 'auto',
    autonomous_level: 'moderate',
    current_action: 'Idle',
    last_decision: 'Initializing...',
    enabled: true,
    game_running: false,
    timestamp: '',
  });

  // Memory State
  const [memoryState, setMemoryState] = useState<MemoryWatch>({
    addresses: [],
    values: [],
    timestamp: '',
  });

  // Decision logs (for agent panel)
  const [decisionLogs, setDecisionLogs] = useState<LogEntry[]>([]);
  const [actionLogs, setActionLogs] = useState<LogEntry[]>([]);

  // UI State
  const [isRomLoaded, setIsRomLoaded] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [lastButtonPressed, setLastButtonPressed] = useState<GameButton | null>(null);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  
  // Panel collapse states
  const [agentPanelCollapsed, setAgentPanelCollapsed] = useState(false);
  const [memoryPanelCollapsed, setMemoryPanelCollapsed] = useState(false);

  // Refs
  const screenBlobRef = useRef<Blob | null>(null);
  const screenUrlRef = useRef<string | null>(null);
  const [gameScreenUrl, setGameScreenUrl] = useState<string | null>(null);
  const decisionIdRef = useRef(0);

  // ============ API HELPERS ============

  const addDecisionLog = useCallback((type: LogEntry['type'], message: string) => {
    const entry: LogEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      type,
      message,
    };
    setDecisionLogs(prev => [...prev.slice(-(DECISION_LOG_MAX - 1)), entry]);
  }, []);

  const addActionLog = useCallback((type: LogEntry['type'], message: string) => {
    const entry: LogEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      type,
      message,
    };
    setActionLogs(prev => [...prev.slice(-49), entry]);
  }, []);

  const refreshGameState = useCallback(async () => {
    try {
      const state = await apiService.getGameState();
      setGameState(state);
      setIsRomLoaded(state.rom_loaded);
      if (state.rom_loaded && connectionStatus !== 'connected') {
        setConnectionStatus('connected');
        addActionLog('system', 'Connected to game server');
      }
    } catch (error) {
      if (connectionStatus !== 'disconnected') {
        setConnectionStatus('disconnected');
        addActionLog('error', 'Failed to connect to game server');
      }
    }
  }, [connectionStatus, addActionLog]);

  const refreshAgentStatus = useCallback(async () => {
    try {
      const status = await apiService.getAgentStatus();
      setAgentState(status);
      
      // Add decision to logs if it's new
      if (status.last_decision && status.last_decision !== agentState.last_decision) {
        addDecisionLog('thought', status.last_decision);
      }
      if (status.current_action && status.current_action !== agentState.current_action) {
        addDecisionLog('action', `→ ${status.current_action}`);
      }
    } catch (error) {
      // Silent fail for agent status
    }
  }, [agentState.last_decision, agentState.current_action, addDecisionLog]);

  const refreshMemory = useCallback(async () => {
    if (!isRomLoaded) return;
    try {
      const memory = await apiService.getMemoryWatch();
      setMemoryState(memory);
    } catch (error) {
      // Silent fail for memory
    }
  }, [isRomLoaded]);

  const refreshScreen = useCallback(async () => {
    if (!isRomLoaded || connectionStatus !== 'connected') return;
    try {
      const blob = await apiService.getScreen();
      screenBlobRef.current = blob;
      const newUrl = URL.createObjectURL(blob);
      
      // Cleanup old URL
      if (screenUrlRef.current) {
        URL.revokeObjectURL(screenUrlRef.current);
      }
      screenUrlRef.current = newUrl;
      setGameScreenUrl(newUrl);
    } catch (error) {
      // Silent fail for screen refresh
    }
  }, [isRomLoaded, connectionStatus]);

  // ============ EFFECTS ============

  // Initial connection on mount - Auto-connect
  useEffect(() => {
    const connect = async () => {
      setConnectionStatus('checking');
      try {
        apiService.setBaseUrl(settings.backendUrl);
        await refreshGameState();
        await refreshAgentStatus();
        setConnectionStatus('connected');
        addActionLog('system', `Auto-connected to ${settings.backendUrl}`);
      } catch (error) {
        setConnectionStatus('disconnected');
        addActionLog('error', `Auto-connect failed to ${settings.backendUrl}`);
      }
    };
    connect();
  }, []); // Only run once on mount

  // Re-connect when backend URL changes
  useEffect(() => {
    const reconnect = async () => {
      if (settings.backendUrl) {
        apiService.setBaseUrl(settings.backendUrl);
        await refreshGameState();
        await refreshAgentStatus();
      }
    };
    reconnect();
  }, [settings.backendUrl]);

  // Screen refresh interval (real-time)
  useEffect(() => {
    if (!isRomLoaded || connectionStatus !== 'connected') return;
    
    const interval = setInterval(refreshScreen, SCREEN_REFRESH_MS);
    return () => clearInterval(interval);
  }, [isRomLoaded, connectionStatus, refreshScreen]);

  // Game state refresh interval
  useEffect(() => {
    const interval = setInterval(() => {
      refreshGameState();
      refreshAgentStatus();
    }, STATE_REFRESH_MS);
    return () => clearInterval(interval);
  }, [refreshGameState, refreshAgentStatus]);

  // Memory refresh interval
  useEffect(() => {
    const interval = setInterval(refreshMemory, MEMORY_REFRESH_MS);
    return () => clearInterval(interval);
  }, [refreshMemory]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (screenUrlRef.current) {
        URL.revokeObjectURL(screenUrlRef.current);
      }
    };
  }, []);

  // ============ KEYBOARD SHORTCUTS ============
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const key = e.key.toUpperCase();
      const buttonMap: Record<string, GameButton> = {
        'ARROWUP': 'UP',
        'ARROWDOWN': 'DOWN',
        'ARROWLEFT': 'LEFT',
        'ARROWRIGHT': 'RIGHT',
        'Z': 'A',
        'X': 'B',
        'ENTER': 'START',
        'SHIFT': 'SELECT',
      };

      const button = buttonMap[key];
      if (button) {
        e.preventDefault();
        handleButtonPress(button);
      }

      // Toggle keyboard help
      if (e.key === '?') {
        setShowKeyboardHelp(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRomLoaded]);

  // ============ HANDLERS ============

  const handleButtonPress = async (button: GameButton) => {
    if (!isRomLoaded) return;
    
    setLastButtonPressed(button);
    try {
      await apiService.pressButton(button);
      addActionLog('action', `Pressed: ${button}`);
      // Refresh screen after button press
      setTimeout(refreshScreen, 50);
    } catch (error) {
      addActionLog('error', `Failed to press ${button}`);
    }
    setTimeout(() => setLastButtonPressed(null), 200);
  };

  const handleLoadRom = async (file: File) => {
    try {
      await apiService.loadRom(file);
      setIsRomLoaded(true);
      addActionLog('system', `Loaded ROM: ${file.name}`);
      // Save last ROM name
      localStorage.setItem(STORAGE_KEYS.LAST_ROM, file.name);
      await refreshGameState();
    } catch (error) {
      addActionLog('error', `Failed to load ROM: ${error}`);
    }
  };

  const handleSaveState = async () => {
    if (!isRomLoaded) return;
    try {
      await apiService.saveState();
      addActionLog('system', '💾 Game state saved');
    } catch (error) {
      addActionLog('error', 'Failed to save state');
    }
  };

  const handleLoadState = async () => {
    if (!isRomLoaded) return;
    try {
      await apiService.loadState();
      addActionLog('system', '📂 Game state loaded');
      await refreshScreen();
    } catch (error) {
      addActionLog('error', 'Failed to load state');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleLoadRom(file);
    }
  };

  const handleSettingsSave = (newSettings: AppSettings) => {
    setSettings(newSettings);
    saveSettings(newSettings);
    apiService.setBaseUrl(newSettings.backendUrl);
    addActionLog('system', '⚙️ Settings saved');
  };

  // ============ RENDER HELPERS ============

  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-400';
      case 'disconnected': return 'text-red-400';
      case 'checking': return 'text-yellow-400';
    }
  };

  const getModeColor = () => {
    return agentState.mode === 'auto' ? 'text-blue-400' : 'text-orange-400';
  };

  // ============ RENDER ============

  return (
    <div className="h-screen w-full flex flex-col bg-neutral-950 text-gray-100 font-sans overflow-hidden">
      {/* Keyboard Shortcuts Help Modal */}
      {showKeyboardHelp && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={() => setShowKeyboardHelp(false)}>
          <div className="bg-neutral-900 rounded-xl p-6 max-w-md border border-neutral-800" onClick={e => e.stopPropagation()}>
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Keyboard className="w-5 h-5" /> Keyboard Shortcuts
            </h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between"><span>↑ Arrow</span><span className="text-neutral-400">Up</span></div>
              <div className="flex justify-between"><span>↓ Arrow</span><span className="text-neutral-400">Down</span></div>
              <div className="flex justify-between"><span>← Arrow</span><span className="text-neutral-400">Left</span></div>
              <div className="flex justify-between"><span>→ Arrow</span><span className="text-neutral-400">Right</span></div>
              <div className="flex justify-between"><span>Z</span><span className="text-neutral-400">A Button</span></div>
              <div className="flex justify-between"><span>X</span><span className="text-neutral-400">B Button</span></div>
              <div className="flex justify-between"><span>Enter</span><span className="text-neutral-400">Start</span></div>
              <div className="flex justify-between"><span>Shift</span><span className="text-neutral-400">Select</span></div>
              <div className="flex justify-between"><span>?</span><span className="text-neutral-400">Toggle this help</span></div>
            </div>
            <button onClick={() => setShowKeyboardHelp(false)} className="mt-4 w-full px-4 py-2 bg-neutral-700 hover:bg-neutral-600 rounded">Close</button>
          </div>
        </div>
      )}

      {/* ============ HEADER ============ */}
      <header className="flex items-center justify-between px-4 py-3 bg-neutral-900 border-b border-neutral-800">
        <div className="flex items-center gap-3">
          <Gamepad2 className="w-6 h-6 text-blue-400" />
          <h1 className="text-xl font-bold">AI Game Assistant</h1>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Keyboard help button */}
          <button 
            onClick={() => setShowKeyboardHelp(true)}
            className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
            title="Keyboard Shortcuts"
          >
            <Keyboard className="w-5 h-5 text-neutral-400 hover:text-white" />
          </button>

          {/* Agent Status */}
          <div className="flex items-center gap-2 px-3 py-1 bg-neutral-800 rounded-lg">
            <Activity className={`w-4 h-4 ${getConnectionColor()}`} />
            <span className={`text-sm font-medium ${getConnectionColor()}`}>
              {agentState.enabled ? 'Agent Active' : 'Manual Mode'}
            </span>
            <span className="text-neutral-500">|</span>
            <span className={`text-sm ${getModeColor()}`}>
              {agentState.mode.toUpperCase()}
            </span>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-400' : 
              connectionStatus === 'disconnected' ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'
            }`} />
            <span className="text-sm text-neutral-400">{settings.backendUrl}</span>
          </div>

          {/* Settings Button */}
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 hover:bg-neutral-800 rounded-lg transition-colors"
          >
            <Settings className="w-5 h-5 text-neutral-400 hover:text-white" />
          </button>
        </div>
      </header>

      {/* ============ MAIN CONTENT ============ */}
      <main className="flex-grow flex flex-col lg:flex-row gap-4 p-4 min-h-0 overflow-hidden">
        
        {/* GAME CANVAS (Left/Center) */}
        <div className="flex-1 flex flex-col min-h-0 bg-neutral-900 rounded-xl border border-neutral-800 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-neutral-800/50 border-b border-neutral-800">
            <h2 className="font-semibold">Game Canvas</h2>
            <div className="flex items-center gap-2">
              {/* ROM File Picker */}
              <label className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm cursor-pointer transition-colors flex items-center gap-1">
                <Upload className="w-3 h-3" />
                <input type="file" accept=".gb,.gbc,.gba,.zip" onChange={handleFileSelect} className="hidden" />
                Load ROM
              </label>
              <button onClick={handleSaveState} className="p-1 hover:bg-neutral-700 rounded" title="Save State (F5)" disabled={!isRomLoaded}>
                <Save className="w-4 h-4" />
              </button>
              <button onClick={handleLoadState} className="p-1 hover:bg-neutral-700 rounded" title="Load State (F6)" disabled={!isRomLoaded}>
                <FolderOpen className="w-4 h-4" />
              </button>
              <button onClick={refreshScreen} className="p-1 hover:bg-neutral-700 rounded" title="Refresh Screen">
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="flex-1 flex items-center justify-center p-4 bg-black">
            {gameScreenUrl ? (
              <img 
                src={gameScreenUrl} 
                alt="Game Screen" 
                className="max-w-full max-h-full object-contain aspect-ratio-[160:144]"
              />
            ) : (
              <div className="text-neutral-500 text-center">
                <Gamepad2 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>No ROM loaded</p>
                <p className="text-sm mt-2">Load a ROM to start</p>
                <p className="text-xs mt-4 text-neutral-600">Or press ? for keyboard shortcuts</p>
              </div>
            )}
          </div>

          {/* FPS / Frame info */}
          {isRomLoaded && (
            <div className="px-4 py-1 bg-neutral-800/50 text-xs text-neutral-400 flex justify-between">
              <span>Frame: {gameState.frame_count}</span>
              <span>ROM: {gameState.rom_name || 'Unknown'}</span>
              <span>Screen: {SCREEN_REFRESH_MS}ms</span>
            </div>
          )}
        </div>

        {/* AGENT PANEL (Middle) - Live Decision Logs */}
        <div className={`flex flex-col min-h-0 bg-neutral-900 rounded-xl border border-neutral-800 overflow-hidden transition-all ${
          agentPanelCollapsed ? 'w-12' : 'w-full lg:w-80'
        }`}>
          <button 
            onClick={() => setAgentPanelCollapsed(!agentPanelCollapsed)}
            className="flex items-center justify-between px-4 py-2 bg-neutral-800/50 border-b border-neutral-800 hover:bg-neutral-800"
          >
            {!agentPanelCollapsed && <h2 className="font-semibold">Agent Panel</h2>}
            <div className="flex items-center gap-2">
              <Activity className={`w-4 h-4 ${agentState.enabled ? 'text-green-400' : 'text-neutral-500'}`} />
              {agentPanelCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </div>
          </button>

          {!agentPanelCollapsed && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Mode Section */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-400 uppercase">Mode</h3>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setAgentState(prev => ({ ...prev, mode: 'auto', enabled: true }))}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      agentState.mode === 'auto' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                    }`}
                  >
                    Auto
                  </button>
                  <button 
                    onClick={() => setAgentState(prev => ({ ...prev, mode: 'manual', enabled: false }))}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      agentState.mode === 'manual' 
                        ? 'bg-orange-600 text-white' 
                        : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
                    }`}
                  >
                    Manual
                  </button>
                </div>
              </div>

              {/* Current Action */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-400 uppercase">Current Action</h3>
                <div className="p-3 bg-neutral-800 rounded-lg">
                  <p className="font-mono text-green-400">{agentState.current_action}</p>
                </div>
              </div>

              {/* Last Decision */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-400 uppercase">Last Decision</h3>
                <div className="p-3 bg-neutral-800 rounded-lg">
                  <p className="text-sm">{agentState.last_decision}</p>
                </div>
              </div>

              {/* Live Decision Logs */}
              <div className="space-y-2 flex-1">
                <h3 className="text-sm font-medium text-neutral-400 uppercase flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                  Live Decision Log
                </h3>
                <div className="bg-neutral-800 rounded-lg p-2 h-64 overflow-y-auto font-mono text-xs space-y-1">
                  {decisionLogs.slice(-30).reverse().map((entry) => (
                    <div key={entry.id} className={`${
                      entry.type === 'error' ? 'text-red-400' :
                      entry.type === 'action' ? 'text-yellow-400' :
                      entry.type === 'thought' ? 'text-blue-400' :
                      'text-neutral-300'
                    }`}>
                      <span className="text-neutral-600">[{entry.timestamp.split('T')[1].slice(0,8)}]</span>{' '}
                      {entry.message}
                    </div>
                  ))}
                  {decisionLogs.length === 0 && (
                    <div className="text-neutral-500 text-center py-4">
                      Waiting for agent decisions...
                    </div>
                  )}
                </div>
              </div>

              {/* Action Log */}
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-neutral-400 uppercase">Action Log</h3>
                <div className="bg-neutral-800 rounded-lg p-2 h-32 overflow-y-auto font-mono text-xs space-y-1">
                  {actionLogs.slice(-15).reverse().map((entry) => (
                    <div key={entry.id} className={`${
                      entry.type === 'error' ? 'text-red-400' :
                      entry.type === 'action' ? 'text-green-400' :
                      entry.type === 'system' ? 'text-blue-400' :
                      'text-neutral-300'
                    }`}>
                      <span className="text-neutral-600">[{entry.timestamp.split('T')[1].slice(0,8)}]</span>{' '}
                      {entry.message}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* MEMORY INSPECTOR (Right) */}
        <div className={`flex flex-col min-h-0 bg-neutral-900 rounded-xl border border-neutral-800 overflow-hidden transition-all ${
          memoryPanelCollapsed ? 'w-12' : 'w-full lg:w-72'
        }`}>
          <button 
            onClick={() => setMemoryPanelCollapsed(!memoryPanelCollapsed)}
            className="flex items-center justify-between px-4 py-2 bg-neutral-800/50 border-b border-neutral-800 hover:bg-neutral-800"
          >
            {!memoryPanelCollapsed && <h2 className="font-semibold">Memory Inspector</h2>}
            <div className="flex items-center gap-2">
              <MemoryStick className={`w-4 h-4 ${isRomLoaded ? 'text-purple-400' : 'text-neutral-500'}`} />
              {memoryPanelCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </div>
          </button>

          {!memoryPanelCollapsed && (
            <div className="flex-1 overflow-y-auto p-4">
              {isRomLoaded ? (
                <div className="space-y-1 font-mono text-xs">
                  {memoryState.values.map((mem, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2 hover:bg-neutral-800 rounded">
                      <span className="text-neutral-400">{mem.name}</span>
                      <span className="text-purple-400">
                        {mem.value !== null ? `0x${mem.value.toString(16).toUpperCase().padStart(2, '0')} (${mem.value})` : 'N/A'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-neutral-500 py-8">
                  <MemoryStick className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Load a ROM to view memory</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* ============ CONTROLS ============ */}
      <div className="bg-neutral-900 border-t border-neutral-800 p-4">
        <Controls 
          lastButton={lastButtonPressed} 
          onButtonPress={handleButtonPress}
          disabled={!isRomLoaded}
        />
      </div>

      {/* ============ SETTINGS MODAL ============ */}
      <SettingsModal 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSettingsSave}
      />
    </div>
  );
};

// ============ CONTROLS COMPONENT ============
interface ControlsProps {
  lastButton: GameButton | null;
  onButtonPress: (button: GameButton) => void;
  disabled: boolean;
}

const Controls: React.FC<ControlsProps> = ({ lastButton, onButtonPress, disabled }) => {
  const buttons: { label: string; action: GameButton; color: string }[] = [
    { label: 'A', action: 'A', color: 'bg-green-500 hover:bg-green-600' },
    { label: 'B', action: 'B', color: 'bg-red-500 hover:bg-red-600' },
  ];

  const dpadButtons: { label: string; action: GameButton; gridArea: string }[] = [
    { label: '', action: 'UP', gridArea: '1 / 2' },
    { label: '', action: 'LEFT', gridArea: '2 / 1' },
    { label: '', action: 'RIGHT', gridArea: '2 / 3' },
    { label: '', action: 'DOWN', gridArea: '3 / 2' },
  ];

  const systemButtons: { label: string; action: GameButton }[] = [
    { label: 'SELECT', action: 'SELECT' },
    { label: 'START', action: 'START' },
  ];

  return (
    <div className="flex items-center justify-center gap-8">
      {/* D-Pad */}
      <div 
        className="grid grid-cols-3 grid-rows-3 gap-1 w-32 h-32"
        style={{ gridTemplateAreas: `". up ." "left . right" ". down ."` }}
      >
        {dpadButtons.map(({ label, action, gridArea }) => (
          <button
            key={action}
            style={{ gridArea }}
            onClick={() => onButtonPress(action)}
            disabled={disabled}
            className={`
              w-12 h-12 flex items-center justify-center rounded-lg transition-all
              ${disabled ? 'bg-neutral-800 cursor-not-allowed' : 'bg-neutral-700 hover:bg-neutral-600 active:scale-95'}
              ${lastButton === action ? 'bg-blue-600 scale-95' : ''}
            `}
          >
            <span className="text-neutral-400 text-xs">{label}</span>
          </button>
        ))}
        <div className="w-12 h-12" style={{ gridArea: '2 / 2' }} />
      </div>

      {/* System Buttons */}
      <div className="flex flex-col gap-2">
        {systemButtons.map(({ label, action }) => (
          <button
            key={action}
            onClick={() => onButtonPress(action)}
            disabled={disabled}
            className={`
              px-4 py-2 rounded-full text-xs font-medium transition-all
              ${disabled ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed' : 'bg-neutral-700 hover:bg-neutral-600 active:scale-95'}
              ${lastButton === action ? 'bg-blue-600 scale-95' : ''}
            `}
          >
            {label}
          </button>
        ))}
      </div>

      {/* A/B Buttons */}
      <div className="flex items-center gap-4">
        {buttons.map(({ label, action, color }) => (
          <button
            key={action}
            onClick={() => onButtonPress(action)}
            disabled={disabled}
            className={`
              w-14 h-14 rounded-full font-bold text-lg transition-all
              ${disabled ? 'bg-neutral-700 text-neutral-500 cursor-not-allowed' : `${color} text-white active:scale-95`}
              ${lastButton === action ? 'scale-95 ring-2 ring-white' : ''}
            `}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};

export default App;