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
  backendUrl: 'http://localhost:5002',
  agentMode: true,
  openclawMcpEndpoint: 'http://localhost:18789',
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

  // ============ RENDER ============

  return (
    <div className="h-screen w-full flex flex-col bg-neutral-950 text-gray-100 font-sans overflow-hidden">
      {/* Keyboard Shortcuts Help Modal */}
      {showKeyboardHelp && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={() => setShowKeyboardHelp(false)}>
          <div className="bg-neutral-900 rounded-xl p-5 max-w-xs border border-neutral-700" onClick={e => e.stopPropagation()}>
            <h2 className="text-base font-semibold mb-3 text-neutral-200">Keyboard Shortcuts</h2>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>↑</span><span className="text-neutral-400">Up</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>↓</span><span className="text-neutral-400">Down</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>←</span><span className="text-neutral-400">Left</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>→</span><span className="text-neutral-400">Right</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>Z</span><span className="text-neutral-400">A</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>X</span><span className="text-neutral-400">B</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>Enter</span><span className="text-neutral-400">Start</span></div>
              <div className="flex justify-between px-2 py-1 bg-neutral-800 rounded"><span>Shift</span><span className="text-neutral-400">Select</span></div>
            </div>
            <p className="text-xs text-neutral-500 mt-3 text-center">Press <kbd className="px-1 py-0.5 bg-neutral-800 rounded">?</kbd> to toggle</p>
            <button onClick={() => setShowKeyboardHelp(false)} className="mt-3 w-full px-3 py-1.5 bg-neutral-700 hover:bg-neutral-600 rounded text-sm">Close</button>
          </div>
        </div>
      )}

      {/* ============ HEADER ============ */}
      <header className="flex items-center justify-between px-4 py-2 bg-neutral-900 border-b border-neutral-800">
        <div className="flex items-center gap-3">
          <Gamepad2 className="w-5 h-5 text-cyan-400" />
          <h1 className="text-lg font-semibold">AI Game Assistant</h1>
          {isRomLoaded && (
            <span className="text-xs px-2 py-0.5 bg-neutral-800 rounded text-neutral-400">{gameState.rom_name}</span>
          )}
        </div>
        
        <div className="flex items-center gap-3">
          {/* Connection Status */}
          <div className="flex items-center gap-1.5 px-2 py-1 bg-neutral-800 rounded text-xs">
            <span className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-400' : 
              connectionStatus === 'disconnected' ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'
            }`} />
            <span className="text-neutral-400">{connectionStatus}</span>
          </div>

          {/* Mode Badge */}
          <div className={`px-2 py-1 rounded text-xs font-medium ${
            agentState.enabled ? 'bg-green-900/50 text-green-400' : 'bg-orange-900/50 text-orange-400'
          }`}>
            {agentState.enabled ? '🤖 Auto' : '👆 Manual'}
          </div>

          {/* Keyboard help */}
          <button 
            onClick={() => setShowKeyboardHelp(true)}
            className="p-1.5 hover:bg-neutral-800 rounded transition-colors"
            title="Keyboard Shortcuts"
          >
            <Keyboard className="w-4 h-4 text-neutral-500" />
          </button>

          {/* Settings */}
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="p-1.5 hover:bg-neutral-800 rounded transition-colors"
          >
            <Settings className="w-4 h-4 text-neutral-500" />
          </button>
        </div>
      </header>

      {/* ============ MAIN CONTENT ============ */}
      <main className="flex-grow flex flex-col lg:flex-row gap-4 p-4 min-h-0 overflow-hidden">
        
        {/* GAME CANVAS (Left/Center) */}
        <div className="flex-1 flex flex-col min-h-0 bg-neutral-900 rounded-xl border border-neutral-800 overflow-hidden">
          <div className="flex items-center justify-between px-3 py-1.5 bg-neutral-800/50 border-b border-neutral-800">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-medium text-neutral-300">Game</h2>
              {isRomLoaded && (
                <span className="text-xs text-neutral-500">{gameState.frame_count} frames</span>
              )}
            </div>
            <div className="flex items-center gap-1">
              <label className="px-2 py-1 bg-cyan-600 hover:bg-cyan-500 rounded text-xs cursor-pointer transition-colors">
                <input type="file" accept=".gb,.gbc,.gba,.zip" onChange={handleFileSelect} className="hidden" />
                Load ROM
              </label>
              {isRomLoaded && (
                <>
                  <button onClick={handleSaveState} className="p-1 hover:bg-neutral-700 rounded" title="Save" disabled={!isRomLoaded}>
                    <Save className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={handleLoadState} className="p-1 hover:bg-neutral-700 rounded" title="Load" disabled={!isRomLoaded}>
                    <FolderOpen className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={refreshScreen} className="p-1 hover:bg-neutral-700 rounded" title="Refresh">
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
            </div>
          </div>
          
          <div className="flex-1 flex items-center justify-center p-2 bg-black">
            {gameScreenUrl ? (
              <img 
                src={gameScreenUrl} 
                alt="Game Screen" 
                className="max-w-full max-h-full object-contain aspect-[160:144]"
              />
            ) : (
              <div className="text-neutral-500 text-center">
                <Gamepad2 className="w-12 h-12 mx-auto mb-3 opacity-40" />
                <p className="text-sm">No ROM loaded</p>
                <p className="text-xs mt-1 text-neutral-600">Click "Load ROM" to start</p>
              </div>
            )}
          </div>
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
            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {/* Mode Toggle */}
              <div className="flex gap-1 p-1 bg-neutral-800 rounded-lg">
                <button 
                  onClick={() => setAgentState(prev => ({ ...prev, mode: 'auto', enabled: true }))}
                  className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                    agentState.mode === 'auto' ? 'bg-green-600 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  🤖 Auto
                </button>
                <button 
                  onClick={() => setAgentState(prev => ({ ...prev, mode: 'manual', enabled: false }))}
                  className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
                    agentState.mode === 'manual' ? 'bg-orange-600 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  👆 Manual
                </button>
              </div>

              {/* Status */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-neutral-500">Current</span>
                  <span className="text-xs text-green-400">{agentState.current_action}</span>
                </div>
                <div className="p-2 bg-neutral-800 rounded text-xs text-neutral-300">
                  {agentState.last_decision}
                </div>
              </div>

              {/* Decision Log */}
              <div className="space-y-1">
                <h3 className="text-xs font-medium text-neutral-500">Decision Log</h3>
                <div className="bg-neutral-800 rounded p-2 h-48 overflow-y-auto font-mono text-xs space-y-0.5">
                  {decisionLogs.slice(-20).reverse().map((entry) => (
                    <div key={entry.id} className={`${
                      entry.type === 'error' ? 'text-red-400' :
                      entry.type === 'action' ? 'text-yellow-400' :
                      entry.type === 'thought' ? 'text-cyan-400' : 'text-neutral-300'
                    }`}>
                      <span className="text-neutral-600">[{entry.timestamp.split('T')[1].slice(0,5)}]</span>{' '}
                      {entry.message}
                    </div>
                  ))}
                  {decisionLogs.length === 0 && (
                    <div className="text-neutral-600 text-center py-4">Waiting...</div>
                  )}
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
            <div className="flex-1 overflow-y-auto p-2">
              {isRomLoaded ? (
                <div className="space-y-0.5 font-mono text-xs">
                  {memoryState.values.map((mem, idx) => (
                    <div key={idx} className="flex justify-between items-center px-2 py-1 hover:bg-neutral-800 rounded">
                      <span className="text-neutral-500">{mem.name}</span>
                      <span className="text-purple-400 text-xs">
                        {mem.value !== null ? `0x${mem.value.toString(16).toUpperCase().padStart(2, '0')}` : '—'}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-neutral-600 py-6 text-xs">
                  Load ROM for memory
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
    { label: 'A', action: 'A', color: 'bg-green-600 hover:bg-green-500' },
    { label: 'B', action: 'B', color: 'bg-red-600 hover:bg-red-500' },
  ];

  const dpadButtons: { label: string; action: GameButton; gridArea: string }[] = [
    { label: '↑', action: 'UP', gridArea: '1 / 2' },
    { label: '←', action: 'LEFT', gridArea: '2 / 1' },
    { label: '→', action: 'RIGHT', gridArea: '2 / 3' },
    { label: '↓', action: 'DOWN', gridArea: '3 / 2' },
  ];

  return (
    <div className="flex items-center justify-center gap-6">
      {/* D-Pad */}
      <div 
        className="grid grid-cols-3 grid-rows-3 gap-0.5 w-24 h-24"
        style={{ gridTemplateAreas: `". up ." "left . right" ". down ."` }}
      >
        {dpadButtons.map(({ label, action, gridArea }) => (
          <button
            key={action}
            style={{ gridArea }}
            onClick={() => onButtonPress(action)}
            disabled={disabled}
            className={`
              flex items-center justify-center rounded transition-all text-sm
              ${disabled ? 'bg-neutral-800 text-neutral-600 cursor-not-allowed' : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 active:scale-90'}
              ${lastButton === action ? 'bg-cyan-600 scale-90' : ''}
            `}
          >
            {label}
          </button>
        ))}
        <div className="w-8 h-8" style={{ gridArea: '2 / 2' }} />
      </div>

      {/* System Buttons */}
      <div className="flex flex-col gap-1.5">
        {(['SELECT', 'START'] as const).map(label => (
          <button
            key={label}
            onClick={() => onButtonPress(label as GameButton)}
            disabled={disabled}
            className={`
              px-3 py-1.5 rounded text-xs font-medium transition-all
              ${disabled ? 'bg-neutral-800 text-neutral-600 cursor-not-allowed' : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 active:scale-95'}
              ${lastButton === label ? 'bg-cyan-600 scale-95' : ''}
            `}
          >
            {label}
          </button>
        ))}
      </div>

      {/* A/B Buttons */}
      <div className="flex items-center gap-3">
        {buttons.map(({ label, action, color }) => (
          <button
            key={action}
            onClick={() => onButtonPress(action)}
            disabled={disabled}
            className={`
              w-11 h-11 rounded-full font-bold text-sm transition-all
              ${disabled ? 'bg-neutral-700 text-neutral-500 cursor-not-allowed' : `${color} text-white active:scale-90`}
              ${lastButton === action ? 'scale-90 ring-2 ring-white' : ''}
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