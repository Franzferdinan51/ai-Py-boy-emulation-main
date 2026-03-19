import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import AgentStatusPanel from './components/AgentStatusPanel';
import SettingsModal from './components/SettingsModal';
import { ErrorBoundary } from './components/ErrorBoundary';
import EmulatorScreen from './components/EmulatorScreen';
import Controls from './components/Controls';
import type { GameAction, AppSettings, AgentStatus, AIState } from './types';
import { AIState as AIStateEnum } from './types';
import { getScreen, sendAction, loadRom, saveState as saveGameState, loadState as loadGameState, checkBackendStatus } from './services/backendService';

// Icons
const GamepadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="6" y1="12" x2="10" y2="12"/><line x1="8" y1="10" x2="8" y2="14"/>
    <line x1="15" y1="13" x2="15.01" y2="13"/><line x1="18" y1="11" x2="18.01" y2="11"/>
    <rect width="8" height="14" x="8" y="5" rx="4"/>
  </svg>
);

const SaveIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
    <polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
  </svg>
);

const LoadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
    <line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/>
  </svg>
);

const App: React.FC = () => {
  const [aiState, setAiState] = useState<AIState>(AIStateEnum.IDLE);
  const [gameScreenUrl, setGameScreenUrl] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<GameAction | null>(null);
  const [actionLog, setActionLog] = useState<{ time: string; action: string; result: string }[]>([]);
  const [isRomLoaded, setIsRomLoaded] = useState(false);
  const [romName, setRomName] = useState<string | null>(null);
  
  // Backend connection state
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Settings
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings>({
    aiActionInterval: 4000,
    backendUrl: 'http://localhost:5000',
    agentMode: true,
    openclawMcpEndpoint: 'http://localhost:3000/mcp',
    visionModel: 'kimi-k2.5',
    autonomousLevel: 'moderate',
    agentObjectives: 'Complete Pokemon Red, defeat the Elite Four',
    agentPersonality: 'strategic',
  });

  // Agent status
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({
    connected: true,
    agentName: 'DuckBot',
    currentAction: 'Idle',
    lastDecision: 'Waiting for ROM...',
    heartbeat: 0,
    decisionHistory: ['Agent initialized'],
  });

  // Memory state (mock for display)
  const [memoryState, setMemoryState] = useState({
    x: 0,
    y: 0,
    map: 'Pallet Town',
    party: ['Charmander'],
    money: 0,
    steps: 0,
  });

  const screenRefreshRef = useRef<number | null>(null);
  const heartbeatRef = useRef<number | null>(null);

  // Check backend status on mount
  useEffect(() => {
    const checkConnection = async () => {
      setBackendStatus('checking');
      try {
        const isConnected = await checkBackendStatus(appSettings.backendUrl || 'http://localhost:5000');
        setBackendStatus(isConnected ? 'connected' : 'disconnected');
        setConnectionError(isConnected ? null : 'Backend not responding');
      } catch {
        setBackendStatus('disconnected');
        setConnectionError('Failed to connect to backend');
      }
    };
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [appSettings.backendUrl]);

  // Screen refresh loop
  useEffect(() => {
    if (!isRomLoaded) return;
    
    const fetchScreen = async () => {
      try {
        const screenData = await getScreen(appSettings.backendUrl || 'http://localhost:5000');
        if (screenData) {
          const timestamp = Date.now();
          setGameScreenUrl(`${screenData}?t=${timestamp}`);
        }
      } catch (err) {
        console.error('Screen fetch error:', err);
      }
    };

    fetchScreen();
    screenRefreshRef.current = window.setInterval(fetchScreen, 250); // 4 FPS
    
    return () => {
      if (screenRefreshRef.current) {
        clearInterval(screenRefreshRef.current);
      }
    };
  }, [isRomLoaded, appSettings.backendUrl]);

  // Heartbeat
  useEffect(() => {
    heartbeatRef.current = window.setInterval(() => {
      setAgentStatus(prev => ({
        ...prev,
        heartbeat: prev.heartbeat + 1,
        currentAction: isRomLoaded ? (appSettings.agentMode ? 'Playing autonomously' : 'Waiting for input') : 'Idle - No ROM',
      }));
    }, 1000);
    
    return () => {
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }
    };
  }, [isRomLoaded, appSettings.agentMode]);

  // Handle control inputs
  useEffect(() => {
    const handleControlPress = async (e: CustomEvent) => {
      const action = e.detail as GameAction;
      setLastAction(action);
      addToLog(action, 'Processing...');
      
      try {
        await sendAction(action, appSettings.backendUrl || 'http://localhost:5000');
        addToLog(action, '✓ Success');
        setAgentStatus(prev => ({
          ...prev,
          lastDecision: `Executed ${action}`,
          decisionHistory: [...prev.decisionHistory.slice(-9), `(${new Date().toLocaleTimeString()}) ${action}`],
        }));
      } catch (err) {
        addToLog(action, '✗ Failed');
      }
    };

    window.addEventListener('game-control-press', handleControlPress as EventListener);
    return () => {
      window.removeEventListener('game-control-press', handleControlPress as EventListener);
    };
  }, [appSettings.backendUrl]);

  const addToLog = (action: string, result: string) => {
    const time = new Date().toLocaleTimeString();
    setActionLog(prev => [...prev.slice(-19), { time, action, result }]);
  };

  const handleRomLoad = async (file: File) => {
    try {
      setBackendStatus('checking');
      const result = await loadRom(file, appSettings.backendUrl || 'http://localhost:5000');
      setIsRomLoaded(true);
      setRomName(file.name);
      setBackendStatus('connected');
      setAgentStatus(prev => ({
        ...prev,
        lastDecision: `Loaded ${file.name}`,
        decisionHistory: [...prev.decisionHistory, `Loaded ROM: ${file.name}`],
      }));
      addToLog('LOAD', `Loaded ${file.name}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load ROM';
      setConnectionError(message);
      setBackendStatus('disconnected');
      addToLog('LOAD', `✗ ${message}`);
    }
  };

  const handleSaveState = async () => {
    if (!isRomLoaded) return;
    try {
      await saveGameState(appSettings.backendUrl || 'http://localhost:5000');
      addToLog('SAVE', '✓ State saved');
    } catch (err) {
      addToLog('SAVE', '✗ Failed');
    }
  };

  const handleLoadState = async () => {
    if (!isRomLoaded) return;
    try {
      await loadGameState(appSettings.backendUrl || 'http://localhost:5000');
      addToLog('LOAD', '✓ State loaded');
    } catch (err) {
      addToLog('LOAD', '✗ Failed');
    }
  };

  const handleToggleAgentMode = () => {
    const newMode = !appSettings.agentMode;
    setAppSettings(prev => ({ ...prev, agentMode: newMode }));
    setAgentStatus(prev => ({
      ...prev,
      currentAction: newMode ? 'Autonomous play' : 'Manual control',
      lastDecision: newMode ? 'Agent mode enabled' : 'Manual override',
      decisionHistory: [...prev.decisionHistory, `(${new Date().toLocaleTimeString()}) ${newMode ? 'Agent' : 'Manual'} mode`],
    }));
    addToLog(newMode ? 'AGENT' : 'MANUAL', newMode ? 'Agent enabled' : 'Manual control');
  };

  return (
    <ErrorBoundary>
      <div className="game-dashboard">
        {/* Header */}
        <header className="dashboard-header">
          <div className="header-left">
            <div className="logo">
              <GamepadIcon />
              <span>PyBoy Control</span>
            </div>
            {romName && <span className="rom-name">{romName}</span>}
          </div>
          <div className="header-center">
            <div className={`status-indicator ${backendStatus}`}>
              <span className="status-dot"></span>
              <span className="status-text">
                {backendStatus === 'connected' ? '● Connected' : 
                 backendStatus === 'checking' ? '○ Connecting...' : '○ Disconnected'}
              </span>
            </div>
          </div>
          <div className="header-right">
            <button className="header-btn" onClick={handleSaveState} disabled={!isRomLoaded} title="Save State">
              <SaveIcon />
            </button>
            <button className="header-btn" onClick={handleLoadState} disabled={!isRomLoaded} title="Load State">
              <LoadIcon />
            </button>
            <button className="header-btn agent-toggle" onClick={handleToggleAgentMode}>
              {appSettings.agentMode ? '🤖 Agent' : '👤 Manual'}
            </button>
            <button className="header-btn settings-btn" onClick={() => setIsSettingsOpen(true)}>
              ⚙️
            </button>
          </div>
        </header>

        {/* Main Content */}
        <main className="dashboard-main">
          {/* Left Panel - Agent Status */}
          <aside className="panel agent-panel">
            <AgentStatusPanel 
              status={agentStatus}
              aiState={aiState}
              isAgentMode={appSettings.agentMode}
              onTakeover={handleToggleAgentMode}
              onOverride={handleToggleAgentMode}
            />
            
            {/* Memory Inspector */}
            <div className="memory-inspector">
              <h3>🧠 Memory State</h3>
              <div className="memory-grid">
                <div className="memory-item">
                  <span className="label">Position</span>
                  <span className="value">{memoryState.x}, {memoryState.y}</span>
                </div>
                <div className="memory-item">
                  <span className="label">Map</span>
                  <span className="value">{memoryState.map}</span>
                </div>
                <div className="memory-item">
                  <span className="label">Party</span>
                  <span className="value">{memoryState.party.join(', ')}</span>
                </div>
                <div className="memory-item">
                  <span className="label">Money</span>
                  <span className="value">${memoryState.money}</span>
                </div>
                <div className="memory-item">
                  <span className="label">Steps</span>
                  <span className="value">{memoryState.steps}</span>
                </div>
              </div>
            </div>
          </aside>

          {/* Center - Game Screen */}
          <section className="game-section">
            <div className="game-frame">
              <EmulatorScreen
                emulatorMode="gb"
                romName={romName}
                onRomLoad={handleRomLoad}
                aiState={aiState}
                screenImage={gameScreenUrl || undefined}
                streamingStatus={backendStatus === 'connected' ? 'connected' : 
                                 backendStatus === 'checking' ? 'connecting' : 'disconnected'}
              />
            </div>
            
            {/* Controls */}
            <Controls lastAction={lastAction} />
          </section>

          {/* Right Panel - Action Log */}
          <aside className="panel log-panel">
            <div className="log-header">
              <h3>📋 Action Log</h3>
            </div>
            <div className="log-content">
              {actionLog.length === 0 ? (
                <p className="log-empty">No actions yet...</p>
              ) : (
                actionLog.map((entry, idx) => (
                  <div key={idx} className={`log-entry ${entry.result.includes('✓') ? 'success' : entry.result.includes('✗') ? 'error' : ''}`}>
                    <span className="log-time">{entry.time}</span>
                    <span className="log-action">{entry.action}</span>
                    <span className="log-result">{entry.result}</span>
                  </div>
                ))
              )}
            </div>
          </aside>
        </main>

        {/* Settings Modal */}
        <SettingsModal 
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={appSettings}
          onSettingsChange={setAppSettings}
        />
      </div>
    </ErrorBoundary>
  );
};

export default App;