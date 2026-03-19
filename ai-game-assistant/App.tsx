import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Bot,
  ChevronRight,
  FolderOpen,
  Gamepad2,
  Keyboard,
  RefreshCw,
  Save,
  Settings,
  ShieldAlert,
  Upload,
} from 'lucide-react';
import apiService, {
  type AgentStatus,
  type AppSettings,
  type GameButton,
  type GameState,
  type HealthResponse,
  type InventoryData,
  type LogEntry,
  type MemoryWatch,
  type OpenClawHealthResponse,
  type PartyData,
} from './services/apiService';
import InventoryPanel from './src/components/InventoryPanel';
import PartyPanel from './src/components/PartyPanel';
import SettingsModal from './src/components/SettingsModal';
import { DEFAULT_SETTINGS, loadSettings, saveSettings } from './services/webUiSettings';

const STATUS_REFRESH_MS = 3000;
const SCREEN_REFRESH_MS = 500;
const MEMORY_REFRESH_MS = 5000;
const LAST_ROM_STORAGE_KEY = 'openclaw_webui_last_rom_name';

const EMPTY_GAME_STATE: GameState = {
  running: false,
  rom_loaded: false,
  rom_name: '',
  screen_available: false,
  frame_count: 0,
  fps: 0,
  emulator: 'gb',
  timestamp: '',
};

const EMPTY_AGENT_STATE: AgentStatus = {
  connected: false,
  agent_name: 'OpenClaw',
  mode: 'manual',
  autonomous_level: 'moderate',
  current_action: '',
  last_decision: '',
  enabled: false,
  game_running: false,
  timestamp: '',
};

const EMPTY_MEMORY_STATE: MemoryWatch = {
  addresses: [],
  values: [],
  timestamp: '',
};

type ConnectionStatus = 'checking' | 'connected' | 'disconnected';
type InsightTab = 'party' | 'inventory' | 'memory';
type StatusTone = ConnectionStatus | 'standby';
type MetricTone = 'berry' | 'olive' | 'amber' | 'slate';

const createLogEntry = (type: LogEntry['type'], message: string): LogEntry => ({
  id: Date.now() + Math.floor(Math.random() * 1000),
  timestamp: new Date().toISOString(),
  type,
  message,
});

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const formatClock = (value: string | null | undefined) => {
  if (!value) {
    return 'Not yet';
  }

  return new Date(value).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  });
};

const capitalize = (value: string) => (value ? value.charAt(0).toUpperCase() + value.slice(1) : value);

const summarizeProbeError = (message: string | null | undefined, fallback = 'Unavailable') => {
  if (!message) {
    return fallback;
  }

  if (message.includes('404')) {
    return 'Route missing';
  }

  if (message.includes('429')) {
    return 'Rate limited';
  }

  return fallback;
};

const getBackendTone = (autoConnect: boolean, status: ConnectionStatus): StatusTone => {
  if (!autoConnect) {
    return 'standby';
  }

  return status;
};

const getOpenClawTone = (autoConnect: boolean, health: OpenClawHealthResponse | null): StatusTone => {
  if (!autoConnect) {
    return 'standby';
  }

  if (!health) {
    return 'checking';
  }

  return health.ok ? 'connected' : 'disconnected';
};

const App: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [objectiveDraft, setObjectiveDraft] = useState(() => loadSettings().agentObjectives);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking');
  const [backendHealth, setBackendHealth] = useState<HealthResponse | null>(null);
  const [openClawHealth, setOpenClawHealth] = useState<OpenClawHealthResponse | null>(null);
  const [gameState, setGameState] = useState<GameState>(EMPTY_GAME_STATE);
  const [agentState, setAgentState] = useState<AgentStatus>(EMPTY_AGENT_STATE);
  const [memoryState, setMemoryState] = useState<MemoryWatch>(EMPTY_MEMORY_STATE);
  const [gameScreenUrl, setGameScreenUrl] = useState<string | null>(null);
  const [decisionLog, setDecisionLog] = useState<LogEntry[]>([]);
  const [systemLog, setSystemLog] = useState<LogEntry[]>([]);
  const [lastButtonPressed, setLastButtonPressed] = useState<GameButton | null>(null);
  const [isApplyingRuntime, setIsApplyingRuntime] = useState(false);
  const [isRomLoading, setIsRomLoading] = useState(false);
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null);
  const [partyData, setPartyData] = useState<PartyData | null>(null);
  const [inventoryData, setInventoryData] = useState<InventoryData | null>(null);
  const [insightTab, setInsightTab] = useState<InsightTab>('party');
  const [lastRomName, setLastRomName] = useState<string | null>(() => localStorage.getItem(LAST_ROM_STORAGE_KEY));

  const lastDecisionRef = useRef(agentState.last_decision);
  const lastActionRef = useRef(agentState.current_action);
  const connectionStatusRef = useRef<ConnectionStatus>('checking');

  const appendSystemLog = useCallback((type: LogEntry['type'], message: string) => {
    setSystemLog((entries) => [...entries.slice(-59), createLogEntry(type, message)]);
  }, []);

  const appendDecisionLog = useCallback((type: LogEntry['type'], message: string) => {
    setDecisionLog((entries) => [...entries.slice(-79), createLogEntry(type, message)]);
  }, []);

  const setConnection = useCallback((nextStatus: ConnectionStatus) => {
    connectionStatusRef.current = nextStatus;
    setConnectionStatus(nextStatus);
  }, []);

  const syncOpenClawRuntime = useCallback(
    async (nextSettings: AppSettings, objective: string, successMessage: string) => {
      setIsApplyingRuntime(true);
      apiService.setBaseUrl(nextSettings.backendUrl);

      try {
        const [openClawConfig, modeResponse] = await Promise.all([
          apiService.updateOpenClawConfig({
            endpoint: nextSettings.openclawMcpEndpoint,
            vision_model: nextSettings.visionModel,
            objectives: objective,
            personality: nextSettings.agentPersonality,
          }),
          apiService.setAgentMode({
            mode: nextSettings.agentMode ? 'auto' : 'manual',
            enabled: nextSettings.agentMode,
            autonomous_level: nextSettings.autonomousLevel,
          }),
        ]);

        setOpenClawHealth((current) => (current ? { ...current, endpoint: openClawConfig.endpoint } : current));
        setAgentState((current) => ({
          ...current,
          agent_name: 'OpenClaw',
          enabled: modeResponse.enabled,
          mode: modeResponse.mode,
          autonomous_level: modeResponse.autonomous_level,
          current_action: modeResponse.current_action || current.current_action,
          last_decision: modeResponse.last_decision || current.last_decision,
          timestamp: modeResponse.timestamp,
        }));
        setLastSyncedAt(new Date().toISOString());
        appendSystemLog('system', successMessage);
      } catch (error) {
        appendSystemLog('error', error instanceof Error ? error.message : 'Failed to sync OpenClaw runtime');
        throw error;
      } finally {
        setIsApplyingRuntime(false);
      }
    },
    [appendSystemLog],
  );

  const refreshScreen = useCallback(async () => {
    if (connectionStatusRef.current !== 'connected') {
      return;
    }

    try {
      const screen = await apiService.getScreen();
      setGameScreenUrl(`data:image/png;base64,${screen.image}`);
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : 'Failed to refresh emulator screen');
    }
  }, [appendSystemLog]);

  const refreshMemory = useCallback(async () => {
    if (connectionStatusRef.current !== 'connected') {
      return;
    }

    try {
      setMemoryState(await apiService.getMemoryWatch());
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : 'Failed to refresh memory watch');
    }
  }, [appendSystemLog]);

  const refreshStatus = useCallback(async () => {
    apiService.setBaseUrl(settings.backendUrl);

    try {
      const [health, nextGameState, nextAgentState] = await Promise.all([
        apiService.getHealth(),
        apiService.getGameState(),
        apiService.getAgentStatus(),
      ]);

      setBackendHealth(health);
      setGameState(nextGameState);
      setAgentState(nextAgentState);

      const wasDisconnected = connectionStatusRef.current !== 'connected';
      setConnection('connected');

      if (wasDisconnected) {
        appendSystemLog('system', 'Connected to backend');
      }

      if (nextAgentState.last_decision && nextAgentState.last_decision !== lastDecisionRef.current) {
        appendDecisionLog('thought', nextAgentState.last_decision);
        lastDecisionRef.current = nextAgentState.last_decision;
      }

      if (nextAgentState.current_action && nextAgentState.current_action !== lastActionRef.current) {
        appendDecisionLog('action', `Action: ${nextAgentState.current_action}`);
        lastActionRef.current = nextAgentState.current_action;
      }

      if (nextGameState.rom_name) {
        localStorage.setItem(LAST_ROM_STORAGE_KEY, nextGameState.rom_name);
        setLastRomName(nextGameState.rom_name);
      }

      if (nextGameState.rom_loaded) {
        void refreshScreen();
        void refreshMemory();
      } else {
        setGameScreenUrl(null);
        setMemoryState(EMPTY_MEMORY_STATE);
      }
    } catch (error) {
      const wasDisconnected = connectionStatusRef.current === 'disconnected';
      setConnection('disconnected');
      setBackendHealth(null);
      setGameState(EMPTY_GAME_STATE);
      setAgentState(EMPTY_AGENT_STATE);
      setMemoryState(EMPTY_MEMORY_STATE);
      setGameScreenUrl(null);
      lastDecisionRef.current = EMPTY_AGENT_STATE.last_decision;
      lastActionRef.current = EMPTY_AGENT_STATE.current_action;

      if (!wasDisconnected) {
        appendSystemLog('error', error instanceof Error ? error.message : 'Failed to reach backend');
      }
    }

    try {
      const nextOpenClawHealth = await apiService.checkOpenClawHealth(settings.openclawMcpEndpoint);
      setOpenClawHealth(nextOpenClawHealth);
    } catch (error) {
      setOpenClawHealth({
        ok: false,
        endpoint: settings.openclawMcpEndpoint,
        status: null,
        error: error instanceof Error ? error.message : 'Failed to reach OpenClaw',
        checked_at: new Date().toISOString(),
      });
    }
  }, [appendDecisionLog, appendSystemLog, refreshMemory, refreshScreen, setConnection, settings.backendUrl, settings.openclawMcpEndpoint]);

  useEffect(() => {
    if (!settings.autoConnect) {
      setConnection('disconnected');
      setBackendHealth(null);
      setOpenClawHealth(null);
      return;
    }

    void (async () => {
      try {
        await syncOpenClawRuntime(settings, settings.agentObjectives, 'Applied saved OpenClaw runtime defaults');
      } catch {
        // The status poll below exposes the backend and OpenClaw state.
      } finally {
        await refreshStatus();
      }
    })();
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshStatus();
    }, STATUS_REFRESH_MS);

    return () => window.clearInterval(intervalId);
  }, [refreshStatus]);

  useEffect(() => {
    if (!gameState.rom_loaded || connectionStatus !== 'connected') {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshScreen();
    }, SCREEN_REFRESH_MS);

    return () => window.clearInterval(intervalId);
  }, [connectionStatus, gameState.rom_loaded, refreshScreen]);

  useEffect(() => {
    if (!gameState.rom_loaded || connectionStatus !== 'connected') {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      void refreshMemory();
    }, MEMORY_REFRESH_MS);

    return () => window.clearInterval(intervalId);
  }, [connectionStatus, gameState.rom_loaded, refreshMemory]);

  const handleSettingsSave = async (nextSettings: AppSettings) => {
    const normalizedObjective = objectiveDraft.trim() || nextSettings.agentObjectives || DEFAULT_SETTINGS.agentObjectives;
    const mergedSettings = { ...nextSettings, agentObjectives: normalizedObjective };

    setSettings(mergedSettings);
    setObjectiveDraft(normalizedObjective);
    saveSettings(mergedSettings);

    try {
      await syncOpenClawRuntime(mergedSettings, normalizedObjective, 'Updated backend and OpenClaw settings');
    } catch {
      // Runtime sync failure is already logged, but settings stay persisted locally.
    }

    await refreshStatus();
  };

  const engageManualOverride = useCallback(async (reason: string) => {
    const nextSettings = { ...settings, agentMode: false };
    setSettings(nextSettings);
    saveSettings(nextSettings);

    await syncOpenClawRuntime(nextSettings, objectiveDraft.trim() || settings.agentObjectives, reason);
    await refreshStatus();
  }, [objectiveDraft, refreshStatus, settings, syncOpenClawRuntime]);

  const resumeOpenClawControl = useCallback(async () => {
    const objective = objectiveDraft.trim() || DEFAULT_SETTINGS.agentObjectives;
    const nextSettings = { ...settings, agentMode: true, agentObjectives: objective };

    setSettings(nextSettings);
    saveSettings(nextSettings);

    await syncOpenClawRuntime(nextSettings, objective, 'Resumed OpenClaw control');
    await refreshStatus();
  }, [objectiveDraft, refreshStatus, settings, syncOpenClawRuntime]);

  const applyObjective = useCallback(async () => {
    const objective = objectiveDraft.trim() || DEFAULT_SETTINGS.agentObjectives;
    const nextSettings = { ...settings, agentObjectives: objective };

    setSettings(nextSettings);
    saveSettings(nextSettings);
    await syncOpenClawRuntime(nextSettings, objective, 'Updated OpenClaw objective');
    await refreshStatus();
  }, [objectiveDraft, refreshStatus, settings, syncOpenClawRuntime]);

  const handleButtonPress = useCallback(async (button: GameButton) => {
    if (!gameState.rom_loaded || connectionStatus !== 'connected') {
      return;
    }

    setLastButtonPressed(button);

    try {
      if (agentState.enabled) {
        await engageManualOverride(`Manual override engaged by ${button}`);
      }

      await apiService.pressButton(button);
      appendSystemLog('action', `Pressed ${button}`);
      void refreshScreen();
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : `Failed to press ${button}`);
    } finally {
      window.setTimeout(() => setLastButtonPressed(null), 180);
    }
  }, [agentState.enabled, appendSystemLog, connectionStatus, engageManualOverride, gameState.rom_loaded, refreshScreen]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const keyboardMap: Record<string, GameButton> = {
        ArrowUp: 'UP',
        ArrowDown: 'DOWN',
        ArrowLeft: 'LEFT',
        ArrowRight: 'RIGHT',
        z: 'A',
        x: 'B',
        Enter: 'START',
        Shift: 'SELECT',
      };

      if (event.key === '?') {
        event.preventDefault();
        setShowKeyboardHelp((value) => !value);
        return;
      }

      const mappedButton = keyboardMap[event.key];
      if (!mappedButton) {
        return;
      }

      event.preventDefault();
      void handleButtonPress(mappedButton);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleButtonPress]);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setIsRomLoading(true);

    try {
      const response = await apiService.uploadRom(file, settings.emulatorType, settings.launchUiOnRomLoad);
      appendSystemLog('system', response.message || `Loaded ${file.name}`);
      localStorage.setItem(LAST_ROM_STORAGE_KEY, file.name);
      setLastRomName(file.name);
      await refreshStatus();
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : `Failed to load ${file.name}`);
    } finally {
      setIsRomLoading(false);
      event.target.value = '';
    }
  };

  const handleSaveState = async () => {
    try {
      await apiService.saveState();
      appendSystemLog('system', 'Saved emulator state');
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : 'Failed to save state');
    }
  };

  const handleLoadState = async () => {
    try {
      await apiService.loadState();
      appendSystemLog('system', 'Loaded emulator state');
      await refreshStatus();
    } catch (error) {
      appendSystemLog('error', error instanceof Error ? error.message : 'Failed to load state');
    }
  };

  const backendTone = getBackendTone(settings.autoConnect, connectionStatus);
  const openClawTone = getOpenClawTone(settings.autoConnect, openClawHealth);
  const isRuntimeReady = connectionStatus === 'connected' && gameState.rom_loaded;
  const latestDecision =
    agentState.last_decision.trim() ||
    (connectionStatus === 'connected'
      ? 'Waiting for runtime data from OpenClaw.'
      : 'Connect the backend to receive decisions and actions.');
  const controlModeLabel = !settings.autoConnect && connectionStatus !== 'connected'
    ? 'Standby'
    : agentState.enabled
      ? 'OpenClaw Auto'
      : isRuntimeReady
        ? 'Manual Override'
        : connectionStatus === 'connected'
          ? 'Awaiting ROM'
          : 'Awaiting Link';
  const controlModeHint = agentState.enabled
    ? `${capitalize(agentState.autonomous_level)} autonomy`
    : isRuntimeReady
      ? 'Manual input keeps OpenClaw paused'
      : 'Connect and load a ROM to begin';
  const currentActionLabel = agentState.current_action.trim() || (agentState.enabled ? 'Thinking' : 'Idle');
  const backendStatusLabel = backendTone === 'standby'
    ? 'Standby'
    : backendTone === 'checking'
      ? 'Checking'
      : backendTone === 'connected'
        ? capitalize(backendHealth?.status || 'connected')
        : 'Offline';
  const openClawStatusLabel = openClawTone === 'standby'
    ? 'Standby'
    : openClawTone === 'checking'
      ? 'Checking'
      : openClawTone === 'connected'
        ? 'Linked'
        : summarizeProbeError(openClawHealth?.error, 'Needs attention');
  const romLabel = gameState.rom_loaded
    ? gameState.rom_name
    : lastRomName
      ? `Last: ${lastRomName}`
      : 'No cartridge';
  const syncLabel = isApplyingRuntime
    ? 'Syncing...'
    : lastSyncedAt
      ? formatClock(lastSyncedAt)
      : 'Pending';

  return (
    <div className="app">
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSettingsSave}
      />

      {showKeyboardHelp && <KeyboardHelp onClose={() => setShowKeyboardHelp(false)} />}

      <header className="app-header">
        <div className="app-header__title">
          <span className="app-header__eyebrow">OpenClaw Orchestrator</span>
          <h1>Game Boy Control Desk</h1>
          <p>
            Gameplay stays on the handheld. OpenClaw handles orchestration, objective sync, and autonomous control from the sidecar.
          </p>
        </div>

        <div className="app-header__actions">
          <StatusPill label="Backend" detail={backendStatusLabel} status={backendTone} />
          <StatusPill label="OpenClaw" detail={openClawStatusLabel} status={openClawTone} />
          <button
            type="button"
            onClick={() => setShowKeyboardHelp(true)}
            className="icon-button"
          >
            <Keyboard className="icon-button__icon" />
            <span>Keys</span>
          </button>
          <button
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            className="icon-button icon-button--primary"
          >
            <Settings className="icon-button__icon" />
            <span>Settings</span>
          </button>
        </div>
      </header>

      <section className="hero-strip">
        <HeroMetric
          tone="berry"
          label="Control"
          value={controlModeLabel}
          meta={controlModeHint}
        />
        <HeroMetric
          tone="olive"
          label="Action"
          value={currentActionLabel}
          meta={agentState.mode ? `${capitalize(agentState.mode)} mode` : 'Waiting for runtime'}
        />
        <HeroMetric
          tone="amber"
          label="ROM"
          value={romLabel}
          meta={gameState.rom_loaded ? `${gameState.frame_count.toLocaleString()} frames captured` : 'Load a ROM to start play'}
        />
        <HeroMetric
          tone="slate"
          label="Sync"
          value={syncLabel}
          meta={lastSyncedAt ? 'WebUI defaults pushed to OpenClaw' : 'No runtime sync recorded yet'}
        />
      </section>

      <main className="desk-grid">
        <aside className="desk-column desk-column--mission">
          <PanelCard
            eyebrow="Mission Control"
            title="OpenClaw Directives"
            subtitle="Set the objective and takeover rules here. The handheld stays focused on gameplay."
            action={<Bot className="panel-card__icon" />}
          >
            <div className="runtime-chip-row">
              <span className={classNames('runtime-chip', agentState.enabled ? 'runtime-chip--berry' : 'runtime-chip--muted')}>
                {controlModeLabel}
              </span>
              <span className="runtime-chip runtime-chip--olive">{capitalize(settings.agentPersonality)}</span>
              <span className={classNames('runtime-chip', gameState.rom_loaded ? 'runtime-chip--success' : 'runtime-chip--muted')}>
                {gameState.rom_loaded ? 'ROM active' : 'No ROM'}
              </span>
            </div>

            <label className="form-field">
              <span className="form-field__label">Objective</span>
              <textarea
                value={objectiveDraft}
                onChange={(event) => setObjectiveDraft(event.target.value)}
                rows={6}
                className="text-area"
                placeholder="Describe what OpenClaw should optimize for."
              />
              <span className="form-field__help">
                This defines OpenClaw behavior, but it should not become visible as gameplay UI.
              </span>
            </label>

            <div className="action-row">
              <button
                type="button"
                onClick={() => {
                  void applyObjective().catch(() => undefined);
                }}
                disabled={isApplyingRuntime}
                className="action-button action-button--primary"
              >
                Apply Objective
              </button>
              <button
                type="button"
                onClick={() => {
                  void resumeOpenClawControl().catch(() => undefined);
                }}
                disabled={isApplyingRuntime || !gameState.rom_loaded}
                className="action-button action-button--secondary"
              >
                Resume OpenClaw
              </button>
              <button
                type="button"
                onClick={() => {
                  void engageManualOverride('Manual override enabled from WebUI').catch(() => undefined);
                }}
                disabled={isApplyingRuntime || !gameState.rom_loaded}
                className="action-button action-button--warning"
              >
                Manual Override
              </button>
            </div>

            <div className="decision-callout">
              <span className="decision-callout__label">Latest runtime decision</span>
              <p>{latestDecision}</p>
              <span className="decision-callout__meta">
                {lastSyncedAt ? `Last sync ${formatClock(lastSyncedAt)}` : 'Runtime sync happens after save or resume'}
              </span>
            </div>
          </PanelCard>

          <PanelCard
            eyebrow="Decision Feed"
            title="Live Agent Log"
            subtitle="Thoughts and actions from backend state updates."
            action={<Activity className="panel-card__icon" />}
          >
            <div className="log-list">
              {decisionLog.length === 0 ? (
                <EmptyLog message="Waiting for the first decision." />
              ) : (
                decisionLog.slice().reverse().map((entry) => (
                  <LogRow key={entry.id} entry={entry} />
                ))
              )}
            </div>
          </PanelCard>
        </aside>

        <section className="desk-column desk-column--console">
          <section className="console-shell">
            <div className="console-shell__top">
              <div>
                <span className="console-shell__eyebrow">Gameplay Surface</span>
                <h2>Handheld View</h2>
                <p>The screen and physical controls stay central. Orchestration lives around the edges.</p>
              </div>
              <div className="console-shell__sync">
                <span className="console-shell__sync-label">Mode</span>
                <strong>{controlModeLabel}</strong>
              </div>
            </div>

            <div className="screen-bezel">
              <div className="screen-bezel__label">
                <span className="screen-bezel__led screen-bezel__led--berry" />
                DOT MATRIX MISSION SCREEN
              </div>

              <div className="screen-bezel__frame">
                {gameScreenUrl ? (
                  <img
                    src={gameScreenUrl}
                    alt="Emulator screen"
                    className="screen-bezel__image"
                    style={{ imageRendering: 'pixelated', aspectRatio: '160 / 144' }}
                  />
                ) : (
                  <div className="screen-bezel__empty">
                    <Gamepad2 className="screen-bezel__empty-icon" />
                    <p className="screen-bezel__empty-title">No active screen</p>
                    <p className="screen-bezel__empty-copy">
                      Load a ROM and the backend will stream frames into the handheld view.
                    </p>
                  </div>
                )}
              </div>

              <div className="screen-bezel__meta">
                <span>Screen {gameState.screen_available ? 'live' : 'idle'}</span>
                <span>{Math.round(gameState.fps || 0)} FPS</span>
                <span>{gameState.frame_count.toLocaleString()} frames</span>
                <span>{(gameState.emulator || settings.emulatorType).toUpperCase()}</span>
              </div>
            </div>

            <div className="console-toolbar">
              <label className="toolbar-button toolbar-button--primary">
                <Upload className="toolbar-button__icon" />
                <span>{isRomLoading ? 'Loading...' : 'Load ROM'}</span>
                <input
                  type="file"
                  accept=".gb,.gbc,.gba,.zip"
                  className="visually-hidden"
                  onChange={handleFileSelect}
                />
              </label>

              <ToolbarButton icon={Save} label="Save" disabled={!gameState.rom_loaded} onClick={handleSaveState} />
              <ToolbarButton icon={FolderOpen} label="Load" disabled={!gameState.rom_loaded} onClick={handleLoadState} />
              <ToolbarButton
                icon={RefreshCw}
                label="Refresh"
                disabled={connectionStatus === 'disconnected'}
                onClick={() => void refreshStatus()}
              />
            </div>

            <div className="signal-grid">
              <SignalCard
                label="Backend"
                value={backendStatusLabel}
                meta={backendHealth?.service || settings.backendUrl}
                tone={backendTone}
              />
              <SignalCard
                label="OpenClaw"
                value={openClawStatusLabel}
                meta={openClawHealth?.ok ? 'Health verified through backend' : openClawHealth?.error || 'Waiting for backend health proxy'}
                tone={openClawTone}
              />
              <SignalCard
                label="ROM"
                value={gameState.rom_loaded ? 'Loaded' : lastRomName ? 'Ready to reload' : 'Idle'}
                meta={gameState.rom_loaded ? gameState.rom_name : lastRomName || 'Insert a cartridge to play'}
                tone={gameState.rom_loaded ? 'connected' : 'standby'}
              />
            </div>

            <div className="control-deck">
              <div className="control-deck__guide">
                <div className="control-deck__guide-header">
                  <span className="control-deck__eyebrow">Manual Deck</span>
                  {agentState.enabled ? (
                    <span className="control-deck__notice control-deck__notice--warning">
                      <ShieldAlert className="control-deck__notice-icon" />
                      Manual input will pause OpenClaw first
                    </span>
                  ) : (
                    <span className="control-deck__notice">OpenClaw is currently paused</span>
                  )}
                </div>
                <p className="control-deck__copy">
                  Use the physical controls below for deliberate intervention. Keyboard shortcuts stay mapped to the original handheld layout.
                </p>
                <div className="keyboard-chip-row">
                  <span className="keyboard-chip">Arrows = D-pad</span>
                  <span className="keyboard-chip">Z = A</span>
                  <span className="keyboard-chip">X = B</span>
                  <span className="keyboard-chip">Enter = Start</span>
                  <span className="keyboard-chip">Shift = Select</span>
                </div>
              </div>

              <ControllerPad
                disabled={!gameState.rom_loaded || connectionStatus !== 'connected'}
                lastButton={lastButtonPressed}
                onPress={(button) => void handleButtonPress(button)}
              />
            </div>
          </section>
        </section>

        <aside className="desk-column desk-column--insights">
          <PanelCard
            eyebrow="Runtime State"
            title="Party, Bag, and Memory"
            subtitle="Operational data stays readable without intruding on the handheld."
            action={
              <div className="insight-tabs">
                <InsightButton active={insightTab === 'party'} label="Party" onClick={() => setInsightTab('party')} />
                <InsightButton active={insightTab === 'inventory'} label="Inventory" onClick={() => setInsightTab('inventory')} />
                <InsightButton active={insightTab === 'memory'} label="Memory" onClick={() => setInsightTab('memory')} />
              </div>
            }
          >
            <p className="panel-card__note">
              These panels are tools for the operator, not overlays on the game itself.
            </p>
          </PanelCard>

          {insightTab === 'party' && (
            <PartyPanel
              isRomLoaded={gameState.rom_loaded}
              onPartyUpdate={setPartyData}
            />
          )}

          {insightTab === 'inventory' && (
            <InventoryPanel
              isRomLoaded={gameState.rom_loaded}
              onInventoryUpdate={setInventoryData}
            />
          )}

          {insightTab === 'memory' && (
            <MemorySummaryCard
              isRomLoaded={gameState.rom_loaded}
              memoryState={memoryState}
            />
          )}

          <PanelCard
            eyebrow="Quick Read"
            title="Session Snapshot"
            subtitle="High-signal numbers without opening additional tools."
          >
            <div className="summary-grid">
              <SummaryStat label="Party size" value={partyData ? String(partyData.party_count) : '--'} />
              <SummaryStat label="Bag items" value={inventoryData ? String(inventoryData.item_count) : '--'} />
              <SummaryStat label="FPS" value={String(Math.round(gameState.fps || 0))} />
              <SummaryStat label="Frame count" value={gameState.frame_count.toLocaleString()} />
            </div>
          </PanelCard>

          <PanelCard
            eyebrow="System Log"
            title="Backend Events"
            subtitle="Connection changes, ROM actions, and manual overrides."
            action={<RefreshCw className="panel-card__icon" />}
          >
            <div className="log-list">
              {systemLog.length === 0 ? (
                <EmptyLog message="System log is quiet." />
              ) : (
                systemLog.slice().reverse().map((entry) => (
                  <LogRow key={entry.id} entry={entry} />
                ))
              )}
            </div>
          </PanelCard>
        </aside>
      </main>
    </div>
  );
};

interface PanelCardProps {
  eyebrow: string;
  title: string;
  subtitle: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}

const PanelCard: React.FC<PanelCardProps> = ({ eyebrow, title, subtitle, action, children }) => (
  <section className="panel-card">
    <div className="panel-card__header">
      <div className="panel-card__title-group">
        <span className="panel-card__eyebrow">{eyebrow}</span>
        <h2 className="panel-card__title">{title}</h2>
        <p className="panel-card__subtitle">{subtitle}</p>
      </div>
      {action ? <div className="panel-card__action">{action}</div> : null}
    </div>
    <div className="panel-card__body">{children}</div>
  </section>
);

const HeroMetric: React.FC<{
  tone: MetricTone;
  label: string;
  value: string;
  meta: string;
}> = ({ tone, label, value, meta }) => (
  <article className={classNames('hero-metric', `hero-metric--${tone}`)}>
    <span className="hero-metric__label">{label}</span>
    <strong className="hero-metric__value">{value}</strong>
    <span className="hero-metric__meta">{meta}</span>
  </article>
);

const ToolbarButton: React.FC<{
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  disabled?: boolean;
  onClick: () => void;
}> = ({ icon: Icon, label, disabled, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    className="toolbar-button"
  >
    <Icon className="toolbar-button__icon" />
    <span>{label}</span>
  </button>
);

const StatusPill: React.FC<{
  label: string;
  detail: string;
  status: StatusTone;
}> = ({ label, detail, status }) => (
  <span className={classNames('status-pill', `status-pill--${status}`)}>
    <span className="status-pill__dot" />
    <span className="status-pill__copy">
      <strong>{label}</strong>
      <small>{detail}</small>
    </span>
  </span>
);

const SignalCard: React.FC<{
  label: string;
  value: string;
  meta: string;
  tone: StatusTone;
}> = ({ label, value, meta, tone }) => (
  <article className={classNames('signal-card', `signal-card--${tone}`)}>
    <span className="signal-card__label">{label}</span>
    <strong className="signal-card__value">{value}</strong>
    <span className="signal-card__meta">{meta}</span>
  </article>
);

const SummaryStat: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <article className="summary-stat">
    <span className="summary-stat__label">{label}</span>
    <strong className="summary-stat__value">{value}</strong>
  </article>
);

const LogRow: React.FC<{ entry: LogEntry }> = ({ entry }) => (
  <article className={classNames('log-entry', `log-entry--${entry.type}`)}>
    <div className="log-entry__meta">
      <span>{entry.type}</span>
      <span>{formatClock(entry.timestamp)}</span>
    </div>
    <p className="log-entry__message">{entry.message}</p>
  </article>
);

const EmptyLog: React.FC<{ message: string }> = ({ message }) => (
  <div className="empty-log">
    {message}
  </div>
);

const InsightButton: React.FC<{ active: boolean; label: string; onClick: () => void }> = ({ active, label, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className={classNames('insight-tab', active && 'insight-tab--active')}
  >
    {label}
  </button>
);

const MemorySummaryCard: React.FC<{ isRomLoaded: boolean; memoryState: MemoryWatch }> = ({ isRomLoaded, memoryState }) => (
  <section className="data-panel">
    <div className="data-panel__header">
      <div>
        <span className="data-panel__eyebrow">Memory Watch</span>
        <h3 className="data-panel__title">Watched Addresses</h3>
        <p className="data-panel__subtitle">Backend watch values are shown here for operator debugging.</p>
      </div>
      <ChevronRight className="data-panel__icon" />
    </div>

    <div className="data-panel__body">
      {!isRomLoaded ? (
        <div className="empty-panel">
          Load a ROM to inspect watched memory.
        </div>
      ) : memoryState.values.length === 0 ? (
        <div className="empty-panel">
          No watched addresses were returned.
        </div>
      ) : (
        <div className="memory-list">
          {memoryState.values.map((value) => (
            <div key={value.address} className="memory-row">
              <div className="memory-row__meta">
                <strong>{value.name}</strong>
                <span>0x{value.address.toString(16).toUpperCase()}</span>
              </div>
              <div className="memory-row__value">
                <strong>{value.hex}</strong>
                <span>{value.value ?? '--'}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </section>
);

const ControllerPad: React.FC<{
  disabled: boolean;
  lastButton: GameButton | null;
  onPress: (button: GameButton) => void;
}> = ({ disabled, lastButton, onPress }) => {
  const buttons: Array<{ label: string; action: GameButton }> = [
    { label: 'A', action: 'A' },
    { label: 'B', action: 'B' },
  ];

  const directions: Array<{ label: string; action: GameButton; gridArea: string }> = [
    { label: '↑', action: 'UP', gridArea: '1 / 2' },
    { label: '←', action: 'LEFT', gridArea: '2 / 1' },
    { label: '→', action: 'RIGHT', gridArea: '2 / 3' },
    { label: '↓', action: 'DOWN', gridArea: '3 / 2' },
  ];

  return (
    <div className="controller-pad">
      <div className="controller-pad__dpad">
        {directions.map((direction) => (
          <button
            key={direction.action}
            type="button"
            style={{ gridArea: direction.gridArea }}
            onClick={() => onPress(direction.action)}
            disabled={disabled}
            className={classNames(
              'controller-key',
              lastButton === direction.action && 'controller-key--active',
            )}
          >
            {direction.label}
          </button>
        ))}
      </div>

      <div className="controller-pad__actions">
        <div className="controller-pad__ab">
          {buttons.map((button) => (
            <button
              key={button.action}
              type="button"
              onClick={() => onPress(button.action)}
              disabled={disabled}
              className={classNames(
                'controller-key',
                'controller-key--round',
                lastButton === button.action && 'controller-key--active',
              )}
            >
              {button.label}
            </button>
          ))}
        </div>

        <div className="controller-pad__utility">
          <SmallControllerButton action="SELECT" label="Select" disabled={disabled} lastButton={lastButton} onPress={onPress} />
          <SmallControllerButton action="START" label="Start" disabled={disabled} lastButton={lastButton} onPress={onPress} />
        </div>
      </div>
    </div>
  );
};

const SmallControllerButton: React.FC<{
  action: GameButton;
  label: string;
  disabled: boolean;
  lastButton: GameButton | null;
  onPress: (button: GameButton) => void;
}> = ({ action, label, disabled, lastButton, onPress }) => (
  <button
    type="button"
    onClick={() => onPress(action)}
    disabled={disabled}
    className={classNames(
      'controller-key',
      'controller-key--utility',
      lastButton === action && 'controller-key--active',
    )}
  >
    {label}
  </button>
);

const KeyboardHelp: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <div className="modal-backdrop" role="presentation">
    <div className="modal-scrim" onClick={onClose} />
    <div className="modal modal--keyboard" role="dialog" aria-modal="true" aria-labelledby="keyboard-help-title">
      <div className="modal__header">
        <div>
          <span className="modal__eyebrow">Manual Controls</span>
          <h2 id="keyboard-help-title">Keyboard shortcuts</h2>
        </div>
        <button type="button" onClick={onClose} className="action-button action-button--ghost">
          Close
        </button>
      </div>

      <div className="modal__body">
        <div className="shortcut-grid">
          <ShortcutRow keys="↑ ↓ ← →" action="D-pad" />
          <ShortcutRow keys="Z / X" action="A / B" />
          <ShortcutRow keys="Enter" action="Start" />
          <ShortcutRow keys="Shift" action="Select" />
          <ShortcutRow keys="?" action="Toggle this help" />
        </div>
      </div>
    </div>
  </div>
);

const ShortcutRow: React.FC<{ keys: string; action: string }> = ({ keys, action }) => (
  <div className="shortcut-card">
    <span className="shortcut-card__keys">{keys}</span>
    <strong className="shortcut-card__action">{action}</strong>
  </div>
);

export default App;
