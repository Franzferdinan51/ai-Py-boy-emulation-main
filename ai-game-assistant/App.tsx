import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Bot,
  ChevronRight,
  FolderOpen,
  Gamepad2,
  Keyboard,
  Package,
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
  current_action: 'Idle',
  last_decision: 'Waiting for runtime data.',
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

const createLogEntry = (type: LogEntry['type'], message: string): LogEntry => ({
  id: Date.now() + Math.floor(Math.random() * 1000),
  timestamp: new Date().toISOString(),
  type,
  message,
});

const App: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [objectiveDraft, setObjectiveDraft] = useState(loadSettings().agentObjectives);
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

        setOpenClawHealth((current) => current ? { ...current, endpoint: openClawConfig.endpoint } : current);
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
      setAgentState((current) => ({ ...current, connected: false }));

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
      return;
    }

    void (async () => {
      try {
        await syncOpenClawRuntime(settings, settings.agentObjectives, 'Applied saved OpenClaw runtime defaults');
      } catch {
        // The status poll below will expose the backend/OpenClaw state.
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
      // Runtime sync failure is already logged, but settings are still persisted locally.
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

  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={handleSettingsSave}
      />

      {showKeyboardHelp && <KeyboardHelp onClose={() => setShowKeyboardHelp(false)} />}

      <header className="sticky top-0 z-30 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-4 lg:px-6">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-900/60 bg-cyan-950/40 text-cyan-400">
              <Bot className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-cyan-400">OpenClaw Game Control</h1>
              <p className="text-sm text-neutral-400">Autonomous play with manual override</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <StatusPill label="Backend" status={connectionStatus} />
            <StatusPill label="OpenClaw" status={openClawHealth?.ok ? 'connected' : 'disconnected'} />
            <button
              onClick={() => setShowKeyboardHelp(true)}
              className="rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-300 transition hover:border-neutral-500 hover:text-white"
            >
              <span className="flex items-center gap-2">
                <Keyboard className="h-4 w-4" />
                Keys
              </span>
            </button>
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="rounded-xl border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-300 transition hover:border-neutral-500 hover:text-white"
            >
              <span className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Settings
              </span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1600px] gap-4 px-4 py-4 lg:grid-cols-[320px_minmax(0,1fr)_360px] lg:px-6">
        <section className="space-y-4">
          <div className="rounded-3xl border border-cyan-900/60 bg-neutral-900/80 p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">OpenClaw Runtime</h2>
                <p className="mt-2 text-sm text-neutral-400">Push objective, vision model, and control mode to backend</p>
              </div>
              <Bot className="h-5 w-5 text-cyan-400" />
            </div>

            <div className="mt-4 grid gap-3">
              <RuntimeStat
                label="Mode"
                value={agentState.enabled ? 'Autonomous' : 'Manual override'}
                hint={`${settings.autonomousLevel} autonomy`}
              />
              <RuntimeStat
                label="Personality"
                value={settings.agentPersonality}
                hint={`Vision ${settings.visionModel}`}
              />
              <RuntimeStat
                label="Current action"
                value={agentState.current_action || 'Idle'}
                hint={agentState.mode || 'manual'}
              />
            </div>

            <label className="mt-5 block">
              <span className="mb-2 block text-sm font-medium text-neutral-200">Objective</span>
              <textarea
                value={objectiveDraft}
                onChange={(event) => setObjectiveDraft(event.target.value)}
                rows={5}
                className="w-full rounded-2xl border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-500"
                placeholder="Describe what OpenClaw should optimize for."
              />
            </label>

            <div className="mt-4 flex flex-wrap gap-3">
              <button
                onClick={() => {
                  void applyObjective().catch(() => undefined);
                }}
                disabled={isApplyingRuntime}
                className="rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-400 disabled:cursor-wait disabled:opacity-60"
              >
                Apply Objective
              </button>
              <button
                onClick={() => {
                  void resumeOpenClawControl().catch(() => undefined);
                }}
                disabled={isApplyingRuntime || !gameState.rom_loaded}
                className="rounded-xl border border-cyan-700 bg-cyan-950/50 px-4 py-2 text-sm font-medium text-cyan-300 transition hover:border-cyan-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                Resume OpenClaw
              </button>
              <button
                onClick={() => {
                  void engageManualOverride('Manual override enabled from WebUI').catch(() => undefined);
                }}
                disabled={isApplyingRuntime || !gameState.rom_loaded}
                className="rounded-xl border border-amber-800 bg-amber-950/40 px-4 py-2 text-sm font-medium text-amber-300 transition hover:border-amber-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                Manual Override
              </button>
            </div>

            <div className="mt-4 rounded-2xl border border-neutral-800 bg-neutral-950/80 px-4 py-3">
              <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">Latest decision</p>
              <p className="mt-2 text-sm text-neutral-300">{agentState.last_decision}</p>
              {lastSyncedAt && (
                <p className="mt-2 text-xs text-neutral-500">Last synced {new Date(lastSyncedAt).toLocaleTimeString()}</p>
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Decision Log</h2>
                <p className="mt-1 text-sm text-neutral-400">Live agent output</p>
              </div>
              <Activity className={`h-4 w-4 ${agentState.enabled ? 'text-green-400' : 'text-neutral-500'}`} />
            </div>

            <div className="mt-4 space-y-2">
              {decisionLog.length === 0 ? (
                <EmptyLog message="Waiting for the first decision." />
              ) : (
                decisionLog.slice().reverse().map((entry) => (
                  <LogRow key={entry.id} entry={entry} />
                ))
              )}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-5 py-4">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Emulator</h2>
                <p className="mt-1 text-sm text-neutral-400">
                  {gameState.rom_loaded
                    ? `${gameState.rom_name} • ${gameState.frame_count.toLocaleString()} frames`
                    : lastRomName
                      ? `Last ROM: ${lastRomName}`
                      : 'No ROM loaded'}
                </p>
              </div>

              <div className="flex flex-wrap gap-2">
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-400">
                  <Upload className="h-4 w-4" />
                  {isRomLoading ? 'Loading...' : 'Load ROM'}
                  <input
                    type="file"
                    accept=".gb,.gbc,.gba,.zip"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                </label>
                <ToolbarButton icon={Save} label="Save" disabled={!gameState.rom_loaded} onClick={handleSaveState} />
                <ToolbarButton icon={FolderOpen} label="Load" disabled={!gameState.rom_loaded} onClick={handleLoadState} />
                <ToolbarButton icon={RefreshCw} label="Refresh" disabled={connectionStatus === 'disconnected'} onClick={() => void refreshStatus()} />
              </div>
            </div>

            <div className="bg-black px-4 py-4">
              <div className="mx-auto flex min-h-[380px] max-w-[720px] items-center justify-center overflow-hidden rounded-[28px] border border-neutral-800 bg-neutral-950">
                {gameScreenUrl ? (
                  <img
                    src={gameScreenUrl}
                    alt="Emulator screen"
                    className="max-h-[72vh] w-full object-contain"
                    style={{ imageRendering: 'pixelated', aspectRatio: '160 / 144' }}
                  />
                ) : (
                  <div className="px-8 py-12 text-center text-neutral-500">
                    <Gamepad2 className="mx-auto mb-4 h-12 w-12 opacity-40" />
                    <p className="text-base font-medium text-neutral-300">No active screen</p>
                    <p className="mt-2 text-sm text-neutral-500">
                      Load a ROM and the backend will start feeding the emulator image here.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {gameState.rom_loaded && gameState.fps > 0 && (
              <div className="flex items-center gap-4 text-sm text-neutral-400">
                <span>{gameState.fps} FPS</span>
                <span>{gameState.frame_count.toLocaleString()} frames</span>
              </div>
            )}
          </div>

          <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Manual Controller</h2>
                <p className="mt-1 text-sm text-neutral-400">Manual input triggers override mode</p>
              </div>
              {agentState.enabled && (
                <div className="flex items-center gap-2 rounded-full border border-amber-800 bg-amber-950/40 px-3 py-1 text-xs text-amber-300">
                  <ShieldAlert className="h-3.5 w-3.5" />
                  Override on input
                </div>
              )}
            </div>

            <div className="mt-5">
              <ControllerPad
                disabled={!gameState.rom_loaded || connectionStatus !== 'connected'}
                lastButton={lastButtonPressed}
                onPress={(button) => void handleButtonPress(button)}
              />
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-neutral-500">
              <span>Keyboard:</span>
              <span className="rounded-full border border-neutral-800 px-2 py-1">Arrows = D-pad</span>
              <span className="rounded-full border border-neutral-800 px-2 py-1">Z = A</span>
              <span className="rounded-full border border-neutral-800 px-2 py-1">X = B</span>
              <span className="rounded-full border border-neutral-800 px-2 py-1">Enter = Start</span>
              <span className="rounded-full border border-neutral-800 px-2 py-1">Shift = Select</span>
            </div>
          </div>

          <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">System Log</h2>
                <p className="mt-1 text-sm text-neutral-400">Connection events and actions</p>
              </div>
              <RefreshCw className="h-4 w-4 text-neutral-500" />
            </div>

            <div className="mt-4 space-y-2">
              {systemLog.length === 0 ? (
                <EmptyLog message="System log is quiet." />
              ) : (
                systemLog.slice().reverse().map((entry) => (
                  <LogRow key={entry.id} entry={entry} />
                ))
              )}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Runtime State</h2>
                <p className="mt-1 text-sm text-neutral-400">Party, inventory, and memory</p>
              </div>
              <Package className="h-4 w-4 text-neutral-500" />
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <InsightButton active={insightTab === 'party'} label="Party" onClick={() => setInsightTab('party')} />
              <InsightButton active={insightTab === 'inventory'} label="Inventory" onClick={() => setInsightTab('inventory')} />
              <InsightButton active={insightTab === 'memory'} label="Memory" onClick={() => setInsightTab('memory')} />
            </div>
          </div>

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
        </section>
      </main>
    </div>
  );
};

const ToolbarButton: React.FC<{
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  disabled?: boolean;
  onClick: () => void;
}> = ({ icon: Icon, label, disabled, onClick }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className="inline-flex items-center gap-2 rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-300 transition hover:border-neutral-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
  >
    <Icon className="h-4 w-4" />
    {label}
  </button>
);

const StatusPill: React.FC<{ label: string; status: ConnectionStatus | 'connected' | 'disconnected' }> = ({ label, status }) => {
  const statusClasses =
    status === 'connected'
      ? 'bg-green-950/60 text-green-300 border-green-900/60'
      : status === 'checking'
        ? 'bg-yellow-950/60 text-yellow-300 border-yellow-900/60'
        : 'bg-red-950/60 text-red-300 border-red-900/60';

  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs uppercase tracking-wide ${statusClasses}`}>
      <span className="h-2 w-2 rounded-full bg-current" />
      {label}
    </span>
  );
};

const RuntimeStat: React.FC<{ label: string; value: string; hint: string }> = ({ label, value, hint }) => (
  <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 px-4 py-3">
    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">{label}</p>
    <p className="mt-2 text-sm font-medium text-white">{value}</p>
    <p className="mt-1 text-xs text-neutral-500">{hint}</p>
  </div>
);

const LogRow: React.FC<{ entry: LogEntry }> = ({ entry }) => {
  const accent =
    entry.type === 'error'
      ? 'border-red-900/60 bg-red-950/40 text-red-200'
      : entry.type === 'action'
        ? 'border-amber-900/60 bg-amber-950/40 text-amber-200'
        : entry.type === 'thought'
          ? 'border-cyan-900/60 bg-cyan-950/40 text-cyan-200'
          : 'border-neutral-800 bg-neutral-950/70 text-neutral-300';

  return (
    <div className={`rounded-2xl border px-4 py-3 ${accent}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-[0.2em] text-neutral-500">{entry.type}</span>
        <span className="text-xs text-neutral-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
      </div>
      <p className="mt-2 text-sm leading-relaxed">{entry.message}</p>
    </div>
  );
};

const EmptyLog: React.FC<{ message: string }> = ({ message }) => (
  <div className="rounded-2xl border border-dashed border-neutral-800 bg-neutral-950/60 px-4 py-6 text-center text-sm text-neutral-500">
    {message}
  </div>
);

const InsightButton: React.FC<{ active: boolean; label: string; onClick: () => void }> = ({ active, label, onClick }) => (
  <button
    onClick={onClick}
    className={`rounded-full border px-3 py-1.5 text-sm transition ${
      active
        ? 'border-cyan-700 bg-cyan-950/50 text-cyan-300'
        : 'border-neutral-700 bg-neutral-950 text-neutral-400 hover:border-neutral-500 hover:text-white'
    }`}
  >
    {label}
  </button>
);

const MemorySummaryCard: React.FC<{ isRomLoaded: boolean; memoryState: MemoryWatch }> = ({ isRomLoaded, memoryState }) => {
  if (!isRomLoaded) {
    return (
      <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5 text-center text-sm text-neutral-500">
        Load a ROM to inspect watched memory.
      </div>
    );
  }

  return (
    <div className="rounded-3xl border border-neutral-800 bg-neutral-900/80 p-5">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Watched Memory</h3>
          <p className="mt-1 text-sm text-neutral-400">Backend watch list</p>
        </div>
        <ChevronRight className="h-4 w-4 text-neutral-500" />
      </div>

      <div className="mt-4 space-y-2">
        {memoryState.values.length === 0 ? (
          <EmptyLog message="No watched addresses were returned." />
        ) : (
          memoryState.values.map((value) => (
            <div
              key={value.address}
              className="flex items-center justify-between rounded-2xl border border-neutral-800 bg-neutral-950/80 px-4 py-3"
            >
              <div>
                <p className="text-sm text-white">{value.name}</p>
                <p className="mt-1 text-xs text-neutral-500">0x{value.address.toString(16).toUpperCase()}</p>
              </div>
              <div className="text-right">
                <p className="font-mono text-sm text-cyan-300">{value.hex}</p>
                <p className="mt-1 text-xs text-neutral-500">{value.value ?? '--'}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

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
    <div className="flex flex-col items-center gap-6 md:flex-row md:justify-center">
      <div
        className="grid h-36 w-36 grid-cols-3 grid-rows-3 gap-2"
        style={{ gridTemplateAreas: '". up ." "left . right" ". down ."' }}
      >
        {directions.map((direction) => (
          <button
            key={direction.action}
            style={{ gridArea: direction.gridArea }}
            onClick={() => onPress(direction.action)}
            disabled={disabled}
            className={`rounded-2xl border text-xl transition ${
              disabled
                ? 'border-neutral-800 bg-neutral-950 text-neutral-700'
                : 'border-neutral-700 bg-neutral-950 text-neutral-300 hover:border-cyan-600 hover:text-white'
            } ${lastButton === direction.action ? 'border-cyan-500 bg-cyan-950/60 text-cyan-300' : ''}`}
          >
            {direction.label}
          </button>
        ))}
      </div>

      <div className="flex flex-col items-center gap-4">
        <div className="flex gap-4">
          {buttons.map((button) => (
            <button
              key={button.action}
              onClick={() => onPress(button.action)}
              disabled={disabled}
              className={`flex h-16 w-16 items-center justify-center rounded-full border text-lg font-semibold transition ${
                disabled
                  ? 'border-neutral-800 bg-neutral-950 text-neutral-700'
                  : 'border-neutral-700 bg-neutral-950 text-neutral-300 hover:border-cyan-600 hover:text-white'
              } ${lastButton === button.action ? 'border-cyan-500 bg-cyan-950/60 text-cyan-300' : ''}`}
            >
              {button.label}
            </button>
          ))}
        </div>

        <div className="flex gap-3">
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
    onClick={() => onPress(action)}
    disabled={disabled}
    className={`rounded-full border px-4 py-2 text-sm transition ${
      disabled
        ? 'border-neutral-800 bg-neutral-950 text-neutral-700'
        : 'border-neutral-700 bg-neutral-950 text-neutral-300 hover:border-cyan-600 hover:text-white'
    } ${lastButton === action ? 'border-cyan-500 bg-cyan-950/60 text-cyan-300' : ''}`}
  >
    {label}
  </button>
);

const KeyboardHelp: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/75 px-4">
    <div className="w-full max-w-md rounded-3xl border border-neutral-800 bg-neutral-950 p-6 shadow-2xl shadow-black/60">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Keyboard shortcuts</h2>
        <button
          onClick={onClose}
          className="rounded-full border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 transition hover:border-neutral-500 hover:text-white"
        >
          Close
        </button>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
        <ShortcutRow keys="↑ ↓ ← →" action="D-pad" />
        <ShortcutRow keys="Z / X" action="A / B" />
        <ShortcutRow keys="Enter" action="Start" />
        <ShortcutRow keys="Shift" action="Select" />
        <ShortcutRow keys="?" action="Toggle this help" />
      </div>
    </div>
  </div>
);

const ShortcutRow: React.FC<{ keys: string; action: string }> = ({ keys, action }) => (
  <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70 px-4 py-3">
    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">{keys}</p>
    <p className="mt-2 text-sm text-neutral-300">{action}</p>
  </div>
);

export default App;
