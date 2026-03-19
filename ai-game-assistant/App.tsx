import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Bot,
  BrainCircuit,
  ChevronRight,
  Cpu,
  Database,
  Gamepad2,
  LoaderCircle,
  MonitorCog,
  Package2,
  RefreshCw,
  Save,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  Upload,
  WandSparkles,
} from 'lucide-react';
import apiService, {
  type AgentAutonomy,
  type AgentModeResponse,
  type AgentStatus,
  type AiActionResponse,
  type ChatResponse,
  type ConfigValidationResponse,
  type EmulatorModeResponse,
  type GameAction,
  type GameButton,
  type GameState,
  type HealthResponse,
  type InventoryResponse,
  type MemoryReadResponse,
  type MemoryWatch,
  type MemoryWriteResponse,
  type PartyResponse,
  type PerformanceResponse,
  type ProviderStatus,
  type ScreenFormat,
  type ScreenResponse,
  type UiStatusResponse,
} from './services/apiService';

type ConnectionState = 'checking' | 'online' | 'offline';
type ActivityLevel = 'info' | 'success' | 'warning' | 'error';
type DataTab = 'party' | 'inventory' | 'memory';

interface DashboardSettings {
  backendUrl: string;
  provider: string;
  model: string;
  apiEndpoint: string;
  apiKey: string;
  goal: string;
  emulatorType: 'gb' | 'gba';
  launchUiOnLoad: boolean;
  liveScreen: boolean;
  screenRefreshMs: number;
}

interface ActivityEntry {
  id: number;
  level: ActivityLevel;
  source: string;
  message: string;
  timestamp: string;
}

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  provider?: string | null;
  timestamp: string;
}

interface AgentDraft {
  mode: string;
  enabled: boolean;
  autonomousLevel: AgentAutonomy;
  direction: string;
  target: string;
}

interface MemoryForm {
  address: string;
  size: string;
  format: ScreenFormat;
  writeValues: string;
}

const SETTINGS_STORAGE_KEY = 'pyboy_webui_settings_v2';
const MAX_ACTIVITY_ENTRIES = 80;
const MANUAL_ACTIONS: GameAction[] = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'];
const BUTTON_LAYOUT: Array<{ label: string; button: GameButton; kind?: 'accent' | 'ghost' }> = [
  { label: 'Up', button: 'UP' },
  { label: 'Left', button: 'LEFT' },
  { label: 'Down', button: 'DOWN' },
  { label: 'Right', button: 'RIGHT' },
  { label: 'A', button: 'A', kind: 'accent' },
  { label: 'B', button: 'B', kind: 'accent' },
  { label: 'Start', button: 'START' },
  { label: 'Select', button: 'SELECT' },
];

const PROVIDER_META: Record<
  string,
  {
    label: string;
    helper: string;
    endpointPlaceholder?: string;
    keyLabel?: string;
  }
> = {
  mock: {
    label: 'Mock',
    helper: 'No key required. Useful for backend validation and fallback behavior.',
  },
  gemini: {
    label: 'Google Gemini',
    helper: 'Use server-side env config or provide a Gemini key here.',
    keyLabel: 'Gemini API key',
  },
  openrouter: {
    label: 'OpenRouter',
    helper: 'Pick a specific vendor/model combination if you want deterministic routing.',
    keyLabel: 'OpenRouter API key',
  },
  'openai-compatible': {
    label: 'OpenAI Compatible',
    helper: 'Best for LM Studio, Ollama proxies, or any OpenAI-compatible local endpoint.',
    endpointPlaceholder: 'http://localhost:1234/v1',
    keyLabel: 'API key (optional for local endpoints)',
  },
  nvidia: {
    label: 'NVIDIA NIM',
    helper: 'Uses the NVIDIA API catalog endpoint and model list exposed by the backend.',
    keyLabel: 'NVIDIA API key',
  },
};

const DEFAULT_SETTINGS: DashboardSettings = {
  backendUrl: 'http://localhost:5000',
  provider: 'mock',
  model: '',
  apiEndpoint: 'http://localhost:1234/v1',
  apiKey: '',
  goal: 'Make safe progress and explain important decisions.',
  emulatorType: 'gb',
  launchUiOnLoad: true,
  liveScreen: true,
  screenRefreshMs: 500,
};

function loadSettings(): DashboardSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) {
      return DEFAULT_SETTINGS;
    }
    return {
      ...DEFAULT_SETTINGS,
      ...JSON.parse(raw),
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(settings: DashboardSettings) {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
}

function formatTime(timestamp?: string | number) {
  if (!timestamp) {
    return 'Unknown';
  }

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return 'Unknown';
  }

  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatNumber(value?: number | null) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 'n/a';
  }
  return new Intl.NumberFormat().format(value);
}

function formatCompactNumber(value?: number | null) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 'n/a';
  }
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

function toHex(value: number, padding = 4) {
  return `0x${value.toString(16).toUpperCase().padStart(padding, '0')}`;
}

function parseNumericInput(raw: string) {
  const value = raw.trim();
  if (!value) {
    return null;
  }
  if (/^0x/i.test(value)) {
    return Number.parseInt(value, 16);
  }
  return Number.parseInt(value, 10);
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return 'Unknown error';
}

function toScreenSource(screen: ScreenResponse | null) {
  if (!screen?.image) {
    return null;
  }
  return `data:image/jpeg;base64,${screen.image}`;
}

const App: React.FC = () => {
  const [settings, setSettings] = useState<DashboardSettings>(loadSettings);
  const [settingsDraft, setSettingsDraft] = useState<DashboardSettings>(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [connectionState, setConnectionState] = useState<ConnectionState>('checking');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [configInfo, setConfigInfo] = useState<Record<string, unknown> | null>(null);
  const [configValidation, setConfigValidation] = useState<ConfigValidationResponse | null>(null);
  const [providers, setProviders] = useState<Record<string, ProviderStatus>>({});
  const [models, setModels] = useState<string[]>([]);
  const [emulatorMode, setEmulatorMode] = useState<EmulatorModeResponse | null>(null);
  const [performance, setPerformance] = useState<PerformanceResponse | null>(null);
  const [uiStatus, setUiStatus] = useState<UiStatusResponse | null>(null);

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [agentMode, setAgentMode] = useState<AgentModeResponse | null>(null);
  const [agentDraft, setAgentDraft] = useState<AgentDraft>({
    mode: 'manual',
    enabled: true,
    autonomousLevel: 'moderate',
    direction: 'UP',
    target: '',
  });
  const [agentDraftDirty, setAgentDraftDirty] = useState(false);

  const [screen, setScreen] = useState<ScreenResponse | null>(null);
  const [party, setParty] = useState<PartyResponse | null>(null);
  const [inventory, setInventory] = useState<InventoryResponse | null>(null);
  const [memoryWatch, setMemoryWatch] = useState<MemoryWatch | null>(null);
  const [memoryRead, setMemoryRead] = useState<MemoryReadResponse | null>(null);
  const [memoryWrite, setMemoryWrite] = useState<MemoryWriteResponse | null>(null);

  const [activeDataTab, setActiveDataTab] = useState<DataTab>('party');
  const [romPath, setRomPath] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [aiSuggestion, setAiSuggestion] = useState<AiActionResponse | null>(null);
  const [manualAction, setManualAction] = useState<GameAction>('A');
  const [manualFrames, setManualFrames] = useState(1);
  const [memoryForm, setMemoryForm] = useState<MemoryForm>({
    address: '0xD163',
    size: '1',
    format: 'hex',
    writeValues: '',
  });
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const connectionRef = useRef<ConnectionState>('checking');
  const screenSource = useMemo(() => toScreenSource(screen), [screen]);

  const selectedProviderStatus = providers[settings.provider];
  const selectedProviderMeta = PROVIDER_META[settings.provider] || {
    label: settings.provider,
    helper: 'This provider is exposed by the backend but does not have frontend-specific guidance.',
  };

  const addActivity = useCallback((level: ActivityLevel, source: string, message: string) => {
    setActivity((current) => [
      {
        id: Date.now() + Math.random(),
        level,
        source,
        message,
        timestamp: new Date().toISOString(),
      },
      ...current,
    ].slice(0, MAX_ACTIVITY_ENTRIES));
  }, []);

  const setBusyState = useCallback((key: string, value: boolean) => {
    setBusy((current) => ({ ...current, [key]: value }));
  }, []);

  const updateConnection = useCallback((next: ConnectionState) => {
    if (connectionRef.current === next) {
      setConnectionState(next);
      return;
    }

    connectionRef.current = next;
    setConnectionState(next);

    if (next === 'online') {
      addActivity('success', 'backend', `Connected to ${apiService.getBaseUrl()}`);
    } else if (next === 'offline') {
      addActivity('warning', 'backend', `Lost connection to ${apiService.getBaseUrl()}`);
    }
  }, [addActivity]);

  const runTask = useCallback(async <T,>(
    key: string,
    task: () => Promise<T>,
    options?: { successMessage?: string; source?: string; silent?: boolean }
  ) => {
    setBusyState(key, true);
    try {
      const result = await task();
      if (options?.successMessage) {
        addActivity('success', options.source ?? key, options.successMessage);
      }
      return result;
    } catch (error) {
      if (!options?.silent) {
        addActivity('error', options?.source ?? key, getErrorMessage(error));
      }
      throw error;
    } finally {
      setBusyState(key, false);
    }
  }, [addActivity, setBusyState]);

  const loadModels = useCallback(async (provider: string, silent = false) => {
    if (!provider) {
      setModels([]);
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);

    try {
      const result = await apiService.getModels(provider);
      setModels(result.models);
    } catch (error) {
      setModels([]);
      if (!silent) {
        addActivity('warning', 'models', `Could not load models for ${provider}: ${getErrorMessage(error)}`);
      }
    }
  }, [addActivity, settings.backendUrl]);

  const refreshStaticData = useCallback(async () => {
    apiService.setBaseUrl(settings.backendUrl);

    const [
      healthResult,
      configResult,
      validationResult,
      providersResult,
      emulatorModeResult,
    ] = await Promise.allSettled([
      apiService.getHealth(),
      apiService.getConfig(),
      apiService.validateConfig(),
      apiService.getProvidersStatus(),
      apiService.getEmulatorMode(),
    ]);

    if (healthResult.status === 'fulfilled') {
      setHealth(healthResult.value);
      updateConnection('online');
    } else {
      setHealth(null);
      updateConnection('offline');
    }

    setConfigInfo(configResult.status === 'fulfilled' ? configResult.value : null);
    setConfigValidation(validationResult.status === 'fulfilled' ? validationResult.value : {
      error: getErrorMessage(validationResult.reason),
    });
    setProviders(providersResult.status === 'fulfilled' ? providersResult.value : {});
    setEmulatorMode(emulatorModeResult.status === 'fulfilled' ? emulatorModeResult.value : null);
  }, [settings.backendUrl, updateConnection]);

  const refreshLiveData = useCallback(async () => {
    apiService.setBaseUrl(settings.backendUrl);

    const [
      gameResult,
      agentResult,
      agentModeResult,
      performanceResult,
    ] = await Promise.allSettled([
      apiService.getGameState(),
      apiService.getAgentStatus(),
      apiService.getAgentMode(),
      apiService.getPerformance(),
    ]);

    if (gameResult.status === 'fulfilled') {
      setGameState(gameResult.value);
      updateConnection('online');
    } else {
      setGameState(null);
      updateConnection('offline');
    }

    setAgentStatus(agentResult.status === 'fulfilled' ? agentResult.value : null);
    setPerformance(performanceResult.status === 'fulfilled' ? performanceResult.value : null);

    if (agentModeResult.status === 'fulfilled') {
      setAgentMode(agentModeResult.value);
      if (!agentDraftDirty) {
        setAgentDraft({
          mode: agentModeResult.value.mode,
          enabled: agentModeResult.value.enabled,
          autonomousLevel: agentModeResult.value.autonomous_level,
          direction: 'UP',
          target: '',
        });
      }
    }
  }, [agentDraftDirty, settings.backendUrl, updateConnection]);

  const refreshRomData = useCallback(async () => {
    apiService.setBaseUrl(settings.backendUrl);

    const game = await apiService.getGameState().catch(() => null);
    if (!game?.rom_loaded) {
      setScreen(null);
      setParty(null);
      setInventory(null);
      setMemoryWatch(null);
      setUiStatus(null);
      return;
    }

    const [partyResult, inventoryResult, memoryResult, uiStatusResult] = await Promise.allSettled([
      apiService.getParty(),
      apiService.getInventory(),
      apiService.getMemoryWatch(),
      apiService.getUiStatus(),
    ]);

    setParty(partyResult.status === 'fulfilled' ? partyResult.value : null);
    setInventory(inventoryResult.status === 'fulfilled' ? inventoryResult.value : null);
    setMemoryWatch(memoryResult.status === 'fulfilled' ? memoryResult.value : null);
    setUiStatus(uiStatusResult.status === 'fulfilled' ? uiStatusResult.value : null);
  }, [settings.backendUrl]);

  const refreshScreen = useCallback(async () => {
    if (!settings.liveScreen) {
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);

    try {
      const game = await apiService.getGameState();
      if (!game.rom_loaded || !game.screen_available) {
        setScreen(null);
        return;
      }
      const nextScreen = await apiService.getScreen();
      setScreen(nextScreen);
    } catch {
      setScreen(null);
    }
  }, [settings.backendUrl, settings.liveScreen]);

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshStaticData(), refreshLiveData()]);
    await Promise.all([refreshRomData(), refreshScreen()]);
  }, [refreshLiveData, refreshRomData, refreshScreen, refreshStaticData]);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  useEffect(() => {
    apiService.setBaseUrl(settings.backendUrl);
    updateConnection('checking');
    void refreshAll();
    void loadModels(settings.provider, true);
  }, [loadModels, refreshAll, settings.backendUrl, settings.provider, updateConnection]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      void refreshLiveData();
    }, 2500);

    return () => window.clearInterval(poll);
  }, [refreshLiveData]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      void refreshRomData();
    }, 6000);

    return () => window.clearInterval(poll);
  }, [refreshRomData]);

  useEffect(() => {
    if (!settings.liveScreen) {
      return undefined;
    }

    const poll = window.setInterval(() => {
      void refreshScreen();
    }, settings.screenRefreshMs);

    return () => window.clearInterval(poll);
  }, [refreshScreen, settings.liveScreen, settings.screenRefreshMs]);

  useEffect(() => {
    if (!settingsOpen) {
      return;
    }
    setSettingsDraft(settings);
  }, [settings, settingsOpen]);

  useEffect(() => {
    if (!settingsOpen) {
      return;
    }
    void loadModels(settingsDraft.provider, true);
  }, [loadModels, settingsDraft.provider, settingsOpen]);

  const handleOpenSettings = () => {
    setSettingsDraft(settings);
    setSettingsOpen(true);
  };

  const handleSaveSettings = async () => {
    const normalized = {
      ...settingsDraft,
      backendUrl: settingsDraft.backendUrl.trim() || DEFAULT_SETTINGS.backendUrl,
      apiEndpoint: settingsDraft.apiEndpoint.trim(),
      model: settingsDraft.model.trim(),
      apiKey: settingsDraft.apiKey.trim(),
      goal: settingsDraft.goal.trim() || DEFAULT_SETTINGS.goal,
      screenRefreshMs: Math.max(250, Math.min(2000, settingsDraft.screenRefreshMs)),
    };

    setSettings(normalized);
    setSettingsOpen(false);
    addActivity('success', 'settings', 'Settings updated');
    await refreshAll();
  };

  const handleUploadRom = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);

    try {
      await runTask('upload-rom', async () => {
        const result = await apiService.uploadRom(file, settings.emulatorType, settings.launchUiOnLoad);
        return result;
      }, {
        successMessage: `Loaded ROM ${file.name}`,
        source: 'rom',
      });
      await refreshAll();
    } finally {
      if (event.target) {
        event.target.value = '';
      }
    }
  };

  const handleLoadRomFromPath = async () => {
    const trimmedPath = romPath.trim();
    if (!trimmedPath) {
      addActivity('warning', 'rom', 'Enter a ROM path before loading from disk');
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);
    await runTask('load-rom-path', async () => {
      const result = await apiService.loadRomFromPath(trimmedPath, settings.emulatorType, settings.launchUiOnLoad);
      setRomPath(result.rom_path || trimmedPath);
      return result;
    }, {
      successMessage: `Loaded ROM from ${trimmedPath}`,
      source: 'rom',
    });
    await refreshAll();
  };

  const handleSaveState = async () => {
    apiService.setBaseUrl(settings.backendUrl);
    await runTask('save-state', () => apiService.saveState(), {
      successMessage: 'State saved',
      source: 'state',
    });
  };

  const handleLoadState = async () => {
    apiService.setBaseUrl(settings.backendUrl);
    await runTask('load-state', () => apiService.loadState(), {
      successMessage: 'State restored',
      source: 'state',
    });
    await Promise.all([refreshScreen(), refreshRomData(), refreshLiveData()]);
  };

  const handlePressButton = async (button: GameButton) => {
    apiService.setBaseUrl(settings.backendUrl);
    await runTask(`button-${button}`, async () => apiService.pressButton(button), {
      successMessage: `Pressed ${button}`,
      source: 'controls',
      silent: true,
    }).catch(() => undefined);
    void refreshScreen();
    void refreshLiveData();
  };

  const handleManualAction = async () => {
    apiService.setBaseUrl(settings.backendUrl);
    const frames = Math.max(1, Math.min(100, manualFrames));
    await runTask('manual-action', async () => apiService.executeAction(manualAction, frames), {
      successMessage: `Executed ${manualAction} for ${frames} frame${frames === 1 ? '' : 's'}`,
      source: 'controls',
    });
    await Promise.all([refreshScreen(), refreshLiveData()]);
  };

  const requestAiAction = async (applyAfter = false) => {
    if (!settings.goal.trim()) {
      addActivity('warning', 'assistant', 'Set an AI goal before requesting an action');
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);
    const result = await runTask('ai-action', async () => apiService.requestAiAction({
      api_name: settings.provider,
      api_key: settings.apiKey || undefined,
      api_endpoint: settings.apiEndpoint || undefined,
      model: settings.model || undefined,
      goal: settings.goal,
    }), {
      successMessage: 'AI suggested a next move',
      source: 'assistant',
    });

    setAiSuggestion(result);

    if (applyAfter) {
      await runTask('ai-apply', async () => apiService.executeAction(result.action, 1), {
        successMessage: `Applied AI action ${result.action}`,
        source: 'assistant',
      });
      await Promise.all([refreshScreen(), refreshLiveData(), refreshRomData()]);
    }
  };

  const handleApplySuggestion = async () => {
    if (!aiSuggestion) {
      addActivity('warning', 'assistant', 'Request a suggestion before applying it');
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);
    await runTask('apply-suggestion', async () => apiService.executeAction(aiSuggestion.action, 1), {
      successMessage: `Applied suggested action ${aiSuggestion.action}`,
      source: 'assistant',
    });
    await Promise.all([refreshScreen(), refreshLiveData(), refreshRomData()]);
  };

  const handleSendChat = async () => {
    const message = chatInput.trim();
    if (!message) {
      return;
    }

    setChatMessages((current) => [
      ...current,
      {
        id: Date.now(),
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      },
    ]);
    setChatInput('');

    apiService.setBaseUrl(settings.backendUrl);

    try {
      const response = await runTask<ChatResponse>('chat', async () => apiService.chat({
        message,
        api_name: settings.provider,
        api_key: settings.apiKey || undefined,
        api_endpoint: settings.apiEndpoint || undefined,
        model: settings.model || undefined,
      }), {
        successMessage: 'Assistant replied',
        source: 'chat',
      });

      setChatMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.response,
          provider: response.provider_used,
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      setChatMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: `Request failed: ${getErrorMessage(error)}`,
          timestamp: new Date().toISOString(),
        },
      ]);
    }
  };

  const handleApplyAgentMode = async () => {
    apiService.setBaseUrl(settings.backendUrl);
    await runTask('agent-mode', async () => apiService.setAgentMode({
      mode: agentDraft.mode,
      enabled: agentDraft.enabled,
      autonomous_level: agentDraft.autonomousLevel,
      direction: agentDraft.direction,
      target: agentDraft.target.trim() || undefined,
    }), {
      successMessage: `Agent mode set to ${agentDraft.mode}`,
      source: 'agent',
    });
    setAgentDraftDirty(false);
    await refreshLiveData();
  };

  const handleUiControl = async (action: 'launch' | 'stop' | 'restart') => {
    apiService.setBaseUrl(settings.backendUrl);
    const request =
      action === 'launch'
        ? apiService.launchUi()
        : action === 'restart'
          ? apiService.restartUi()
          : apiService.stopUi();

    await runTask(`ui-${action}`, async () => request, {
      successMessage: `UI ${action} request completed`,
      source: 'ui',
    });
    await refreshRomData();
  };

  const handleClearCache = async () => {
    apiService.setBaseUrl(settings.backendUrl);
    await runTask('clear-cache', async () => apiService.clearCache(), {
      successMessage: 'Emulator caches cleared',
      source: 'performance',
    });
    await refreshLiveData();
  };

  const handleReadMemory = async () => {
    const address = parseNumericInput(memoryForm.address);
    const size = parseNumericInput(memoryForm.size);

    if (address === null || Number.isNaN(address)) {
      addActivity('warning', 'memory', 'Enter a valid memory address');
      return;
    }

    if (size === null || Number.isNaN(size) || size < 1) {
      addActivity('warning', 'memory', 'Enter a valid read size');
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);
    const result = await runTask('memory-read', async () => apiService.readMemory(address, size, memoryForm.format), {
      successMessage: `Read ${size} byte${size === 1 ? '' : 's'} from ${toHex(address)}`,
      source: 'memory',
      silent: true,
    }).catch((error) => {
      setMemoryRead(null);
      throw error;
    });

    if (result) {
      setMemoryRead(result);
    }
  };

  const handleWriteMemory = async () => {
    const address = parseNumericInput(memoryForm.address);
    if (address === null || Number.isNaN(address)) {
      addActivity('warning', 'memory', 'Enter a valid memory address');
      return;
    }

    const values = memoryForm.writeValues
      .split(',')
      .map((part) => parseNumericInput(part))
      .filter((value): value is number => value !== null && !Number.isNaN(value));

    if (!values.length) {
      addActivity('warning', 'memory', 'Enter one or more numeric values to write');
      return;
    }

    apiService.setBaseUrl(settings.backendUrl);
    const result = await runTask('memory-write', async () => apiService.writeMemory(address, values), {
      successMessage: `Wrote ${values.length} byte${values.length === 1 ? '' : 's'} to ${toHex(address)}`,
      source: 'memory',
    });
    setMemoryWrite(result);
    await refreshRomData();
  };

  const configWarnings = configValidation?.validation?.warnings || [];
  const memoryRows = (memoryWatch?.values || []).filter(
    (entry): entry is Exclude<MemoryWatch['values'][number], { error: string }> =>
      !('error' in entry)
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__title">
          <div className="topbar__eyebrow">PyBoy Web Control</div>
          <h1>AI Game Assistant</h1>
          <p>One coherent control surface for the emulator, agent, providers, and Pokémon-specific state.</p>
        </div>

        <div className="topbar__actions">
          <div className={`status-pill status-pill--${connectionState}`}>
            <span className="status-pill__dot" />
            {connectionState === 'online' ? 'Backend online' : connectionState === 'offline' ? 'Backend offline' : 'Checking backend'}
          </div>
          <div className="status-pill">
            <Sparkles size={14} />
            {selectedProviderMeta.label}
            {settings.model ? <span className="status-pill__subtle">{settings.model}</span> : null}
          </div>
          <button className="button button--ghost" onClick={() => void refreshAll()} disabled={busy['refresh-all']}>
            <RefreshCw size={16} />
            Refresh
          </button>
          <button className="button button--primary" onClick={handleOpenSettings}>
            <Settings size={16} />
            Settings
          </button>
        </div>
      </header>

      <div className="summary-strip">
        <SummaryCard icon={<Gamepad2 size={16} />} label="ROM" value={gameState?.rom_name || 'No ROM loaded'} secondary={gameState?.emulator?.toUpperCase() || settings.emulatorType.toUpperCase()} />
        <SummaryCard icon={<Activity size={16} />} label="Frame" value={gameState ? formatNumber(gameState.frame_count) : 'n/a'} secondary={gameState?.fps ? `${gameState.fps} FPS` : 'No frame data'} />
        <SummaryCard icon={<Bot size={16} />} label="Agent" value={agentStatus?.mode || 'manual'} secondary={agentStatus?.current_action || 'Idle'} />
        <SummaryCard icon={<ShieldCheck size={16} />} label="Config" value={configValidation?.validation?.valid ? 'Valid' : configValidation?.error ? 'Unavailable' : 'Needs review'} secondary={configWarnings.length ? `${configWarnings.length} warning${configWarnings.length === 1 ? '' : 's'}` : 'No warnings'} />
      </div>

      <main className="dashboard-grid">
        <section className="dashboard-column dashboard-column--main">
          <Panel
            title="Emulator"
            subtitle="Live screen, ROM loading, save states, and a clean summary of the active game."
            action={(
              <button className="button button--ghost" onClick={() => fileInputRef.current?.click()}>
                <Upload size={16} />
                Upload ROM
              </button>
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".gb,.gbc,.gba,.zip"
              className="visually-hidden"
              onChange={handleUploadRom}
            />

            <div className="rom-toolbar">
              <div className="field field--grow">
                <label>Load from server path</label>
                <div className="field__row">
                  <input
                    value={romPath}
                    onChange={(event) => setRomPath(event.target.value)}
                    placeholder="/absolute/path/to/rom.gb"
                  />
                  <button className="button button--secondary" onClick={() => void handleLoadRomFromPath()} disabled={busy['load-rom-path']}>
                    {busy['load-rom-path'] ? <LoaderCircle size={16} className="spin" /> : <ChevronRight size={16} />}
                    Load
                  </button>
                </div>
              </div>

              <div className="toolbar-group">
                <button className="button button--ghost" onClick={() => void handleSaveState()} disabled={busy['save-state']}>
                  <Save size={16} />
                  Save
                </button>
                <button className="button button--ghost" onClick={() => void handleLoadState()} disabled={busy['load-state']}>
                  <RefreshCw size={16} />
                  Load
                </button>
                <button className="button button--ghost" onClick={() => void refreshScreen()}>
                  <MonitorCog size={16} />
                  Refresh screen
                </button>
              </div>
            </div>

            <div className="screen-card">
              {screenSource ? (
                <img className="screen-card__image" src={screenSource} alt="Current emulator frame" />
              ) : (
                <div className="screen-card__empty">
                  <Gamepad2 size={48} />
                  <h3>No active screen</h3>
                  <p>Load a ROM and keep live refresh enabled to stream frames into the WebUI.</p>
                </div>
              )}
              <div className="screen-card__meta">
                <span>{settings.liveScreen ? `Live refresh every ${settings.screenRefreshMs}ms` : 'Live refresh is disabled'}</span>
                <span>{screen?.performance?.current_fps ? `${screen.performance.current_fps} FPS` : 'No screen perf data'}</span>
                <span>{screen?.optimization?.cache_hit ? 'Cache hit' : 'Fresh frame'}</span>
              </div>
            </div>

            <div className="stat-grid">
              <KeyStat label="Backend" value={settings.backendUrl} />
              <KeyStat label="Emulator mode" value={emulatorMode?.current_mode || 'n/a'} />
              <KeyStat label="Screen shape" value={screen?.shape?.join(' × ') || 'n/a'} />
              <KeyStat label="Last frame" value={screen?.pyboy_frame ? formatNumber(screen.pyboy_frame) : 'n/a'} />
            </div>
          </Panel>

          <Panel
            title="Manual Controls"
            subtitle="Direct button presses plus raw action execution for frame-sensitive testing."
          >
            <div className="controls-layout">
              <div className="button-grid">
                {BUTTON_LAYOUT.map((item) => (
                  <button
                    key={item.button}
                    className={`control-button ${item.kind === 'accent' ? 'control-button--accent' : ''}`}
                    onClick={() => void handlePressButton(item.button)}
                    disabled={!gameState?.rom_loaded}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              <div className="manual-action">
                <div className="field">
                  <label>Raw action</label>
                  <select value={manualAction} onChange={(event) => setManualAction(event.target.value as GameAction)}>
                    {MANUAL_ACTIONS.map((action) => (
                      <option key={action} value={action}>
                        {action}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field">
                  <label>Frames</label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={manualFrames}
                    onChange={(event) => setManualFrames(Math.max(1, Math.min(100, Number(event.target.value) || 1)))}
                  />
                </div>

                <button className="button button--primary" onClick={() => void handleManualAction()} disabled={!gameState?.rom_loaded || busy['manual-action']}>
                  <Gamepad2 size={16} />
                  Execute
                </button>
              </div>
            </div>
          </Panel>
        </section>

        <section className="dashboard-column">
          <Panel
            title="AI Assistant"
            subtitle="One place for provider selection results, single-step planning, and in-context chat."
            action={(
              <div className="panel-action-meta">
                {selectedProviderStatus ? (
                  <span className={`availability availability--${selectedProviderStatus.available ? 'up' : 'down'}`}>
                    {selectedProviderStatus.available ? 'Provider available' : 'Provider unavailable'}
                  </span>
                ) : null}
              </div>
            )}
          >
            <div className="field">
              <label>Current objective</label>
              <textarea
                rows={3}
                value={settings.goal}
                onChange={(event) => setSettings((current) => ({ ...current, goal: event.target.value }))}
                placeholder="Describe what the assistant should optimize for."
              />
            </div>

            <div className="assistant-actions">
              <button className="button button--secondary" onClick={() => void requestAiAction(false)} disabled={!gameState?.rom_loaded || busy['ai-action']}>
                <BrainCircuit size={16} />
                Suggest move
              </button>
              <button className="button button--primary" onClick={() => void requestAiAction(true)} disabled={!gameState?.rom_loaded || busy['ai-action'] || busy['ai-apply']}>
                <WandSparkles size={16} />
                Suggest and apply
              </button>
              <button className="button button--ghost" onClick={() => void handleApplySuggestion()} disabled={!aiSuggestion || busy['apply-suggestion']}>
                <ChevronRight size={16} />
                Apply last suggestion
              </button>
            </div>

            <div className="suggestion-card">
              <div>
                <div className="suggestion-card__label">Latest suggestion</div>
                <div className="suggestion-card__action">{aiSuggestion?.action || 'No suggestion yet'}</div>
              </div>
              <div className="suggestion-card__meta">
                <span>Provider: {aiSuggestion?.provider_used || settings.provider}</span>
                <span>Cache: {aiSuggestion?.optimization?.cache_hit ? 'hit' : 'miss'}</span>
                <span>{aiSuggestion?.optimization?.response_time_ms ? `${aiSuggestion.optimization.response_time_ms} ms` : 'No timing'}</span>
              </div>
            </div>

            <div className="chat-panel">
              <div className="chat-panel__messages">
                {chatMessages.length === 0 ? (
                  <div className="empty-state compact">
                    Ask about the current game state, route choice, or why the agent suggested a move.
                  </div>
                ) : (
                  chatMessages.map((message) => (
                    <div key={message.id} className={`chat-bubble chat-bubble--${message.role}`}>
                      <div className="chat-bubble__meta">
                        <span>{message.role === 'user' ? 'You' : message.provider || 'Assistant'}</span>
                        <span>{formatTime(message.timestamp)}</span>
                      </div>
                      <p>{message.content}</p>
                    </div>
                  ))
                )}
              </div>

              <div className="chat-panel__composer">
                <textarea
                  rows={3}
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about the screen, next objective, or a specific memory/state detail."
                />
                <button className="button button--primary" onClick={() => void handleSendChat()} disabled={!gameState?.rom_loaded || busy['chat']}>
                  <Send size={16} />
                  Send
                </button>
              </div>
            </div>
          </Panel>

          <Panel
            title="Agent Mode"
            subtitle="Sync the agent endpoint instead of burying this behind provider settings."
          >
            <div className="agent-summary">
              <div>
                <div className="agent-summary__label">Current action</div>
                <div className="agent-summary__value">{agentStatus?.current_action || 'Idle'}</div>
              </div>
              <div>
                <div className="agent-summary__label">Last decision</div>
                <div className="agent-summary__value agent-summary__value--long">{agentStatus?.last_decision || 'No decision yet'}</div>
              </div>
            </div>

            <div className="form-grid">
              <div className="field">
                <label>Mode</label>
                <select
                  value={agentDraft.mode}
                  onChange={(event) => {
                    setAgentDraftDirty(true);
                    setAgentDraft((current) => ({ ...current, mode: event.target.value }));
                  }}
                >
                  {(agentMode?.valid_modes || ['manual']).map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>Autonomy</label>
                <select
                  value={agentDraft.autonomousLevel}
                  onChange={(event) => {
                    setAgentDraftDirty(true);
                    setAgentDraft((current) => ({ ...current, autonomousLevel: event.target.value as AgentAutonomy }));
                  }}
                >
                  <option value="passive">passive</option>
                  <option value="moderate">moderate</option>
                  <option value="aggressive">aggressive</option>
                </select>
              </div>

              <div className="field">
                <label>Direction</label>
                <select
                  value={agentDraft.direction}
                  onChange={(event) => {
                    setAgentDraftDirty(true);
                    setAgentDraft((current) => ({ ...current, direction: event.target.value }));
                  }}
                >
                  <option value="UP">UP</option>
                  <option value="DOWN">DOWN</option>
                  <option value="LEFT">LEFT</option>
                  <option value="RIGHT">RIGHT</option>
                </select>
              </div>

              <div className="field">
                <label>Target</label>
                <input
                  value={agentDraft.target}
                  onChange={(event) => {
                    setAgentDraftDirty(true);
                    setAgentDraft((current) => ({ ...current, target: event.target.value }));
                  }}
                  placeholder="Optional for auto_center / auto_shop"
                />
              </div>
            </div>

            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={agentDraft.enabled}
                onChange={(event) => {
                  setAgentDraftDirty(true);
                  setAgentDraft((current) => ({ ...current, enabled: event.target.checked }));
                }}
              />
              Agent enabled
            </label>

            <button className="button button--primary" onClick={() => void handleApplyAgentMode()} disabled={busy['agent-mode']}>
              <Bot size={16} />
              Apply mode
            </button>
          </Panel>

          <Panel title="Activity" subtitle="Actionable history. Only user-triggered and state-change events are logged.">
            <div className="activity-list">
              {activity.length === 0 ? (
                <div className="empty-state compact">No activity yet.</div>
              ) : (
                activity.map((entry) => (
                  <div key={entry.id} className={`activity-item activity-item--${entry.level}`}>
                    <div className="activity-item__meta">
                      <span>{entry.source}</span>
                      <span>{formatTime(entry.timestamp)}</span>
                    </div>
                    <p>{entry.message}</p>
                  </div>
                ))
              )}
            </div>
          </Panel>
        </section>

        <section className="dashboard-column">
          <Panel
            title="System"
            subtitle="Backend health, provider availability, UI process control, and runtime performance."
          >
            <div className="system-section">
              <div className="system-section__header">
                <h3><ShieldCheck size={16} /> Backend health</h3>
                <button className="button button--ghost" onClick={() => void refreshStaticData()}>
                  <RefreshCw size={16} />
                  Refresh
                </button>
              </div>
              <div className="key-value-list">
                <KeyValueRow label="Status" value={String(health?.status || 'offline')} />
                <KeyValueRow label="API keys configured" value={String(configValidation?.validation?.api_keys_configured ?? 'n/a')} />
                <KeyValueRow label="Warnings" value={String(configWarnings.length)} />
                <KeyValueRow label="Backend host" value={String((configInfo?.basic_config as Record<string, unknown> | undefined)?.host || (configInfo?.host as string | undefined) || 'n/a')} />
              </div>
              {configWarnings.length ? (
                <div className="callout callout--warning">
                  {configWarnings.join(' ')}
                </div>
              ) : null}
              {configValidation?.error ? (
                <div className="callout callout--muted">{configValidation.error}</div>
              ) : null}
            </div>

            <div className="system-section">
              <div className="system-section__header">
                <h3><Sparkles size={16} /> Providers</h3>
              </div>
              <div className="provider-list">
                {Object.entries(providers)
                  .sort((left, right) => left[1].priority - right[1].priority)
                  .map(([name, status]) => (
                    <div key={name} className="provider-row">
                      <div>
                        <strong>{PROVIDER_META[name]?.label || name}</strong>
                        <span>{status.status}</span>
                      </div>
                      <div className={`availability availability--${status.available ? 'up' : 'down'}`}>
                        {status.available ? 'ready' : 'blocked'}
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            <div className="system-section">
              <div className="system-section__header">
                <h3><MonitorCog size={16} /> UI process</h3>
              </div>
              <div className="key-value-list">
                <KeyValueRow label="ROM loaded" value={uiStatus?.rom_loaded ? 'yes' : 'no'} />
                <KeyValueRow label="Emulator" value={uiStatus?.active_emulator || 'n/a'} />
                <KeyValueRow label="UI state" value={JSON.stringify(uiStatus?.ui_status || {})} />
              </div>
              <div className="button-row">
                <button className="button button--ghost" onClick={() => void handleUiControl('launch')} disabled={busy['ui-launch']}>
                  Launch
                </button>
                <button className="button button--ghost" onClick={() => void handleUiControl('restart')} disabled={busy['ui-restart']}>
                  Restart
                </button>
                <button className="button button--ghost" onClick={() => void handleUiControl('stop')} disabled={busy['ui-stop']}>
                  Stop
                </button>
              </div>
            </div>

            <div className="system-section">
              <div className="system-section__header">
                <h3><Cpu size={16} /> Performance</h3>
                <button className="button button--ghost" onClick={() => void handleClearCache()} disabled={busy['clear-cache']}>
                  Clear cache
                </button>
              </div>
              <div className="key-value-list">
                <KeyValueRow label="CPU count" value={String(performance?.system_info?.cpu_count ?? 'n/a')} />
                <KeyValueRow label="Memory" value={performance?.system_info?.memory_usage_mb ? `${performance.system_info.memory_usage_mb.toFixed(1)} MB` : 'n/a'} />
                <KeyValueRow label="Multi-process" value={performance?.system_info?.multi_process_mode ? 'enabled' : 'disabled'} />
              </div>
            </div>
          </Panel>

          <Panel
            title="Game Data"
            subtitle="Party and inventory are Pokémon-focused, while memory tools stay generic for debugging."
            action={(
              <div className="tab-row">
                <button className={`tab-button ${activeDataTab === 'party' ? 'tab-button--active' : ''}`} onClick={() => setActiveDataTab('party')}>
                  Party
                </button>
                <button className={`tab-button ${activeDataTab === 'inventory' ? 'tab-button--active' : ''}`} onClick={() => setActiveDataTab('inventory')}>
                  Inventory
                </button>
                <button className={`tab-button ${activeDataTab === 'memory' ? 'tab-button--active' : ''}`} onClick={() => setActiveDataTab('memory')}>
                  Memory
                </button>
              </div>
            )}
          >
            {activeDataTab === 'party' ? (
              <div className="data-list">
                {!party?.party?.length ? (
                  <div className="empty-state compact">No party data available.</div>
                ) : (
                  party.party.map((pokemon) => (
                    <div key={`${pokemon.slot}-${pokemon.species_name}`} className="pokemon-card">
                      <div className="pokemon-card__header">
                        <strong>#{pokemon.slot} {pokemon.species_name || 'Unknown'}</strong>
                        <span>Lv. {pokemon.level ?? 'n/a'}</span>
                      </div>
                      <div className="pokemon-card__stats">
                        <span>HP {pokemon.hp ?? 0}/{pokemon.max_hp ?? 0}</span>
                        <span>{pokemon.hp_percent}%</span>
                        <span>{pokemon.type1 || 'Unknown'}{pokemon.type2 ? ` / ${pokemon.type2}` : ''}</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-bar__fill" style={{ width: `${Math.min(100, Math.max(0, pokemon.hp_percent || 0))}%` }} />
                      </div>
                      <div className="chip-row">
                        {(pokemon.moves || []).map((move) => (
                          <span key={`${pokemon.slot}-${move.id}`} className="chip">{move.name}</span>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            ) : activeDataTab === 'inventory' ? (
              <div className="data-list">
                <div className="inventory-summary">
                  <div>
                    <div className="inventory-summary__label">Money</div>
                    <div className="inventory-summary__value">{inventory?.money_formatted || 'n/a'}</div>
                  </div>
                  <div>
                    <div className="inventory-summary__label">Item count</div>
                    <div className="inventory-summary__value">{inventory?.item_count ?? 0}</div>
                  </div>
                </div>
                {!inventory?.items?.length ? (
                  <div className="empty-state compact">No inventory data available.</div>
                ) : (
                  inventory.items.map((item) => (
                    <div key={`${item.slot}-${item.id}`} className="inventory-item">
                      <div>
                        <strong>{item.name}</strong>
                        <span>Slot {item.slot}</span>
                      </div>
                      <div>x{item.quantity}</div>
                    </div>
                  ))
                )}
              </div>
            ) : (
              <div className="memory-panel">
                <div className="form-grid">
                  <div className="field">
                    <label>Address</label>
                    <input
                      value={memoryForm.address}
                      onChange={(event) => setMemoryForm((current) => ({ ...current, address: event.target.value }))}
                      placeholder="0xD163"
                    />
                  </div>
                  <div className="field">
                    <label>Size</label>
                    <input
                      value={memoryForm.size}
                      onChange={(event) => setMemoryForm((current) => ({ ...current, size: event.target.value }))}
                      placeholder="1"
                    />
                  </div>
                  <div className="field">
                    <label>Format</label>
                    <select
                      value={memoryForm.format}
                      onChange={(event) => setMemoryForm((current) => ({ ...current, format: event.target.value as ScreenFormat }))}
                    >
                      <option value="int">int</option>
                      <option value="hex">hex</option>
                      <option value="binary">binary</option>
                    </select>
                  </div>
                </div>

                <div className="button-row">
                  <button className="button button--secondary" onClick={() => void handleReadMemory()} disabled={busy['memory-read']}>
                    <Database size={16} />
                    Read
                  </button>
                </div>

                <div className="field">
                  <label>Write values</label>
                  <input
                    value={memoryForm.writeValues}
                    onChange={(event) => setMemoryForm((current) => ({ ...current, writeValues: event.target.value }))}
                    placeholder="255 or 0xFF,0x01"
                  />
                </div>

                <button className="button button--ghost" onClick={() => void handleWriteMemory()} disabled={busy['memory-write']}>
                  Write
                </button>

                {memoryRead ? (
                  <div className="callout callout--muted">
                    Read {memoryRead.size} byte{memoryRead.size === 1 ? '' : 's'} from {memoryRead.address}: {memoryRead.formatted.join(', ')}
                  </div>
                ) : null}
                {memoryWrite ? (
                  <div className="callout callout--muted">
                    {memoryWrite.message}
                  </div>
                ) : null}

                <div className="memory-watch">
                  <div className="memory-watch__header">
                    <strong>Watched addresses</strong>
                    <span>{memoryWatch?.timestamp ? formatTime(memoryWatch.timestamp) : 'n/a'}</span>
                  </div>
                  {!memoryRows.length ? (
                    <div className="empty-state compact">No watch values available.</div>
                  ) : (
                    <div className="memory-table">
                      {memoryRows.map((row) => (
                        <div key={`${row.address}-${row.name}`} className="memory-table__row">
                          <span>{row.name}</span>
                          <span>{toHex(row.address)}</span>
                          <strong>{row.value === null ? 'n/a' : row.hex}</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </Panel>
        </section>
      </main>

      {settingsOpen ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setSettingsOpen(false)}>
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={(event) => event.stopPropagation()}>
            <div className="modal__header">
              <div>
                <div className="topbar__eyebrow">WebUI Settings</div>
                <h2 id="settings-title">Connection and provider setup</h2>
              </div>
              <button className="button button--ghost" onClick={() => setSettingsOpen(false)}>
                Close
              </button>
            </div>

            <div className="modal__body">
              <div className="form-grid">
                <div className="field field--grow">
                  <label>Backend URL</label>
                  <input
                    value={settingsDraft.backendUrl}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, backendUrl: event.target.value }))}
                    placeholder="http://localhost:5000"
                  />
                </div>
                <div className="field">
                  <label>Default emulator</label>
                  <select
                    value={settingsDraft.emulatorType}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, emulatorType: event.target.value as 'gb' | 'gba' }))}
                  >
                    <option value="gb">Game Boy</option>
                    <option value="gba">Game Boy Advance</option>
                  </select>
                </div>
              </div>

              <div className="form-grid">
                <div className="field">
                  <label>Provider</label>
                  <select
                    value={settingsDraft.provider}
                    onChange={(event) => setSettingsDraft((current) => ({
                      ...current,
                      provider: event.target.value,
                      model: '',
                    }))}
                  >
                    {Object.keys({ ...PROVIDER_META, ...providers }).map((provider) => (
                      <option key={provider} value={provider}>
                        {PROVIDER_META[provider]?.label || provider}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field field--grow">
                  <label>Model</label>
                  <div className="field__row">
                    <input
                      list="provider-models"
                      value={settingsDraft.model}
                      onChange={(event) => setSettingsDraft((current) => ({ ...current, model: event.target.value }))}
                      placeholder="Pick or type a model"
                    />
                    <button className="button button--ghost" onClick={() => void loadModels(settingsDraft.provider)} type="button">
                      Refresh
                    </button>
                  </div>
                  <datalist id="provider-models">
                    {models.map((model) => (
                      <option key={model} value={model} />
                    ))}
                  </datalist>
                </div>
              </div>

              {settingsDraft.provider === 'openai-compatible' ? (
                <div className="field">
                  <label>API endpoint</label>
                  <input
                    value={settingsDraft.apiEndpoint}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, apiEndpoint: event.target.value }))}
                    placeholder={selectedProviderMeta.endpointPlaceholder}
                  />
                </div>
              ) : null}

              <div className="field">
                <label>{selectedProviderMeta.keyLabel || 'API key'}</label>
                <input
                  type="password"
                  value={settingsDraft.apiKey}
                  onChange={(event) => setSettingsDraft((current) => ({ ...current, apiKey: event.target.value }))}
                  placeholder="Leave blank to rely on backend environment configuration"
                />
                <small>{selectedProviderMeta.helper}</small>
              </div>

              <div className="field">
                <label>Default AI objective</label>
                <textarea
                  rows={3}
                  value={settingsDraft.goal}
                  onChange={(event) => setSettingsDraft((current) => ({ ...current, goal: event.target.value }))}
                />
              </div>

              <div className="form-grid">
                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={settingsDraft.launchUiOnLoad}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, launchUiOnLoad: event.target.checked }))}
                  />
                  Launch native UI when a ROM loads
                </label>

                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={settingsDraft.liveScreen}
                    onChange={(event) => setSettingsDraft((current) => ({ ...current, liveScreen: event.target.checked }))}
                  />
                  Enable live screen polling
                </label>
              </div>

              <div className="field">
                <label>Screen refresh interval (ms)</label>
                <input
                  type="number"
                  min={250}
                  max={2000}
                  step={50}
                  value={settingsDraft.screenRefreshMs}
                  onChange={(event) => setSettingsDraft((current) => ({ ...current, screenRefreshMs: Number(event.target.value) || 500 }))}
                />
              </div>

              <div className="callout callout--muted">
                Provider availability is fetched from the backend. The selected provider is currently{' '}
                <strong>{providers[settingsDraft.provider]?.available ? 'available' : 'not confirmed'}</strong>.
              </div>
            </div>

            <div className="modal__footer">
              <button className="button button--ghost" onClick={() => setSettingsOpen(false)}>
                Cancel
              </button>
              <button className="button button--primary" onClick={() => void handleSaveSettings()}>
                Save changes
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

const Panel: React.FC<{
  title: string;
  subtitle: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, subtitle, action, children }) => (
  <section className="panel">
    <div className="panel__header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
      {action ? <div className="panel__action">{action}</div> : null}
    </div>
    <div className="panel__body">{children}</div>
  </section>
);

const SummaryCard: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
  secondary: string;
}> = ({ icon, label, value, secondary }) => (
  <div className="summary-card">
    <div className="summary-card__icon">{icon}</div>
    <div>
      <div className="summary-card__label">{label}</div>
      <div className="summary-card__value">{value}</div>
      <div className="summary-card__secondary">{secondary}</div>
    </div>
  </div>
);

const KeyStat: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="key-stat">
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

const KeyValueRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="key-value-row">
    <span>{label}</span>
    <strong>{value}</strong>
  </div>
);

export default App;
