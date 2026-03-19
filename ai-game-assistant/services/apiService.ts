const DEFAULT_BASE_URL = 'http://localhost:5000';

export type GameButton = 'A' | 'B' | 'START' | 'SELECT' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
export type GameAction = GameButton | 'NOOP';
export type AgentAutonomy = 'passive' | 'moderate' | 'aggressive';
export type ScreenFormat = 'int' | 'hex' | 'binary';

export interface HealthResponse {
  status?: string;
  services?: Record<string, string>;
  [key: string]: unknown;
}

export interface ProviderStatus {
  status: string;
  priority: number;
  error?: string | null;
  available: boolean;
}

export interface GameState {
  running: boolean;
  rom_loaded: boolean;
  rom_name: string;
  screen_available: boolean;
  frame_count: number;
  fps: number;
  emulator: string;
  timestamp: string;
}

export interface AgentStatus {
  connected: boolean;
  agent_name: string;
  mode: string;  // Frontend-friendly mode ('auto' or 'manual' or 'idle')
  actual_mode?: string;  // Actual backend mode (e.g., 'auto_explore', 'auto_battle')
  autonomous_level: AgentAutonomy;
  current_action: string;
  last_decision: string;
  enabled: boolean;
  game_running: boolean;
  timestamp: string;
}

export interface AgentModeResponse {
  mode: string;  // Frontend-friendly mode
  actual_mode?: string;  // Actual backend mode
  enabled: boolean;
  autonomous_level: AgentAutonomy;
  current_action: string;
  last_decision: string;
  valid_modes: string[];
  timestamp: string;
}

export interface ScreenResponse {
  image: string;
  shape?: number[];
  timestamp: number;
  pyboy_frame?: number | null;
  performance?: {
    total_time_ms?: number;
    conversion_time_ms?: number;
    current_fps?: number;
    adaptive_fps_target?: number;
  };
  optimization?: {
    cache_hit?: boolean;
    memory_pressure?: number;
    optimization_enabled?: boolean;
  };
}

export interface MemoryAddress {
  address: number;
  name: string;
  size: number;
}

export interface MemoryValue extends MemoryAddress {
  value: number | null;
  hex: string;
}

export interface MemoryWatch {
  addresses: MemoryAddress[];
  values: Array<MemoryValue | { error: string }>;
  timestamp: string;
}

export interface MemoryReadResponse {
  address: string;
  address_int: number;
  size: number;
  values: number[];
  formatted: Array<number | string>;
  format: ScreenFormat;
  timestamp: string;
}

export interface MemoryWriteResponse {
  success: boolean;
  message: string;
  writes: Array<{ address: string; value: number }>;
  timestamp: string;
}

export interface PartyPokemon {
  slot: number;
  species_id: number | null;
  species_name: string | null;
  level: number | null;
  hp: number | null;
  max_hp: number | null;
  hp_percent: number;
  status: number | null;
  status_text?: string;
  type1: string | null;
  type2: string | null;
  moves: Array<{ id: number; name: string }>;
  ot_id: number | null;
}

export interface PartyResponse {
  party_count: number;
  party: PartyPokemon[];
  timestamp: string;
}

export interface InventoryItem {
  slot: number;
  id: number;
  name: string;
  quantity: number;
}

export interface InventoryResponse {
  money: number;
  money_formatted: string;
  item_count: number;
  items: InventoryItem[];
  timestamp: string;
}

export interface UiStatusResponse {
  ui_status: Record<string, unknown>;
  rom_loaded: boolean;
  active_emulator: string;
}

export interface PerformanceResponse {
  server_performance?: Record<string, unknown>;
  emulator_performance?: Record<string, unknown>;
  system_info?: {
    cpu_count?: number;
    memory_usage_mb?: number;
    multi_process_mode?: boolean;
    timestamp?: number;
  };
}

export interface EmulatorModeResponse {
  multi_process_mode: boolean;
  available_modes: string[];
  current_mode: string;
}

export interface ConfigValidationResponse {
  validation?: {
    valid: boolean;
    missing_required?: string[];
    missing_optional?: string[];
    warnings?: string[];
    api_keys_configured?: number;
  };
  configuration?: Record<string, unknown>;
  timestamp?: number;
  error?: string;
  fallback?: string;
  basic_config?: Record<string, unknown>;
}

export interface UiControlResponse {
  message: string;
  ui_status?: Record<string, unknown>;
}

export interface AiActionResponse {
  action: GameAction;
  provider_used?: string | null;
  history: string[];
  optimization?: {
    cache_hit?: boolean;
    response_time_ms?: number;
    memory_pressure?: number;
    optimization_enabled?: boolean;
  };
}

export interface ChatResponse {
  response: string;
  provider_used?: string | null;
}

export interface RomLoadResponse {
  success?: boolean;
  message?: string;
  rom_name?: string;
  rom_path?: string;
  rom_size?: number;
  emulator_type?: string;
  ui_launched?: boolean;
  timestamp?: string;
}

export interface StatusResponse extends Record<string, unknown> {}

class ApiService {
  private baseUrl = DEFAULT_BASE_URL;

  setBaseUrl(url: string) {
    const normalized = url.trim().replace(/\/+$/, '');
    this.baseUrl = normalized || DEFAULT_BASE_URL;
  }

  getBaseUrl() {
    return this.baseUrl;
  }

  private buildUrl(path: string) {
    return `${this.baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    const isFormData = typeof FormData !== 'undefined' && init.body instanceof FormData;

    if (!isFormData && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    const response = await fetch(this.buildUrl(path), {
      ...init,
      headers,
    });

    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');
    const body = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof body === 'object' && body !== null && 'error' in body
          ? String(body.error)
          : typeof body === 'string' && body
            ? body
            : `Request failed with status ${response.status}`;
      throw new Error(message);
    }

    return body as T;
  }

  getHealth() {
    return this.request<HealthResponse>('/health');
  }

  getStatus() {
    return this.request<StatusResponse>('/api/status');
  }

  getConfig() {
    return this.request<Record<string, unknown>>('/api/config');
  }

  validateConfig() {
    return this.request<ConfigValidationResponse>('/api/config/validate');
  }

  getProvidersStatus() {
    return this.request<Record<string, ProviderStatus>>('/api/providers/status');
  }

  getModels(provider: string) {
    const query = encodeURIComponent(provider);
    return this.request<{ models: string[] }>(`/api/models?provider=${query}`);
  }

  getGameState() {
    return this.request<GameState>('/api/game/state');
  }

  getAgentStatus() {
    return this.request<AgentStatus>('/api/agent/status');
  }

  getAgentMode() {
    return this.request<AgentModeResponse>('/api/agent/mode');
  }

  setAgentMode(payload: {
    mode: string;
    enabled: boolean;
    autonomous_level: AgentAutonomy;
    direction?: string;
    target?: string;
  }) {
    return this.request<AgentModeResponse>('/api/agent/mode', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  getScreen() {
    return this.request<ScreenResponse>('/api/screen');
  }

  getMemoryWatch() {
    return this.request<MemoryWatch>('/api/memory/watch');
  }

  readMemory(address: number, size: number, format: ScreenFormat) {
    const params = new URLSearchParams({
      size: String(size),
      format,
    });
    return this.request<MemoryReadResponse>(`/api/memory/${address}?${params.toString()}`);
  }

  writeMemory(address: number, values: number[]) {
    const body = values.length === 1 ? { value: values[0] } : { values };
    return this.request<MemoryWriteResponse>(`/api/memory/${address}`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  getParty() {
    return this.request<PartyResponse>('/api/party');
  }

  getInventory() {
    return this.request<InventoryResponse>('/api/inventory');
  }

  getUiStatus() {
    return this.request<UiStatusResponse>('/api/ui/status');
  }

  launchUi() {
    return this.request<UiControlResponse>('/api/ui/launch', { method: 'POST' });
  }

  stopUi() {
    return this.request<UiControlResponse>('/api/ui/stop', { method: 'POST' });
  }

  restartUi() {
    return this.request<UiControlResponse>('/api/ui/restart', { method: 'POST' });
  }

  getPerformance() {
    return this.request<PerformanceResponse>('/api/performance');
  }

  getEmulatorMode() {
    return this.request<EmulatorModeResponse>('/api/emulator/mode');
  }

  clearCache() {
    return this.request<{ message: string; cleared_caches?: string[] }>('/api/emulator/clear-cache', {
      method: 'POST',
    });
  }

  uploadRom(file: File, emulatorType: string, launchUi: boolean) {
    const formData = new FormData();
    formData.append('rom', file);
    formData.append('emulator_type', emulatorType);
    formData.append('launch_ui', String(launchUi));
    return this.request<RomLoadResponse>('/api/upload-rom', {
      method: 'POST',
      body: formData,
    });
  }

  loadRomFromPath(path: string, emulatorType: string, launchUi: boolean) {
    return this.request<RomLoadResponse>('/api/rom/load', {
      method: 'POST',
      body: JSON.stringify({
        path,
        emulator_type: emulatorType,
        launch_ui: launchUi,
      }),
    });
  }

  saveState() {
    return this.request<{ message?: string }>('/api/save_state', { method: 'POST' });
  }

  loadState() {
    return this.request<{ message?: string }>('/api/load_state', { method: 'POST' });
  }

  pressButton(button: GameButton) {
    return this.request<{ success: boolean; button: string; timestamp: string }>('/api/game/button', {
      method: 'POST',
      body: JSON.stringify({ button }),
    });
  }

  executeAction(action: GameAction, frames: number) {
    return this.request<{ message: string; action: GameAction; frames: number; history_length: number }>('/api/action', {
      method: 'POST',
      body: JSON.stringify({ action, frames }),
    });
  }

  requestAiAction(payload: {
    api_name?: string;
    api_key?: string;
    api_endpoint?: string;
    model?: string;
    goal: string;
  }) {
    return this.request<AiActionResponse>('/api/ai-action', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  chat(payload: {
    message: string;
    api_name?: string;
    api_key?: string;
    api_endpoint?: string;
    model?: string;
  }) {
    return this.request<ChatResponse>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

const apiService = new ApiService();

export default apiService;
