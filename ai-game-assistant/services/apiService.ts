/// <reference types="vite/client" />

const DEFAULT_BACKEND_PORT = '5002';

const normalizeBaseUrl = (url: string) => url.trim().replace(/\/+$/, '');

export const resolveDefaultBackendUrl = () => {
  const envUrl =
    typeof import.meta !== 'undefined' && typeof import.meta.env?.VITE_BACKEND_URL === 'string'
      ? import.meta.env.VITE_BACKEND_URL
      : '';

  if (envUrl.trim()) {
    return normalizeBaseUrl(envUrl);
  }

  if (typeof window === 'undefined') {
    return `http://localhost:${DEFAULT_BACKEND_PORT}`;
  }

  const { hostname, origin, protocol } = window.location;
  if (!hostname || protocol === 'file:' || hostname === 'localhost' || hostname === '127.0.0.1') {
    return `${protocol === 'file:' ? 'http:' : protocol}//${hostname || 'localhost'}:${DEFAULT_BACKEND_PORT}`;
  }

  return origin;
};

const DEFAULT_BASE_URL = resolveDefaultBackendUrl();

export type GameButton = 'A' | 'B' | 'START' | 'SELECT' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
export type GameAction = GameButton | 'NOOP';
export type AgentAutonomy = 'passive' | 'moderate' | 'aggressive';
export type AgentPersonality = 'strategic' | 'casual' | 'speedrun' | 'explorer';
export type EmulatorType = 'gb' | 'gba';
export type AiProvider = 'openclaw' | 'lmstudio' | 'gemini' | 'openrouter' | 'openai-compatible' | 'nvidia' | 'mock' | 'tetris-genetic';

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  capabilities: string[];
  context_window: number;
  is_vision_capable: boolean;
  is_free: boolean;
  description: string;
  priority: number;
}

export interface AppSettings {
  backendUrl: string;
  openclawMcpEndpoint: string;
  emulatorType: EmulatorType;
  launchUiOnRomLoad: boolean;
  autoConnect: boolean;
  aiActionInterval: number;
  agentMode: boolean;
  visionModel: string; // Now dynamic instead of hardcoded
  planningModel: string;
  useDualModel: boolean;
  autonomousLevel: AgentAutonomy;
  agentPersonality: AgentPersonality;
  agentObjectives: string;
  // LM Studio / Local Model Settings
  aiProvider?: AiProvider;
  lmStudioUrl?: string;
  lmStudioThinkingModel?: string;
  lmStudioVisionModel?: string;
  customEndpoint?: string;
  customThinkingModel?: string;
  customVisionModel?: string;
}

export interface LogEntry {
  id: number;
  timestamp: string;
  type: 'info' | 'action' | 'error' | 'system' | 'vision' | 'thought';
  message: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
  checks: Record<string, string>;
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
  mode: string;
  actual_mode?: string;  // Actual backend mode (auto_explore, manual, etc.)
  autonomous_level: AgentAutonomy;
  current_action: string;
  last_decision: string;
  enabled: boolean;
  game_running: boolean;
  provider?: string;     // Current AI provider being used
  model?: string;
  vision_model?: string; // Current vision model
  planning_model?: string;
  use_dual_model?: boolean;
  objectives?: string;   // Current objectives
  personality?: string;  // Agent personality
  timestamp: string;
}

export interface AgentModeResponse {
  success?: boolean;
  message?: string;
  mode: string;
  actual_mode?: string;
  enabled: boolean;
  autonomous_level: AgentAutonomy;
  current_action?: string;
  last_decision?: string;
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
}

// Vision analysis result type (used by VisionAnalysisPanel)
export interface VisionAnalysis {
  screenshot_url: string;
  analysis: string;
  recommended_action: string;
  confidence: number;
  timestamp: string;
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
  values: MemoryValue[];
  timestamp: string;
}

export interface PokemonMove {
  id: number;
  name: string;
}

export interface Pokemon {
  slot: number;
  species_id: number | null;
  species_name: string | null;
  level: number | null;
  hp: number | null;
  max_hp: number | null;
  status: number | null;
  status_text?: string;
  type1: string | null;
  type2: string | null;
  moves: PokemonMove[];
  ot_id: number | null;
  hp_percent?: number;
}

export interface PartyData {
  party_count: number;
  party: Pokemon[];
  timestamp: string;
}

export interface InventoryItem {
  slot: number;
  id: number;
  name: string;
  quantity: number;
}

export interface InventoryData {
  money: number;
  money_formatted: string;
  item_count: number;
  items: InventoryItem[];
  timestamp: string;
}

export interface OpenClawConfig {
  endpoint: string;
  vision_model: AppSettings['visionModel'];
  planning_model: AppSettings['planningModel'];
  use_dual_model: boolean;
  objectives: string;
  personality: AgentPersonality;
  timestamp: string;
  dual_model_status?: DualModelStatus; // NEW: Dual-model status
}

export interface AiRuntimeConfig {
  provider: AiProvider;
  model: string;
  api_endpoint: string;
  available_providers: string[];
  provider_status: Record<string, { status: string; priority: number; error: string | null; available: boolean }>;
  timestamp: string;
}

export interface OpenClawHealthResponse {
  ok: boolean;
  endpoint: string;
  status: number | null;
  service_status?: string | null;
  error?: string | null;
  checked_at: string;
  details?: Record<string, unknown> | null;
}

// NEW: Dual-model architecture types
export interface DualModelConfig {
  vision_model: string;
  planning_model: string;
  use_dual_model: boolean;
  available_vision_models: ModelOption[];
  available_planning_models: ModelOption[];
  dual_model_status?: DualModelStatus;
  timestamp: string;
}

export interface ModelOption {
  id: string;
  name: string;
  description: string;
}

export interface DualModelStatus {
  vision_model: string;
  planning_model: string;
  openclaw_endpoint: string;
  last_vision_response?: string;
  last_planning_response?: string;
  available_vision_models?: string[];
  available_planning_models?: string[];
  timestamp?: string;
}

class ApiService {
  private baseUrl = DEFAULT_BASE_URL;

  setBaseUrl(url: string) {
    const normalized = normalizeBaseUrl(url);
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
    const payload = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof payload === 'object' && payload !== null && 'error' in payload
          ? String(payload.error)
          : typeof payload === 'string' && payload
            ? payload
            : `Request failed with status ${response.status}`;
      throw new Error(message);
    }

    return payload as T;
  }

  getHealth() {
    return this.request<HealthResponse>('/health');
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

  getParty() {
    return this.request<PartyData>('/api/party');
  }

  getInventory() {
    return this.request<InventoryData>('/api/inventory');
  }

  uploadRom(file: File, emulatorType: EmulatorType, launchUi: boolean) {
    const formData = new FormData();
    formData.append('rom', file);
    formData.append('emulator_type', emulatorType);
    formData.append('launch_ui', String(launchUi));

    return this.request<{
      message: string;
      rom_name: string;
      original_filename?: string;
      emulator_type?: string;
      ui_launched?: boolean;
    }>('/api/upload-rom', {
      method: 'POST',
      body: formData,
    });
  }

  saveState() {
    return this.request<{ success?: boolean; message?: string }>('/api/save_state', {
      method: 'POST',
    });
  }

  loadState() {
    return this.request<{ success?: boolean; message?: string }>('/api/load_state', {
      method: 'POST',
    });
  }

  pressButton(button: GameButton) {
    return this.request<{ success: boolean; button: string; timestamp: string }>('/api/game/button', {
      method: 'POST',
      body: JSON.stringify({ button }),
    });
  }

  getOpenClawConfig() {
    return this.request<OpenClawConfig>('/api/openclaw/config');
  }

  updateOpenClawConfig(payload: {
    endpoint: string;
    vision_model: AppSettings['visionModel'];
    planning_model: AppSettings['planningModel'];
    use_dual_model: boolean;
    objectives: string;
    personality: AgentPersonality;
  }) {
    return this.request<OpenClawConfig>('/api/openclaw/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  checkOpenClawHealth(endpoint?: string) {
    const params = endpoint ? `?endpoint=${encodeURIComponent(endpoint)}` : '';
    return this.request<OpenClawHealthResponse>(`/api/openclaw/health${params}`);
  }

  getProviderStatus() {
    return this.request<Record<string, { status: string; priority: number; error: string | null; available: boolean }>>('/api/providers/status');
  }

  getAiRuntimeConfig() {
    return this.request<AiRuntimeConfig>('/api/ai/runtime');
  }

  updateAiRuntimeConfig(payload: {
    provider: AiProvider;
    model: string;
    api_endpoint: string;
  }) {
    return this.request<AiRuntimeConfig>('/api/ai/runtime', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // LM Studio API methods
  getLmStudioConfig() {
    return this.request<{
      endpoint: string;
      thinking_model: string;
      vision_model: string;
      timestamp: string;
    }>('/api/lmstudio/config');
  }

  updateLmStudioConfig(payload: {
    endpoint: string;
    thinking_model: string;
    vision_model: string;
  }) {
    return this.request<{
      endpoint: string;
      thinking_model: string;
      vision_model: string;
      message: string;
      timestamp: string;
    }>('/api/lmstudio/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  getLmStudioModels() {
    return this.request<{
      models: string[];
      timestamp: string;
    }>('/api/lmstudio/models');
  }

  // OpenClaw Model Discovery endpoints
  getOpenClawModels(refresh = false) {
    const params = refresh ? '?refresh=true' : '';
    return this.request<{
      models: ModelInfo[];
      timestamp: string;
      cached: boolean;
    }>(`/api/openclaw/models${params}`);
  }

  getVisionModels(refresh = false) {
    const params = refresh ? '?refresh=true' : '';
    return this.request<{
      models: ModelInfo[];
      timestamp: string;
    }>(`/api/openclaw/models/vision${params}`);
  }

  getPlanningModels(refresh = false) {
    const params = refresh ? '?refresh=true' : '';
    return this.request<{
      models: ModelInfo[];
      timestamp: string;
    }>(`/api/openclaw/models/planning${params}`);
  }

  recommendModel(useCase: 'vision' | 'planning' | 'fast' | 'quality' | 'free', refresh = false) {
    const params = `?use_case=${useCase}${refresh ? '&refresh=true' : ''}`;
    return this.request<{
      model: ModelInfo;
      use_case: string;
      timestamp: string;
    }>(`/api/openclaw/models/recommend${params}`);
  }

  // NEW: Dual-model architecture API methods
  
  /**
   * Get dual-model configuration (vision + planning models)
   */
  getDualModelConfig() {
    return this.request<DualModelConfig>('/api/dual-model/config');
  }

  /**
   * Update dual-model configuration
   */
  updateDualModelConfig(payload: {
    vision_model?: string;
    planning_model?: string;
    use_dual_model?: boolean;
  }) {
    return this.request<{
      success: boolean;
      message: string;
      config: {
        vision_model: string;
        planning_model: string;
        use_dual_model: boolean;
      };
      timestamp: string;
    }>('/api/dual-model/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  /**
   * Get detailed dual-model status
   */
  getDualModelStatus() {
    return this.request<DualModelStatus>('/api/dual-model/status');
  }
}

const apiService = new ApiService();

export default apiService;
