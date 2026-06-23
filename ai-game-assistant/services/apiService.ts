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

  // Always use relative URL - proxy server handles forwarding to backend
  // This ensures mobile and desktop both work without configuration
  if (typeof window !== 'undefined') {
    const { hostname } = window.location;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `http://localhost:${DEFAULT_BACKEND_PORT}`;
    }
    // Use relative URL for remote access - proxy will forward
    return '';
  }

  return `http://localhost:${DEFAULT_BACKEND_PORT}`;
};

const DEFAULT_BASE_URL = resolveDefaultBackendUrl();

export type GameButton = 'A' | 'B' | 'START' | 'SELECT' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
export type GameAction = GameButton | 'NOOP';
export type AgentAutonomy = 'passive' | 'moderate' | 'aggressive';
export type AgentPersonality = 'strategic' | 'casual' | 'speedrun' | 'explorer';
export type EmulatorType = 'gb' | 'gba';
export type AiProvider = 'openclaw' | 'lmstudio' | 'gemini' | 'openrouter' | 'openai-compatible' | 'nvidia' | 'mock' | 'tetris-genetic';

export interface ModelInfo {
  // Core identity
  id: string;
  name: string;
  label: string;  // Display name with category suffix
  provider: string;
  
  // Classification
  category: 'vision' | 'reasoning' | 'general';
  capabilities: string[];
  role: 'primary' | 'vision' | 'planning' | 'fallback' | 'general';
  
  // Capability flags
  is_vision_capable: boolean;
  is_free: boolean;
  manual_allowed: boolean;
  is_default: boolean;
  
  // Metadata
  context_window: number;
  priority: number;
  description: string;
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

export interface AgentGoalResponse {
  goal: string;
  task: string;
  timestamp: string;
}

export interface AgentRunEvent {
  version?: string;
  kind: string;
  timestamp: string;
  source?: string;
  session_id?: string | null;
  success?: boolean;
  action?: unknown;
  observation?: unknown;
  changes?: unknown;
  data?: Record<string, unknown>;
}

export interface AgentCapabilityTool {
  name: string;
  access: string;
  category: string;
  backend_route: string;
  mcp_tool: string;
  description: string;
}

export interface AgentCapabilityMemoryPattern {
  sequence?: string[];
  outcome?: string;
  note?: string;
  timestamp?: string;
}

export interface AgentCapabilityMemorySummary {
  total_records: number;
  by_type: Record<string, number>;
  latest_by_type: Record<string, unknown>;
  recent_notes: Array<{ type?: string; text?: string }>;
  learned_control_patterns: AgentCapabilityMemoryPattern[];
}

export interface AgentCapabilityNextAction {
  action: string;
  reason: string;
  source?: string;
  target?: string;
}

export interface AgentCapabilityAutoLearningSignals {
  control_patterns_observed: number;
  suggested_routine_count: number;
  skill_draft_count: number;
}

export interface AgentCapabilityToolbeltSnapshot {
  active_session_id: string | null;
  active_routine: string | null;
  available_tools: AgentCapabilityTool[];
  tool_groups: Record<string, string[]>;
  memory_summary: AgentCapabilityMemorySummary;
  next_recommended_action: AgentCapabilityNextAction;
  planner_hint?: AgentCapabilityNextAction;
  auto_learning_signals?: AgentCapabilityAutoLearningSignals;
  timestamp: string;
}

export interface AgentCapabilityRoutineStep {
  action?: string;
  frames?: number;
  notes?: string;
}

export interface AgentCapabilitySkillDraft {
  id: string;
  name: string;
  source: string;
  status: string;
  summary?: string;
  sequence?: string[];
  outcome?: string;
}

export interface AgentCapabilityRoutine {
  id: string;
  name: string;
  description?: string;
  kind: string;
  origin: string;
  status: string;
  tags?: string[];
  steps: AgentCapabilityRoutineStep[];
  updated_at?: string;
  skill_draft?: AgentCapabilitySkillDraft;
  summary?: string;
}

export interface AgentCapabilityRoutinesSnapshot {
  active_session_id: string | null;
  active_routine: string | null;
  routines: AgentCapabilityRoutine[];
  suggested_routines: AgentCapabilityRoutine[];
  skill_drafts: AgentCapabilitySkillDraft[];
  timestamp: string;
}

export interface AgentRunEventsResponse {
  ok?: boolean;
  events: AgentRunEvent[];
  count: number;
  limit?: number;
  session_id?: string | null;
  loaded?: boolean;
  active_emulator?: string | null;
  rom_loaded?: boolean;
  timestamp: string;
}

export interface AgentStateSnapshot {
  mode: string;
  enabled: boolean;
  current_goal?: string | null;
  current_task?: string | null;
  current_action?: string | null;
  last_decision?: string | null;
  timestamp?: string | null;
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
    // Use relative URL if baseUrl is empty (proxy mode)
    if (!this.baseUrl) {
      return path.startsWith('/') ? path : `/${path}`;
    }
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

  getAgentGoal() {
    return this.request<AgentGoalResponse>('/api/agent/goal');
  }

  getAgentMode() {
    return this.request<AgentModeResponse>('/api/agent/mode');
  }

  getAgentRunEvents(limit = 20, sessionId?: string) {
    const params = new URLSearchParams();
    if (typeof limit === 'number' && Number.isFinite(limit)) {
      params.set('limit', String(limit));
    }
    if (sessionId) {
      params.set('session_id', sessionId);
    }
    const query = params.toString();
    return this.request<AgentRunEventsResponse>(`/api/agent/runs/events${query ? `?${query}` : ''}`);
  }

  getAgentToolbelt() {
    return this.request<AgentCapabilityToolbeltSnapshot>('/api/agent/toolbelt');
  }

  getAgentRoutines() {
    return this.request<AgentCapabilityRoutinesSnapshot>('/api/agent/routines');
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

  // =========================================
  // Sound Control API Methods
  // =========================================

  /**
   * Get current sound configuration and status
   */
  getSoundStatus() {
    return this.request<{
      emulation_enabled: boolean;
      volume: number;
      output_enabled: boolean;
      sdl_audiodriver: string;
      sample_rate?: number | null;
      buffer_length?: number | null;
      message?: string;
    }>('/api/sound/status');
  }

  /**
   * Enable or disable sound emulation
   */
  setSoundEnabled(enabled: boolean) {
    return this.request<{
      success: boolean;
      emulation_enabled: boolean;
      message: string;
    }>('/api/sound/enable', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  }

  /**
   * Set sound volume (0-100)
   */
  setSoundVolume(volume: number) {
    return this.request<{
      success: boolean;
      volume: number;
      message: string;
    }>('/api/sound/volume', {
      method: 'POST',
      body: JSON.stringify({ volume }),
    });
  }

  /**
   * Enable or disable actual audio output (speaker)
   */
  setSoundOutput(enabled: boolean) {
    return this.request<{
      success: boolean;
      output_enabled: boolean;
      message: string;
    }>('/api/sound/output', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  }

  /**
   * Get current sound buffer (raw audio data)
   */
  getSoundBuffer() {
    return this.request<{
      samples: number;
      channels: number;
      sample_rate: number | null;
      data: string | null;
      format: string;
      message?: string;
    }>('/api/sound/buffer');
  }

  // =========================================
  // agent_features — sessions / events / telemetry / memory / collision
  // =========================================

  // ----- Sessions ---------------------------------------------------

  listSessions() {
    return this.request<{
      sessions: Array<{
        id: string;
        name: string;
        rom_name?: string | null;
        emulator?: string | null;
        created_at?: string;
        updated_at?: string;
        active: boolean;
        milestones_count?: number;
        objectives_count?: number;
      }>;
      active_session_id: string | null;
      count: number;
    }>('/api/games');
  }

  getCurrentSession() {
    return this.request<{
      session: Record<string, unknown> | null;
    }>('/api/games/current');
  }

  createSession(payload: { name?: string; rom_name?: string; emulator?: string; objectives?: string[] } = {}) {
    return this.request<{
      session: { id: string; name: string; [k: string]: unknown };
      message: string;
    }>('/api/games/new', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  activateSession(sessionId: string) {
    return this.request<{ success: boolean; session_id: string; message?: string }>(
      `/api/games/${encodeURIComponent(sessionId)}/activate`,
      { method: 'POST' }
    );
  }

  deleteSession(sessionId: string) {
    return this.request<{ success: boolean; message?: string }>(
      `/api/games/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE' }
    );
  }

  saveSessionState(sessionId: string) {
    return this.request<{ success: boolean; bytes?: number; message?: string }>(
      `/api/games/${encodeURIComponent(sessionId)}/save_state`,
      { method: 'POST' }
    );
  }

  loadSessionState(sessionId: string) {
    return this.request<{ success: boolean; bytes?: number; message?: string }>(
      `/api/games/${encodeURIComponent(sessionId)}/save_state`,
      { method: 'GET' }
    );
  }

  // ----- Events (reasoning stream) -----------------------------------

  listEvents(params: { kind?: string; session_id?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.kind) qs.set('kind', params.kind);
    if (params.session_id) qs.set('session_id', params.session_id);
    if (params.limit !== undefined) qs.set('limit', String(params.limit));
    const tail = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<{
      events: Array<{
        id: string | number;
        kind: string;
        session_id?: string | null;
        message: string;
        data?: Record<string, unknown>;
        timestamp: string;
      }>;
      count: number;
    }>(`/api/agent/events${tail}`);
  }

  postEvent(payload: {
    kind: 'THINK' | 'DECIDE' | 'ACT' | 'MILESTONE' | 'ALERT' | 'OBSERVE' | 'REFLECT';
    message: string;
    session_id?: string;
    data?: Record<string, unknown>;
  }) {
    return this.request<{ success: boolean; event?: Record<string, unknown> }>(
      '/api/agent/events',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  getEventStats() {
    return this.request<{
      total: number;
      by_kind: Record<string, number>;
      by_session: Record<string, number>;
    }>('/api/agent/events/stats');
  }

  clearEvents() {
    return this.request<{ success: boolean; cleared?: number }>(
      '/api/agent/events/clear',
      { method: 'POST' }
    );
  }

  // ----- Telemetry (stuck-meter, blackouts, etc.) --------------------

  getTelemetry(sessionId?: string) {
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    return this.request<{
      session_id?: string | null;
      stuck_meter: number;
      position_history_len: number;
      last_positions: Array<{ x: number; y: number; map_id?: number; ts: number }>;
      actions_total: number;
      actions_success: number;
      actions_failure: number;
      battles_won: number;
      battles_lost: number;
      blackouts: number;
      party_hp_total?: number | null;
      party_hp_max?: number | null;
      ts: string;
    }>(`/api/agent/telemetry${qs}`);
  }

  recordPosition(payload: { x: number; y: number; map_id?: number; session_id?: string }) {
    return this.request<{ success: boolean; stuck_meter: number; alert_emitted?: boolean }>(
      '/api/agent/telemetry/position',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  recordAction(payload: { action: string; result: 'success' | 'failure'; session_id?: string; details?: string }) {
    return this.request<{ success: boolean }>(
      '/api/agent/telemetry/action',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  resetTelemetry(sessionId?: string) {
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    return this.request<{ success: boolean }>(`/api/agent/telemetry/reset`, {
      method: 'POST',
      body: JSON.stringify({}),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // ----- Memory (KnowledgeBase) -------------------------------------

  getMemory(params: { type?: string; session_id?: string; limit?: number } = {}) {
    const qs = new URLSearchParams();
    if (params.type) qs.set('type', params.type);
    if (params.session_id) qs.set('session_id', params.session_id);
    if (params.limit !== undefined) qs.set('limit', String(params.limit));
    const tail = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<{
      entries: Array<{
        id: string | number;
        type: string;
        key?: string;
        value?: unknown;
        message?: string;
        session_id?: string;
        timestamp: string;
        tags?: string[];
      }>;
      count: number;
    }>(`/api/agent/memory${tail}`);
  }

  getMemorySummary() {
    return this.request<{
      total: number;
      by_type: Record<string, number>;
      latest_by_type: Record<string, { timestamp: string; message?: string; key?: string }>;
    }>('/api/agent/memory/summary');
  }

  searchMemory(q: string, sessionId?: string) {
    const qs = new URLSearchParams({ q });
    if (sessionId) qs.set('session_id', sessionId);
    return this.request<{
      matches: Array<{
        id: string | number;
        type: string;
        message?: string;
        score: number;
        timestamp: string;
      }>;
      count: number;
    }>(`/api/agent/memory/search?${qs.toString()}`);
  }

  addMemoryNote(payload: { message: string; tags?: string[]; session_id?: string }) {
    return this.request<{ success: boolean; entry?: Record<string, unknown> }>(
      '/api/agent/memory/note',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  addMemoryLocation(payload: { name: string; map_id?: number; x?: number; y?: number; notes?: string; session_id?: string }) {
    return this.request<{ success: boolean; entry?: Record<string, unknown> }>(
      '/api/agent/memory/location',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  completeMemoryObjective(payload: { objective: string; session_id?: string }) {
    return this.request<{ success: boolean; entry?: Record<string, unknown> }>(
      '/api/agent/memory/objective_complete',
      { method: 'POST', body: JSON.stringify(payload) }
    );
  }

  // ----- Collision (spatial) -----------------------------------------

  getCollisionGrid(width = 32, height = 32) {
    return this.request<{
      ok: boolean;
      ascii?: string;
      labeled?: string;
      width: number;
      height: number;
      provider?: string;
      error?: string;
    }>(`/api/spatial/collision?width=${width}&height=${height}`);
  }

  getGridText(width = 20, height = 18) {
    return this.request<{
      ok: boolean;
      text?: string;
      width: number;
      height: number;
      provider?: string;
      error?: string;
    }>(`/api/spatial/grid/text?width=${width}&height=${height}`);
  }
}

const apiService = new ApiService();

export default apiService;
