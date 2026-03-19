export interface APIConfig {
  name: string;
  key: string;
  baseUrl?: string;
}

export enum EmulatorMode {
  GB = 'gb',
  GBA = 'gba',
}

export enum AIState {
  IDLE = 'idle',
  RUNNING = 'running',
  THINKING = 'thinking',
  ERROR = 'error',
}

export type GameAction = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'A' | 'B' | 'START' | 'SELECT';

export interface AILog {
  id: number;
  message: string;
  type: 'info' | 'action' | 'thought' | 'error';
}

export interface ChatMessage {
  id: number;
  sender: 'user' | 'ai';
  text: string;
}

// ============================================================================
// DUAL-MODEL ARCHITECTURE TYPES
// ============================================================================

/**
 * Vision model options - specialized for image/screen analysis
 * These models have vision capabilities and can understand game screenshots
 */
export type VisionModel = 'kimi-k2.5' | 'qwen-vl-plus' | 'glm-4v-flash' | 'MiniMax-M2.7';

/**
 * Planning model options - specialized for reasoning and decision making
 * These models excel at game strategy and action planning
 */
export type PlanningModel = 'glm-5' | 'qwen3.5-plus' | 'MiniMax-M2.7' | 'MiniMax-M2.5';

/**
 * Model configuration for dual-model architecture
 */
export interface DualModelConfig {
  visionModel: VisionModel;
  planningModel: PlanningModel;
  useDualModel: boolean;
}

export interface AppSettings {
  // Legacy fields (for backward compatibility)
  aiActionInterval: number;
  backendUrl: string;
  aiProvider?: string;
  googleApiKey?: string;
  openrouterApiKey?: string;
  lmStudioUrl?: string;
  selectedModel?: string;
  // New unified fields
  apiProvider?: 'gemini' | 'openrouter' | 'openai-compatible' | 'nvidia' | 'openclaw' | 'lmstudio';
  apiEndpoint?: string;
  apiKey?: string;
  model?: string;
  // OpenClaw Agent Settings
  agentMode: boolean; // Whether agent controls the game (DEFAULT: true)
  openclawMcpEndpoint: string; // OpenClaw MCP server endpoint
  
  // Dual-model architecture settings (NEW)
  visionModel: VisionModel; // Vision model for screen analysis (independent)
  planningModel: PlanningModel; // Planning/thinking model for decisions (independent)
  useDualModel: boolean; // Enable dual-model architecture (default: true)
  
  // Legacy settings (kept for backward compatibility)
  autonomousLevel: 'passive' | 'moderate' | 'aggressive'; // How autonomous the agent is
  agentObjectives: string; // Agent's current objectives
  agentPersonality: 'strategic' | 'casual' | 'speedrun' | 'explorer'; // Agent behavior style
  
  // Connection settings
  autoConnect: boolean; // Whether to auto-connect on startup
  
  // LM Studio / Local Model Settings (legacy)
  lmStudioThinkingModel?: string;
  lmStudioVisionModel?: string;
  customEndpoint?: string;
  customThinkingModel?: string;
  customVisionModel?: string;
}

// Agent status for the status panel
export interface AgentStatus {
  connected: boolean;
  agentName: string;
  currentAction: string;
  lastDecision: string;
  heartbeat: number;
  decisionHistory: string[];
  // Dual-model status (NEW)
  visionModel?: VisionModel;
  planningModel?: PlanningModel;
  lastVisionAnalysis?: string;
  lastPlanningReasoning?: string;
}

// Dual-model API response types
export interface VisionAnalysisResponse {
  game_state: string;
  player_position?: string;
  nearby_entities?: string[];
  ui_elements?: string[];
  danger_level: 'low' | 'medium' | 'high' | 'unknown';
  opportunities?: string[];
  raw_description: string;
}

export interface PlanningResponse {
  action: GameAction;
  reasoning: string;
  confidence: number;
  alternative_actions?: GameAction[];
  expected_outcome?: string;
}

export interface DualModelActionResponse {
  action: GameAction;
  models_used: string; // e.g., "vision:kimi-k2.5+planning:glm-5"
  vision_analysis?: VisionAnalysisResponse;
  planning_result?: PlanningResponse;
}