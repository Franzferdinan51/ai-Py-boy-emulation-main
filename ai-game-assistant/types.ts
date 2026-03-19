
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

export interface AppSettings {
  // Legacy fields (for backward compatibility)
  aiActionInterval: number;
  backendUrl?: string;
  aiProvider?: string;
  googleApiKey?: string;
  openrouterApiKey?: string;
  lmStudioUrl?: string;
  selectedModel?: string;
  // New unified fields
  apiProvider?: 'gemini' | 'openrouter' | 'openai-compatible' | 'nvidia' | 'openclaw';
  apiEndpoint?: string;
  apiKey?: string;
  model?: string;
  // OpenClaw Agent Settings
  agentMode: boolean; // Whether agent controls the game (DEFAULT: true)
  openclawMcpEndpoint: string; // OpenClaw MCP server endpoint
  visionModel: 'kimi-k2.5' | 'MiniMax-M2.5' | 'glm-5'; // Vision model for screen analysis
  autonomousLevel: 'passive' | 'moderate' | 'aggressive'; // How autonomous the agent is
  agentObjectives: string; // Agent's current objectives
  agentPersonality: 'strategic' | 'casual' | 'speedrun' | 'explorer'; // Agent behavior style
}

// Agent status for the status panel
export interface AgentStatus {
  connected: boolean;
  agentName: string;
  currentAction: string;
  lastDecision: string;
  heartbeat: number;
  decisionHistory: string[];
}
