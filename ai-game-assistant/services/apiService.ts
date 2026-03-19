/**
 * API Service for AI Game Assistant
 * Uses the new REST API endpoints:
 * - GET /api/game/state - game state
 * - GET /api/agent/status - agent status
 * - POST /api/game/button - press button
 * - GET /api/memory/watch - memory inspector
 * - GET /api/screen - game screen
 */

const DEFAULT_BASE_URL = 'http://localhost:5000';

// Game State Types
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

// Agent Status Types
export interface AgentStatus {
  connected: boolean;
  agent_name: string;
  mode: 'auto' | 'manual';
  autonomous_level: 'passive' | 'moderate' | 'aggressive';
  current_action: string;
  last_decision: string;
  enabled: boolean;
  game_running: boolean;
  timestamp: string;
}

// Memory Watch Types
export interface MemoryAddress {
  address: number;
  name: string;
  size: number;
}

export interface MemoryValue {
  address: number;
  name: string;
  size: number;
  value: number | null;
  hex: string;
}

export interface MemoryWatch {
  addresses: MemoryAddress[];
  values: MemoryValue[];
  timestamp: string;
}

// Log Entry
export interface LogEntry {
  id: number;
  timestamp: string;
  type: 'info' | 'action' | 'error' | 'system';
  message: string;
}

// Button press types
export type GameButton = 'A' | 'B' | 'START' | 'SELECT' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';

// API Service Class
class ApiService {
  private baseUrl: string;
  private logEntries: LogEntry[] = [];
  private logIdCounter = 0;

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  private addLog(type: LogEntry['type'], message: string) {
    const entry: LogEntry = {
      id: this.logIdCounter++,
      timestamp: new Date().toISOString(),
      type,
      message,
    };
    this.logEntries = [...this.logEntries.slice(-99), entry]; // Keep last 100 entries
    return entry;
  }

  getLogs(): LogEntry[] {
    return this.logEntries;
  }

  clearLogs() {
    this.logEntries = [];
  }

  private async fetchWithErrorHandling<T>(url: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      this.addLog('error', `API Error: ${message}`);
      throw error;
    }
  }

  // GET /api/game/state - Get current game state
  async getGameState(): Promise<GameState> {
    this.addLog('info', 'Fetching game state...');
    return this.fetchWithErrorHandling<GameState>(`${this.baseUrl}/api/game/state`);
  }

  // GET /api/agent/status - Get agent status
  async getAgentStatus(): Promise<AgentStatus> {
    this.addLog('info', 'Fetching agent status...');
    return this.fetchWithErrorHandling<AgentStatus>(`${this.baseUrl}/api/agent/status`);
  }

  // POST /api/game/button - Press a game button
  async pressButton(button: GameButton): Promise<{ success: boolean; button: string }> {
    this.addLog('action', `Pressing button: ${button}`);
    return this.fetchWithErrorHandling<{ success: boolean; button: string }>(`${this.baseUrl}/api/game/button`, {
      method: 'POST',
      body: JSON.stringify({ button }),
    });
  }

  // GET /api/screen - Get game screen as blob
  async getScreen(): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/screen`);
    if (!response.ok) {
      throw new Error(`Failed to get screen: ${response.statusText}`);
    }
    return response.blob();
  }

  // GET /api/memory/watch - Get watched memory addresses
  async getMemoryWatch(): Promise<MemoryWatch> {
    this.addLog('info', 'Fetching memory watch...');
    return this.fetchWithErrorHandling<MemoryWatch>(`${this.baseUrl}/api/memory/watch`);
  }

  // Legacy endpoints - keeping for backward compatibility
  async checkBackendStatus(): Promise<{ success: boolean; message: string }> {
    try {
      const state = await this.getGameState();
      return { success: true, message: 'Connected to server' };
    } catch (error) {
      return { 
        success: false, 
        message: error instanceof Error ? error.message : 'Connection failed' 
      };
    }
  }

  async sendAction(action: string): Promise<void> {
    const buttonMap: Record<string, GameButton> = {
      'UP': 'UP',
      'DOWN': 'DOWN',
      'LEFT': 'LEFT',
      'RIGHT': 'RIGHT',
      'A': 'A',
      'B': 'B',
      'START': 'START',
      'SELECT': 'SELECT',
    };
    
    const button = buttonMap[action];
    if (button) {
      await this.pressButton(button);
    }
  }

  async loadRom(romFile: File): Promise<void> {
    const formData = new FormData();
    formData.append('rom', romFile);

    const response = await fetch(`${this.baseUrl}/api/upload-rom`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: 'Failed to load ROM' }));
      throw new Error(errorData.message || 'Failed to load ROM');
    }

    this.addLog('system', `ROM loaded: ${romFile.name}`);
  }

  async saveState(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/save_state`, { method: 'POST' });
    if (!response.ok) {
      throw new Error('Failed to save state');
    }
    this.addLog('system', 'Game state saved');
  }

  async loadState(): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/load_state`, { method: 'POST' });
    if (!response.ok) {
      throw new Error('Failed to load state');
    }
    this.addLog('system', 'Game state loaded');
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;