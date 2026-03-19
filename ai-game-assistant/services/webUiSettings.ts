import { resolveDefaultBackendUrl, type AppSettings } from './apiService';

export const SETTINGS_STORAGE_KEY = 'openclaw_webui_settings';

/**
 * Default settings for the WebUI
 * 
 * Dual-model architecture is enabled by default:
 * - Vision Model: kimi-k2.5 (excellent at game screen analysis)
 * - Planning Model: glm-5 (fast, capable decision making)
 * 
 * Note: Models are now dynamically discovered from OpenClaw.
 * These defaults are fallbacks when OpenClaw is unavailable.
 */
export const DEFAULT_SETTINGS: AppSettings = {
  backendUrl: resolveDefaultBackendUrl(),
  openclawMcpEndpoint: 'http://localhost:18789',
  emulatorType: 'gb',
  launchUiOnRomLoad: true,
  autoConnect: true,
  aiActionInterval: 5000,
  agentMode: true,
  
  // Dual-model architecture settings (NEW)
  visionModel: 'kimi-k2.5',
  planningModel: 'glm-5',
  useDualModel: true, // Enable dual-model by default

  // Provider/runtime defaults
  aiProvider: 'openclaw',
  lmStudioUrl: 'http://localhost:1234/v1',
  lmStudioThinkingModel: '',
  lmStudioVisionModel: '',
  
  // Legacy settings
  autonomousLevel: 'moderate',
  agentPersonality: 'strategic',
  agentObjectives: 'Complete Pokemon Red with safe exploration and clear progress.',
};

export const VISION_MODEL_OPTIONS = [
  { value: 'kimi-k2.5', label: 'Kimi K2.5', description: 'Best for game screen analysis (FREE)' },
  { value: 'qwen-vl-plus', label: 'Qwen VL Plus', description: 'Alternative OpenClaw vision profile.' },
  { value: 'glm-4v-flash', label: 'GLM-4V Flash', description: 'Fast multimodal fallback.' },
  { value: 'MiniMax-M2.7', label: 'MiniMax M2.7', description: 'Balanced multimodal option.' },
] as const;

export const PLANNING_MODEL_OPTIONS = [
  { value: 'glm-5', label: 'GLM-5', description: 'Fast decisions, great for games.' },
  { value: 'qwen3.5-plus', label: 'Qwen 3.5 Plus', description: 'Higher reasoning quality.' },
  { value: 'MiniMax-M2.7', label: 'MiniMax M2.7', description: 'Balanced multimodal option.' },
  { value: 'MiniMax-M2.5', label: 'MiniMax M2.5', description: 'Reliable unlimited planner.' },
] as const;

export const AI_PROVIDER_OPTIONS = [
  { value: 'openclaw', label: 'OpenClaw MCP', description: 'OpenClaw-native routing with dual-model support.' },
  { value: 'lmstudio', label: 'LM Studio', description: 'Local OpenAI-compatible models with separate thinking and vision config.' },
  { value: 'gemini', label: 'Google Gemini', description: 'Server-side Gemini provider.' },
  { value: 'openrouter', label: 'OpenRouter', description: 'Multi-provider gateway for cloud models.' },
  { value: 'openai-compatible', label: 'OpenAI Compatible', description: 'Generic OpenAI-compatible endpoints.' },
  { value: 'nvidia', label: 'NVIDIA NIM', description: 'NVIDIA-hosted models.' },
] as const;

export const normalizeSettings = (value: Partial<AppSettings> | null | undefined): AppSettings => ({
  ...DEFAULT_SETTINGS,
  ...(value || {}),
  // Ensure dual-model fields are set if missing from legacy settings
  visionModel: value?.visionModel || DEFAULT_SETTINGS.visionModel,
  planningModel: value?.planningModel || DEFAULT_SETTINGS.planningModel,
  useDualModel: value?.useDualModel ?? DEFAULT_SETTINGS.useDualModel,
});

export const loadSettings = (): AppSettings => {
  try {
    const stored = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!stored) {
      return DEFAULT_SETTINGS;
    }

    return normalizeSettings(JSON.parse(stored) as Partial<AppSettings>);
  } catch (error) {
    console.error('Failed to load WebUI settings:', error);
    return DEFAULT_SETTINGS;
  }
};

export const saveSettings = (settings: AppSettings) => {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch (error) {
    console.error('Failed to save WebUI settings:', error);
  }
};
