import { resolveDefaultBackendUrl, type AppSettings } from './apiService';

export const SETTINGS_STORAGE_KEY = 'openclaw_webui_settings';

/**
 * Default settings for the WebUI
 * 
 * Dual-model architecture is enabled by default:
 * - Vision Model: kimi-k2.5 (excellent at game screen analysis)
 * - Planning Model: glm-5 (fast, capable decision making)
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
  visionModel: 'kimi-k2.5', // Vision model for screen analysis
  planningModel: 'glm-5', // Planning model for decisions
  useDualModel: true, // Enable dual-model by default
  
  // Legacy settings
  autonomousLevel: 'moderate',
  agentPersonality: 'strategic',
  agentObjectives: 'Complete Pokemon Red with safe exploration and clear progress.',
};

/**
 * Available vision models with descriptions
 */
export const VISION_MODEL_OPTIONS = [
  { value: 'kimi-k2.5', label: 'Kimi K2.5', description: 'Best for game screen analysis (FREE)' },
  { value: 'qwen-vl-plus', label: 'Qwen VL Plus', description: 'High quality vision (quota)' },
  { value: 'glm-4v-flash', label: 'GLM-4V Flash', description: 'Fast vision model' },
  { value: 'MiniMax-M2.7', label: 'MiniMax M2.7', description: 'Multimodal with vision (FREE)' },
] as const;

/**
 * Available planning models with descriptions
 */
export const PLANNING_MODEL_OPTIONS = [
  { value: 'glm-5', label: 'GLM-5', description: 'Fast decisions, great for games (API credits)' },
  { value: 'qwen3.5-plus', label: 'Qwen 3.5 Plus', description: 'Best reasoning (quota)' },
  { value: 'MiniMax-M2.7', label: 'MiniMax M2.7', description: 'Balanced, multimodal (FREE)' },
  { value: 'MiniMax-M2.5', label: 'MiniMax M2.5', description: 'Unlimited, reliable (FREE)' },
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