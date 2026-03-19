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
  visionModel: 'bailian/kimi-k2.5', // Vision model for screen analysis (will be updated from OpenClaw)
  planningModel: 'bailian/glm-5', // Planning model for decisions (will be updated from OpenClaw)
  useDualModel: true, // Enable dual-model by default
  
  // Legacy settings
  autonomousLevel: 'moderate',
  agentPersonality: 'strategic',
  agentObjectives: 'Complete Pokemon Red with safe exploration and clear progress.',
};

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