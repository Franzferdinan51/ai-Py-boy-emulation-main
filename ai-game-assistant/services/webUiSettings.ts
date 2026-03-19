import { resolveDefaultBackendUrl, type AppSettings } from './apiService';

export const SETTINGS_STORAGE_KEY = 'openclaw_webui_settings';

export const DEFAULT_SETTINGS: AppSettings = {
  backendUrl: resolveDefaultBackendUrl(),
  openclawMcpEndpoint: 'http://localhost:18789',
  emulatorType: 'gb',
  launchUiOnRomLoad: true,
  autoConnect: true,
  aiActionInterval: 5000,
  agentMode: true,
  visionModel: 'kimi-k2.5',
  autonomousLevel: 'moderate',
  agentPersonality: 'strategic',
  agentObjectives: 'Complete Pokemon Red with safe exploration and clear progress.',
};

export const normalizeSettings = (value: Partial<AppSettings> | null | undefined): AppSettings => ({
  ...DEFAULT_SETTINGS,
  ...(value || {}),
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
