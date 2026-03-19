import React, { useState, useEffect, useCallback } from 'react';

interface Settings {
  apiProvider: string;
  apiEndpoint: string;
  apiKey: string;
  visionModel: string;
  autonomousLevel: string;
  agentPersonality: string;
  agentObjectives: string;
  openclawMcpEndpoint: string;
  agentMode: boolean;
  autoConnect: boolean;
  backendUrl: string;
}

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (newSettings: Settings) => void;
}

// Auto-detect OpenClaw endpoint
const AUTO_DETECT_OPENCLAW = 'http://localhost:18789';

// Vision-capable models
const VISION_MODELS = [
  // ChatGPT Plus (OAuth)
  { id: 'gpt-4o', name: 'GPT-4o (ChatGPT Plus)', provider: 'OpenAI', badge: '✨' },
  { id: 'gpt-4o-mini', name: 'GPT-4o-mini (ChatGPT Plus)', provider: 'OpenAI', badge: '✨' },
  // OpenAI
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (OpenAI API)', provider: 'OpenAI', badge: '' },
  // Anthropic
  { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet (Anthropic)', provider: 'Anthropic', badge: '' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus (Anthropic)', provider: 'Anthropic', badge: '' },
  // Bailian (Free)
  { id: 'kimi-k2.5', name: 'kimi-k2.5 (Bailian - Free)', provider: 'Bailian', badge: '🆓' },
  { id: 'qwen-vl-plus', name: 'qwen-vl-plus (Bailian)', provider: 'Bailian', badge: '🆓' },
  // Google
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro (Google)', provider: 'Google', badge: '' },
  { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash (Google)', provider: 'Google', badge: '' },
];

// Text/reasoning models
const TEXT_MODELS = [
  // ChatGPT Plus (OAuth)
  { id: 'gpt-4o', name: 'GPT-4o (ChatGPT Plus)', provider: 'OpenAI', badge: '✨' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (ChatGPT Plus)', provider: 'OpenAI', badge: '✨' },
  // OpenAI API
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo (OpenAI API)', provider: 'OpenAI', badge: '' },
  // Anthropic
  { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet (Anthropic)', provider: 'Anthropic', badge: '' },
  // Bailian (Free)
  { id: 'qwen3.5-plus', name: 'qwen3.5-plus (Bailian - Free)', provider: 'Bailian', badge: '🆓' },
  { id: 'MiniMax-M2.5', name: 'MiniMax-M2.5 (Bailian - Free)', provider: 'Bailian', badge: '🆓' },
  { id: 'glm-5', name: 'glm-5 (Bailian - Free)', provider: 'Bailian', badge: '🆓' },
];

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<{success: boolean; message: string} | null>(null);
  const [activeTab, setActiveTab] = useState<'api' | 'agent' | 'advanced'>('api');
  const [detectedOpenClaw, setDetectedOpenClaw] = useState(false);

  // Initialize with smart defaults
  useEffect(() => {
    if (isOpen) {
      if (!localSettings.openclawMcpEndpoint) {
        setLocalSettings(prev => ({ ...prev, openclawMcpEndpoint: AUTO_DETECT_OPENCLAW }));
      }
      if (!localSettings.backendUrl) {
        setLocalSettings(prev => ({ ...prev, backendUrl: 'http://localhost:5002' }));
      }
      if (!localSettings.visionModel) {
        setLocalSettings(prev => ({ ...prev, visionModel: 'gpt-4o' })); // Default to ChatGPT Plus!
      }
      if (!localSettings.apiProvider) {
        setLocalSettings(prev => ({ ...prev, apiProvider: 'gpt-4o' }));
      }
    }
  }, [isOpen]);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings, isOpen]);

  // Auto-detect OpenClaw
  const detectOpenClaw = useCallback(async () => {
    setTestingConnection(true);
    try {
      const response = await fetch(`${AUTO_DETECT_OPENCLAW}/health`);
      if (response.ok) {
        setDetectedOpenClaw(true);
        setLocalSettings(prev => ({ ...prev, openclawMcpEndpoint: AUTO_DETECT_OPENCLAW }));
        setConnectionStatus({ success: true, message: 'OpenClaw detected!' });
      } else {
        setDetectedOpenClaw(false);
        setConnectionStatus({ success: false, message: 'OpenClaw not found' });
      }
    } catch {
      setDetectedOpenClaw(false);
      setConnectionStatus({ success: false, message: 'OpenClaw not running' });
    }
    setTestingConnection(false);
  }, []);

  // Test backend connection
  const testConnection = useCallback(async () => {
    setTestingConnection(true);
    setConnectionStatus(null);
    try {
      const baseUrl = localSettings.backendUrl || 'http://localhost:5002';
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) {
        setConnectionStatus({ success: true, message: 'Backend connected!' });
      } else {
        setConnectionStatus({ success: false, message: `Server returned ${response.status}` });
      }
    } catch (err) {
      setConnectionStatus({ success: false, message: err instanceof Error ? err.message : 'Connection failed' });
    } finally {
      setTestingConnection(false);
    }
  }, [localSettings.backendUrl]);

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h2 className="text-xl font-semibold text-neutral-200">⚙️ Settings</h2>
          <button onClick={onClose} className="p-1 text-neutral-400 hover:text-white">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-neutral-700">
          {(['api', 'agent', 'advanced'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-4 py-2 text-sm font-medium ${
                activeTab === tab 
                  ? 'text-cyan-400 border-b-2 border-cyan-400 bg-neutral-800/50' 
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              {tab === 'api' ? '🔌 API' : tab === 'agent' ? '🤖 Agent' : '⚡ Advanced'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {activeTab === 'api' && (
            <>
              {/* Provider Info Banner */}
              <div className="p-3 bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-lg border border-green-700/50">
                <p className="text-sm text-green-300">
                  ✨ <strong>ChatGPT Plus detected!</strong> Your subscription includes API access.
                </p>
              </div>

              {/* Vision Model - ChatGPT Plus Priority */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Vision Model (for screen analysis)
                </label>
                <select
                  value={localSettings.visionModel || 'gpt-4o'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, visionModel: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  <optgroup label="✨ ChatGPT Plus (Recommended)">
                    {VISION_MODELS.filter(m => m.provider === 'OpenAI' && m.badge === '✨').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="OpenAI API">
                    {VISION_MODELS.filter(m => m.provider === 'OpenAI' && m.badge !== '✨').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Anthropic">
                    {VISION_MODELS.filter(m => m.provider === 'Anthropic').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="🆓 Bailian (Free)">
                    {VISION_MODELS.filter(m => m.provider === 'Bailian').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Google">
                    {VISION_MODELS.filter(m => m.provider === 'Google').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                </select>
              </div>

              {/* Text Model */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Text/Reasoning Model
                </label>
                <select
                  value={localSettings.apiProvider || 'gpt-4o'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, apiProvider: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  <optgroup label="✨ ChatGPT Plus (Recommended)">
                    {TEXT_MODELS.filter(m => m.badge === '✨').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                  <optgroup label="🆓 Bailian (Free)">
                    {TEXT_MODELS.filter(m => m.provider === 'Bailian').map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                </select>
              </div>

              {/* API Key / OAuth */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  API Key {localSettings.visionModel?.includes('gpt') ? '(or OAuth Token)' : ''}
                </label>
                <input
                  type="password"
                  value={localSettings.apiKey || ''}
                  onChange={e => setLocalSettings(prev => ({ ...prev, apiKey: e.target.value }))}
                  placeholder={localSettings.visionModel?.includes('gpt') ? "sk-... or ChatGPT Plus token" : "API key"}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                />
                {localSettings.visionModel?.includes('gpt') && (
                  <p className="text-xs text-green-400 mt-1">
                    ✨ ChatGPT Plus subscribers get API access included!
                  </p>
                )}
              </div>

              {/* Backend URL */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Game Backend URL
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={localSettings.backendUrl || ''}
                    onChange={e => setLocalSettings(prev => ({ ...prev, backendUrl: e.target.value }))}
                    placeholder="http://localhost:5002"
                    className="flex-1 px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                  />
                  <button
                    onClick={testConnection}
                    disabled={testingConnection}
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm"
                  >
                    {testingConnection ? '...' : 'Test'}
                  </button>
                </div>
                {connectionStatus && (
                  <p className={`text-xs mt-1 ${connectionStatus.success ? 'text-green-400' : 'text-red-400'}`}>
                    {connectionStatus.message}
                  </p>
                )}
              </div>
            </>
          )}

          {activeTab === 'agent' && (
            <>
              <div className="flex items-center justify-between p-3 bg-neutral-800 rounded-lg">
                <div>
                  <label className="text-sm font-medium text-neutral-300">Agent Mode</label>
                  <p className="text-xs text-neutral-500">AI controls the game</p>
                </div>
                <button
                  onClick={() => setLocalSettings(prev => ({ ...prev, agentMode: !prev.agentMode }))}
                  className={`relative w-12 h-6 rounded-full ${localSettings.agentMode ? 'bg-green-500' : 'bg-neutral-600'}`}
                >
                  <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${localSettings.agentMode ? 'left-7' : 'left-1'}`} />
                </button>
              </div>

              <div className="flex items-center justify-between p-3 bg-neutral-800 rounded-lg">
                <div>
                  <label className="text-sm font-medium text-neutral-300">Auto Connect</label>
                  <p className="text-xs text-neutral-500">Connect on startup</p>
                </div>
                <button
                  onClick={() => setLocalSettings(prev => ({ ...prev, autoConnect: !prev.autoConnect }))}
                  className={`relative w-12 h-6 rounded-full ${localSettings.autoConnect ? 'bg-green-500' : 'bg-neutral-600'}`}
                >
                  <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${localSettings.autoConnect ? 'left-7' : 'left-1'}`} />
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-300">Autonomous Level</label>
                <select
                  value={localSettings.autonomousLevel || 'moderate'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, autonomousLevel: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  <option value="passive">Passive - Asks before actions</option>
                  <option value="moderate">Moderate - Decides and acts</option>
                  <option value="aggressive">Aggressive - Fast decisions</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-300">Agent Personality</label>
                <select
                  value={localSettings.agentPersonality || 'strategic'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, agentPersonality: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  <option value="strategic">Strategic - Plans ahead</option>
                  <option value="casual">Casual - Relaxed gameplay</option>
                  <option value="speedrun">Speedrun - Time-focused</option>
                  <option value="explorer">Explorer - Explores everything</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-neutral-300">Current Objectives</label>
                <textarea
                  value={localSettings.agentObjectives || ''}
                  onChange={e => setLocalSettings(prev => ({ ...prev, agentObjectives: e.target.value }))}
                  placeholder="Catch 'em all, Beat the game..."
                  rows={2}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 resize-none"
                />
              </div>
            </>
          )}

          {activeTab === 'advanced' && (
            <>
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  OpenClaw MCP Endpoint
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={localSettings.openclawMcpEndpoint || ''}
                    onChange={e => setLocalSettings(prev => ({ ...prev, openclawMcpEndpoint: e.target.value }))}
                    placeholder="Auto-detected"
                    className="flex-1 px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                  />
                  <button
                    onClick={detectOpenClaw}
                    disabled={testingConnection}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-sm"
                  >
                    🔍 Auto
                  </button>
                </div>
                {detectedOpenClaw && (
                  <p className="text-xs text-green-400 mt-1">✓ OpenClaw detected at localhost:18789</p>
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-4 border-t border-neutral-700">
          <button onClick={onClose} className="px-4 py-2 text-neutral-400 hover:text-white">
            Cancel
          </button>
          <button onClick={handleSave} className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg">
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
