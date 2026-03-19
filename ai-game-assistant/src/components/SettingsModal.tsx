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

// Vision-capable models only
const VISION_MODELS = [
  { id: 'kimi-k2.5', name: 'kimi-k2.5 (Bailian - Free)', provider: 'Bailian' },
  { id: 'qwen-vl-plus', name: 'qwen-vl-plus (Bailian)', provider: 'Bailian' },
  { id: 'gpt-4o', name: 'GPT-4o (OpenAI)', provider: 'OpenAI' },
  { id: 'gpt-4o-mini', name: 'GPT-4o-mini (OpenAI)', provider: 'OpenAI' },
  { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet (Anthropic)', provider: 'Anthropic' },
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro (Google)', provider: 'Google' },
];

// Text/reasoning models
const TEXT_MODELS = [
  { id: 'qwen3.5-plus', name: 'qwen3.5-plus (Bailian - Free)', provider: 'Bailian' },
  { id: 'MiniMax-M2.5', name: 'MiniMax-M2.5 (Bailian - Free)', provider: 'Bailian' },
  { id: 'glm-5', name: 'glm-5 (Bailian - Free)', provider: 'Bailian' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (OpenAI)', provider: 'OpenAI' },
  { id: 'claude-3-opus', name: 'Claude 3 Opus (Anthropic)', provider: 'Anthropic' },
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
      // Auto-detect OpenClaw if not set
      if (!localSettings.openclawMcpEndpoint) {
        setLocalSettings(prev => ({ 
          ...prev, 
          openclawMcpEndpoint: AUTO_DETECT_OPENCLAW 
        }));
      }
      // Default backend if not set
      if (!localSettings.backendUrl) {
        setLocalSettings(prev => ({ 
          ...prev, 
          backendUrl: 'http://localhost:5002' 
        }));
      }
      // Default vision model if not set
      if (!localSettings.visionModel) {
        setLocalSettings(prev => ({ 
          ...prev, 
          visionModel: 'kimi-k2.5' 
        }));
      }
      // Default text model if not set
      if (!localSettings.apiProvider) {
        setLocalSettings(prev => ({ 
          ...prev, 
          apiProvider: 'qwen3.5-plus' 
        }));
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
        setLocalSettings(prev => ({ 
          ...prev, 
          openclawMcpEndpoint: AUTO_DETECT_OPENCLAW 
        }));
        setConnectionStatus({ success: true, message: 'OpenClaw detected!' });
      } else {
        setDetectedOpenClaw(false);
        setConnectionStatus({ success: false, message: 'OpenClaw not found at localhost:18789' });
      }
    } catch {
      setDetectedOpenClaw(false);
      setConnectionStatus({ success: false, message: 'OpenClaw not running locally' });
    }
    setTestingConnection(false);
  }, []);

  // Test connection
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
      setConnectionStatus({ 
        success: false, 
        message: err instanceof Error ? err.message : 'Connection failed' 
      });
    } finally {
      setTestingConnection(false);
    }
  }, [localSettings.backendUrl]);

  // Handle save
  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  // Close on escape
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
              {/* Backend URL - Auto-detected */}
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

              {/* Vision Model */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Vision Model (for screen analysis)
                </label>
                <select
                  value={localSettings.visionModel || 'kimi-k2.5'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, visionModel: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  {VISION_MODELS.map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
              </div>

              {/* Text Model */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Text/Reasoning Model
                </label>
                <select
                  value={localSettings.apiProvider || 'qwen3.5-plus'}
                  onChange={e => setLocalSettings(prev => ({ ...prev, apiProvider: e.target.value }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                >
                  {TEXT_MODELS.map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
              </div>

              {/* OpenClaw Token */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  OpenClaw Token (optional)
                </label>
                <input
                  type="password"
                  value={localSettings.apiKey || ''}
                  onChange={e => setLocalSettings(prev => ({ ...prev, apiKey: e.target.value }))}
                  placeholder="Leave empty for local OpenClaw"
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200"
                />
                <p className="text-xs text-neutral-500 mt-1">Only needed for remote OpenClaw</p>
              </div>
            </>
          )}

          {activeTab === 'agent' && (
            <>
              {/* Agent Mode */}
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

              {/* Auto Connect */}
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

              {/* Autonomous Level */}
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

              {/* Agent Personality */}
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

              {/* Objectives */}
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
              {/* OpenClaw MCP - Auto-detected */}
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
