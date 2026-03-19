import React, { useState, useEffect, useCallback } from 'react';
import type { AppSettings } from '../../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentSettings: AppSettings;
  onSave: (newSettings: AppSettings) => void;
  className?: string;
}

// Common models by provider
const PROVIDER_MODELS: Record<string, string[]> = {
  openclaw: ['MiniMax-M2.5', 'kimi-k2.5', 'glm-5', 'qwen3.5-plus'],
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
  openrouter: ['openai/gpt-4o', 'google/gemini-pro', 'anthropic/claude-3-opus'],
  'openai-compatible': [], // Populated dynamically
  nvidia: ['nvidia/llama-3.1-nemotron-70b-instruct', 'nvidia/llama-3.1-8b-instruct'],
  gemini: ['gemini-2.0-flash', 'gemini-2.0-flash-exp'],
};

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  currentSettings,
  onSave,
  className = '',
}) => {
  const [settings, setSettings] = useState<AppSettings>(currentSettings);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<{success: boolean; message: string} | null>(null);
  const [activeTab, setActiveTab] = useState<'api' | 'agent' | 'advanced'>('api');

  useEffect(() => {
    setSettings(currentSettings);
  }, [currentSettings, isOpen]);

  // Fetch models when provider changes
  useEffect(() => {
    const fetchModels = async () => {
      if (!settings.apiProvider || settings.apiProvider === 'openrouter') {
        setAvailableModels(PROVIDER_MODELS[settings.apiProvider || ''] || []);
        return;
      }

      setLoadingModels(true);
      
      // For openai-compatible, try to fetch from backend
      if (settings.apiProvider === 'openai-compatible') {
        try {
          const response = await fetch(`${settings.apiEndpoint || 'http://localhost:5000'}/api/models`);
          if (response.ok) {
            const data = await response.json();
            setAvailableModels(data.models || []);
          } else {
            setAvailableModels([]);
          }
        } catch {
          // LM Studio or local server not available
          const saved = localStorage.getItem('lmStudioModels');
          if (saved) {
            try {
              setAvailableModels(JSON.parse(saved));
            } catch {
              setAvailableModels([]);
            }
          } else {
            setAvailableModels([]);
          }
        }
      } else {
        // Use predefined models
        setAvailableModels(PROVIDER_MODELS[settings.apiProvider] || []);
      }
      
      setLoadingModels(false);
    };

    if (isOpen) {
      fetchModels();
    }
  }, [settings.apiProvider, settings.apiEndpoint, isOpen]);

  // Test backend connection
  const testConnection = useCallback(async () => {
    setTestingConnection(true);
    setConnectionStatus(null);

    try {
      const baseUrl = settings.apiEndpoint || settings.backendUrl || 'http://localhost:5000';
      const response = await fetch(`${baseUrl}/api/screen`);
      
      if (response.ok || response.status === 400) {
        // 400 means ROM not loaded, but connection works
        setConnectionStatus({ success: true, message: 'Connected successfully!' });
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
  }, [settings.apiEndpoint, settings.backendUrl]);

  // Handle save
  const handleSave = () => {
    onSave(settings);
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
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className={`relative bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 max-h-[90vh] overflow-hidden flex flex-col ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h2 className="text-xl font-semibold text-neutral-200">⚙️ Settings</h2>
          <button
            onClick={onClose}
            className="p-1 text-neutral-400 hover:text-white transition-colors"
          >
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
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
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
              {/* API Provider */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  API Provider
                </label>
                <select
                  value={settings.apiProvider || ''}
                  onChange={e => setSettings(prev => ({ ...prev, apiProvider: e.target.value as AppSettings['apiProvider'] }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="">Select Provider</option>
                  <option value="openclaw">OpenClaw (Bailian)</option>
                  <option value="openai-compatible">OpenAI Compatible (LM Studio)</option>
                  <option value="openrouter">OpenRouter</option>
                  <option value="nvidia">NVIDIA</option>
                  <option value="gemini">Google Gemini</option>
                </select>
              </div>

              {/* API Endpoint */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  API Endpoint
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={settings.apiEndpoint || ''}
                    onChange={e => setSettings(prev => ({ ...prev, apiEndpoint: e.target.value }))}
                    placeholder="http://localhost:5000"
                    className="flex-1 px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
                  />
                  <button
                    onClick={testConnection}
                    disabled={testingConnection}
                    className="px-3 py-2 bg-neutral-700 hover:bg-neutral-600 text-neutral-300 rounded-lg text-sm disabled:opacity-50"
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

              {/* API Key */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  API Key
                </label>
                <input
                  type="password"
                  value={settings.apiKey || ''}
                  onChange={e => setSettings(prev => ({ ...prev, apiKey: e.target.value }))}
                  placeholder="Enter your API key"
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Model */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Model
                  {loadingModels && <span className="ml-2 text-neutral-500">Loading...</span>}
                </label>
                <select
                  value={settings.model || ''}
                  onChange={e => setSettings(prev => ({ ...prev, model: e.target.value }))}
                  disabled={loadingModels || availableModels.length === 0}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500 disabled:opacity-50"
                >
                  <option value="">Select Model</option>
                  {availableModels.map(model => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </div>
            </>
          )}

          {activeTab === 'agent' && (
            <>
              {/* Agent Mode */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-neutral-300">Agent Mode</label>
                  <p className="text-xs text-neutral-500">Let the AI control the game</p>
                </div>
                <button
                  onClick={() => setSettings(prev => ({ ...prev, agentMode: !prev.agentMode }))}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    settings.agentMode ? 'bg-green-500' : 'bg-neutral-600'
                  }`}
                >
                  <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                    settings.agentMode ? 'left-7' : 'left-1'
                  }`} />
                </button>
              </div>

              {/* Vision Model */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Vision Model
                </label>
                <select
                  value={settings.visionModel || 'kimi-k2.5'}
                  onChange={e => setSettings(prev => ({ ...prev, visionModel: e.target.value as AppSettings['visionModel'] }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="kimi-k2.5">kimi-k2.5 (Bailian)</option>
                  <option value="MiniMax-M2.5">MiniMax-M2.5 (Bailian)</option>
                  <option value="glm-5">glm-5 (Bailian)</option>
                </select>
              </div>

              {/* Autonomous Level */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Autonomous Level
                </label>
                <select
                  value={settings.autonomousLevel || 'moderate'}
                  onChange={e => setSettings(prev => ({ ...prev, autonomousLevel: e.target.value as AppSettings['autonomousLevel'] }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="passive">Passive - Asks before actions</option>
                  <option value="moderate">Moderate - Decides and acts</option>
                  <option value="aggressive">Aggressive - Fast decisions</option>
                </select>
              </div>

              {/* Agent Personality */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Agent Personality
                </label>
                <select
                  value={settings.agentPersonality || 'strategic'}
                  onChange={e => setSettings(prev => ({ ...prev, agentPersonality: e.target.value as AppSettings['agentPersonality'] }))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="strategic">Strategic - Plans ahead</option>
                  <option value="casual">Casual - Relaxed gameplay</option>
                  <option value="speedrun">Speedrun - Time-focused</option>
                  <option value="explorer">Explorer - Explores everything</option>
                </select>
              </div>

              {/* Objectives */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Current Objectives
                </label>
                <textarea
                  value={settings.agentObjectives || ''}
                  onChange={e => setSettings(prev => ({ ...prev, agentObjectives: e.target.value }))}
                  placeholder="Complete the game, Catch 'em all..."
                  rows={3}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500 resize-none"
                />
              </div>
            </>
          )}

          {activeTab === 'advanced' && (
            <>
              {/* OpenClaw MCP Endpoint */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  OpenClaw MCP Endpoint
                </label>
                <input
                  type="text"
                  value={settings.openclawMcpEndpoint || ''}
                  onChange={e => setSettings(prev => ({ ...prev, openclawMcpEndpoint: e.target.value }))}
                  placeholder="http://localhost:3000/mcp"
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* AI Action Interval */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  AI Action Interval (ms)
                </label>
                <input
                  type="number"
                  value={settings.aiActionInterval || 1000}
                  onChange={e => setSettings(prev => ({ ...prev, aiActionInterval: parseInt(e.target.value) || 1000 }))}
                  min={100}
                  max={10000}
                  step={100}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Backend URL (Legacy) */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-1">
                  Backend URL (Legacy)
                </label>
                <input
                  type="text"
                  value={settings.backendUrl || ''}
                  onChange={e => setSettings(prev => ({ ...prev, backendUrl: e.target.value }))}
                  placeholder="http://localhost:5000"
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
                />
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-4 border-t border-neutral-700 bg-neutral-800/50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-neutral-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;