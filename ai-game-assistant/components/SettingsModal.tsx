import React, { useState, useEffect } from 'react';
import type { AppSettings } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentSettings: AppSettings;
  onSave: (newSettings: AppSettings) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, currentSettings, onSave }) => {
  const [settings, setSettings] = useState<AppSettings>(currentSettings);
  const [models, setModels] = useState<string[]>([]);
  const [modelSearch, setModelSearch] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  const [lmStudioModels, setLmStudioModels] = useState<string[]>([]);

  useEffect(() => {
    setSettings(currentSettings);
  }, [currentSettings, isOpen]);

  // Load recent models from localStorage when component mounts
  useEffect(() => {
    if (isOpen) {
      const recentModels = localStorage.getItem('lmStudioRecentModels');
      if (recentModels) {
        try {
          setLmStudioModels(JSON.parse(recentModels));
        } catch (e) {
          console.error('Error parsing recent models:', e);
        }
      }
    }
  }, [isOpen]);

  useEffect(() => {
    const fetchModels = async () => {
      if (!settings.apiProvider) return;
      
      // Don't fetch for openrouter as it requires manual input
      if (settings.apiProvider === 'openrouter') {
        setModels([]);
        return;
      }

      setLoadingModels(true);
      try {
        const response = await fetch(`http://localhost:5000/api/models?provider=${settings.apiProvider}`);
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        setModels(data.models || []);
        
        // Special handling for LM Studio to detect models
        if (settings.apiProvider === 'openai-compatible') {
          setLmStudioModels(data.models || []);
          // Save models to localStorage
          if (data.models && data.models.length > 0) {
            localStorage.setItem('lmStudioModels', JSON.stringify(data.models));
          }
        }
      } catch (error) {
        console.error('Error fetching models:', error);
        setModels([]);
        // For LM Studio, try to load from localStorage as fallback
        if (settings.apiProvider === 'openai-compatible') {
          const savedModels = localStorage.getItem('lmStudioModels');
          if (savedModels) {
            try {
              const parsedModels = JSON.parse(savedModels);
              setLmStudioModels(parsedModels);
            } catch (e) {
              console.error('Error parsing saved models:', e);
            }
          }
        }
      }
      setLoadingModels(false);
    };

    if (isOpen) {
      fetchModels();
    }
  }, [settings.apiProvider, isOpen]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  const handleSave = () => {
    onSave(settings);
  };

  const handleIntervalChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Math.max(1000, Math.min(30000, Number(e.target.value)));
    setSettings(prev => ({ ...prev, aiActionInterval: value }));
  };
  
  if (!isOpen) {
    return null;
  }

  return (
    <div 
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
    >
      <div 
        className="bg-neutral-900 rounded-lg shadow-2xl shadow-black/50 border border-neutral-800 w-full max-w-md m-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-neutral-800">
          <h2 id="settings-title" className="text-xl font-bold font-display">Settings</h2>
          <button onClick={onClose} className="text-neutral-500 hover:text-white" aria-label="Close settings">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
          <div>
            <label htmlFor="ai-interval" className="block text-sm font-medium text-neutral-300 mb-2">
              AI Action Interval (ms)
            </label>
            <input 
              type="number" 
              id="ai-interval"
              value={settings.aiActionInterval}
              onChange={handleIntervalChange}
              min="1000"
              max="30000"
              step="500"
              className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
            />
            <p className="text-xs text-neutral-500 mt-2">Time between each AI move. Lower is faster. (1000-30000ms)</p>
          </div>
          
          <div className="border-t border-neutral-800 pt-6 space-y-4">
            <h3 className="text-md font-semibold text-neutral-300">AI Provider</h3>
            <div>
              <label htmlFor="api-provider" className="block text-sm font-medium text-neutral-300 mb-2">
                API Provider
              </label>
              <select
                id="api-provider"
                value={settings.apiProvider}
                onChange={(e) => {
                  const provider = e.target.value as AppSettings['apiProvider'];
                  setSettings(prev => ({
                    ...prev,
                    apiProvider: provider,
                    model: '', // Reset model when provider changes
                  }));
                }}
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              >
                <option value="openclaw">🦆 OpenClaw Agent (Default)</option>
                <option value="gemini">Google Gemini</option>
                <option value="openrouter">OpenRouter</option>
                <option value="openai-compatible">OpenAI Compatible (LM Studio, etc)</option>
                <option value="nvidia">NVIDIA</option>
              </select>
            </div>

            {/* Endpoint field for relevant providers */}
            {(settings.apiProvider === 'openai-compatible' || settings.apiProvider === 'nvidia') && (
              <div>
                <label htmlFor="api-endpoint" className="block text-sm font-medium text-neutral-300 mb-2">
                  API Endpoint
                </label>
                <input
                  type="text"
                  id="api-endpoint"
                  value={settings.apiEndpoint || ''}
                  onChange={(e) => setSettings(prev => ({...prev, apiEndpoint: e.target.value}))}
                  placeholder={
                    settings.apiProvider === 'openai-compatible' 
                      ? "e.g., http://localhost:1234/v1" 
                      : "e.g., https://integrate.api.nvidia.com/v1"
                  }
                  className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
                />
                {settings.apiProvider === 'openai-compatible' && (
                  <p className="text-xs text-neutral-500 mt-2">For LM Studio, use the address from the server logs.</p>
                )}
                 {settings.apiProvider === 'nvidia' && (
                  <p className="text-xs text-neutral-500 mt-2">NVIDIA API Catalog endpoint. Defaults to the official one.</p>
                )}
              </div>
            )}

            {/* Model Selection */}
            {settings.apiProvider && (
              <div className="space-y-2">
                <label htmlFor="model-input" className="block text-sm font-medium text-neutral-300">
                  Model
                </label>
                
                {/* Search and Dropdown for Gemini */}
                {settings.apiProvider === 'gemini' && (
                  <div className="space-y-2">
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
                    />
                    <select
                      id="model-select"
                      value={settings.model || ''}
                      onChange={(e) => setSettings(prev => ({...prev, model: e.target.value}))}
                      disabled={loadingModels || models.length === 0}
                      className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none disabled:opacity-50"
                    >
                      <option value="" disabled>{loadingModels ? 'Loading...' : 'Select a model'}</option>
                      {models
                        .filter(m => m.toLowerCase().includes(modelSearch.toLowerCase()))
                        .map(model => (
                          <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                  </div>
                )}

                {/* Search and Dropdown for OpenAI Compatible (LM Studio) with Auto-detect */}
                {settings.apiProvider === 'openai-compatible' && (
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <input
                        type="text"
                        placeholder="Search models or enter custom model..."
                        value={settings.model || ''}
                        onChange={(e) => setSettings(prev => ({...prev, model: e.target.value}))}
                        className="flex-grow p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
                      />
                      <button
                        onClick={async () => {
                          setLoadingModels(true);
                          try {
                            // First try to get models from the server
                            const response = await fetch(`http://localhost:5000/api/models?provider=openai-compatible`);
                            if (!response.ok) {
                              throw new Error('Failed to fetch models');
                            }
                            const data = await response.json();
                            setModels(data.models || []);
                            setLmStudioModels(data.models || []);
                            // Save models to localStorage
                            if (data.models && data.models.length > 0) {
                              localStorage.setItem('lmStudioModels', JSON.stringify(data.models));
                            }
                          } catch (error) {
                            console.error('Error fetching models:', error);
                            // If we can't fetch models, try to get them from localStorage as a fallback
                            const savedModels = localStorage.getItem('lmStudioModels');
                            if (savedModels) {
                              try {
                                const parsedModels = JSON.parse(savedModels);
                                setModels(parsedModels);
                                setLmStudioModels(parsedModels);
                              } catch (e) {
                                console.error('Error parsing saved models:', e);
                              }
                            }
                          }
                          setLoadingModels(false);
                        }}
                        disabled={loadingModels}
                        className="px-3 py-2 bg-neutral-700 hover:bg-neutral-600 text-white rounded-md disabled:opacity-50"
                      >
                        {loadingModels ? 'Detecting...' : 'Detect'}
                      </button>
                    </div>
                    
                    {/* Recent models and suggestions dropdown */}
                    {(lmStudioModels.length > 0 || localStorage.getItem('lmStudioRecentModels')) && (
                      <div className="max-h-40 overflow-y-auto bg-neutral-800 border border-neutral-700 rounded-md">
                        {/* Show recent models first */}
                        {(() => {
                          const recentModels = JSON.parse(localStorage.getItem('lmStudioRecentModels') || '[]');
                          return recentModels.map((model: string) => (
                            <div
                              key={`recent-${model}`}
                              className="p-2 hover:bg-neutral-700 cursor-pointer flex items-center"
                              onClick={() => {
                                setSettings(prev => ({...prev, model: model}));
                                // Move to top of recent list
                                const updatedRecent = [model, ...recentModels.filter((m: string) => m !== model)];
                                if (updatedRecent.length > 10) updatedRecent.pop();
                                localStorage.setItem('lmStudioRecentModels', JSON.stringify(updatedRecent));
                              }}
                            >
                              <span className="mr-2">🕒</span> {model}
                            </div>
                          ));
                        })()}
                        {/* Show detected models */}
                        {lmStudioModels
                          .filter(m => {
                            // Don't show models that are already in recent models
                            const recentModels = JSON.parse(localStorage.getItem('lmStudioRecentModels') || '[]');
                            return !recentModels.includes(m) && m.toLowerCase().includes((settings.model || '').toLowerCase());
                          })
                          .map(model => (
                            <div
                              key={model}
                              className="p-2 hover:bg-neutral-700 cursor-pointer"
                              onClick={() => {
                                setSettings(prev => ({...prev, model: model}));
                                // Add to recent models
                                const recentModels = JSON.parse(localStorage.getItem('lmStudioRecentModels') || '[]');
                                if (!recentModels.includes(model)) {
                                  recentModels.unshift(model);
                                  if (recentModels.length > 10) recentModels.pop();
                                  localStorage.setItem('lmStudioRecentModels', JSON.stringify(recentModels));
                                }
                              }}
                            >
                              {model}
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Search and Dropdown for NVIDIA */}
                {settings.apiProvider === 'nvidia' && (
                  <div className="space-y-2">
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
                    />
                    <select
                      id="model-select"
                      value={settings.model || ''}
                      onChange={(e) => setSettings(prev => ({...prev, model: e.target.value}))}
                      disabled={loadingModels || models.length === 0}
                      className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none disabled:opacity-50"
                    >
                      <option value="" disabled>{loadingModels ? 'Loading...' : 'Select a model'}</option>
                      {models
                        .filter(m => m.toLowerCase().includes(modelSearch.toLowerCase()))
                        .map(model => (
                          <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                  </div>
                )}

                {/* Manual input for OpenRouter */}
                {settings.apiProvider === 'openrouter' && (
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      id="model-input"
                      value={settings.model || ''}
                      onChange={(e) => setSettings(prev => ({...prev, model: e.target.value}))}
                      placeholder="vendor/model-name:version (e.g., qwen/qwen3-coder:free)"
                      className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
                    />
                  </div>
                )}

                <p className="text-xs text-neutral-500 mt-2">
                  {
                    settings.apiProvider === 'gemini' ? 'Select from available Gemini models' :
                    settings.apiProvider === 'openrouter' ? 'Enter model in format "vendor/model-name:version" (e.g., "qwen/qwen3-coder:free")' :
                    settings.apiProvider === 'nvidia' ? 'Select from available NVIDIA models' :
                    'For LM Studio, click "Detect" to find available models or enter model name manually'
                  }
                </p>
              </div>
            )}

            <div>
              <label htmlFor="api-key" className="block text-sm font-medium text-neutral-300 mb-2">
                API Key
              </label>
              <input
                type="password"
                id="api-key"
                value={settings.apiKey || ''}
                onChange={(e) => setSettings(prev => ({...prev, apiKey: e.target.value}))}
                placeholder="Required for most providers"
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              />
              <p className="text-xs text-neutral-500 mt-2">
                Your API key is stored locally in your browser. For LM Studio, this can often be left blank.
              </p>
            </div>
          </div>

          {/* OpenClaw Agent Settings Section */}
          <div className="border-t border-neutral-800 pt-6 space-y-4">
            <h3 className="text-md font-semibold text-neutral-300 flex items-center">
              🦆 OpenClaw Agent Settings
            </h3>
            
            {/* Agent Mode Toggle */}
            <div className="flex items-center justify-between p-3 bg-neutral-800/50 rounded-md border border-neutral-700">
              <div>
                <label htmlFor="agent-mode" className="block text-sm font-medium text-neutral-300">
                  🤖 Agent Mode
                </label>
                <p className="text-xs text-neutral-500">Enable AI agent to control the game</p>
              </div>
              <button
                id="agent-mode"
                onClick={() => setSettings(prev => ({ ...prev, agentMode: !prev.agentMode }))}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  settings.agentMode ? 'bg-green-500' : 'bg-neutral-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    settings.agentMode ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {/* MCP Endpoint */}
            <div>
              <label htmlFor="mcp-endpoint" className="block text-sm font-medium text-neutral-300 mb-2">
                🔗 MCP Server Endpoint
              </label>
              <input
                type="text"
                id="mcp-endpoint"
                value={settings.openclawMcpEndpoint || ''}
                onChange={(e) => setSettings(prev => ({ ...prev, openclawMcpEndpoint: e.target.value }))}
                placeholder="http://localhost:3000/mcp"
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              />
              <p className="text-xs text-neutral-500 mt-2">OpenClaw MCP server URL for agent communication</p>
            </div>

            {/* Vision Model Selection */}
            <div>
              <label htmlFor="vision-model" className="block text-sm font-medium text-neutral-300 mb-2">
                👁️ Vision Model
              </label>
              <select
                id="vision-model"
                value={settings.visionModel || 'kimi-k2.5'}
                onChange={(e) => setSettings(prev => ({ ...prev, visionModel: e.target.value as AppSettings['visionModel'] }))}
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              >
                <option value="kimi-k2.5">kimi-k2.5 (Bailian - FREE)</option>
                <option value="MiniMax-M2.5">MiniMax-M2.5 (Bailian - FREE)</option>
                <option value="glm-5">glm-5 (Bailian)</option>
              </select>
              <p className="text-xs text-neutral-500 mt-2">Model used for screen analysis and game state understanding</p>
            </div>

            {/* Autonomous Level */}
            <div>
              <label htmlFor="autonomous-level" className="block text-sm font-medium text-neutral-300 mb-2">
                ⚡ Autonomous Level
              </label>
              <select
                id="autonomous-level"
                value={settings.autonomousLevel || 'moderate'}
                onChange={(e) => setSettings(prev => ({ ...prev, autonomousLevel: e.target.value as AppSettings['autonomousLevel'] }))}
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              >
                <option value="passive">Passive - Only acts when requested</option>
                <option value="moderate">Moderate - Makes decisions with human oversight</option>
                <option value="aggressive">Aggressive - Full autonomous control</option>
              </select>
              <p className="text-xs text-neutral-500 mt-2">How independently the agent operates</p>
            </div>

            {/* Agent Personality */}
            <div>
              <label htmlFor="agent-personality" className="block text-sm font-medium text-neutral-300 mb-2">
                🎭 Agent Personality
              </label>
              <select
                id="agent-personality"
                value={settings.agentPersonality || 'strategic'}
                onChange={(e) => setSettings(prev => ({ ...prev, agentPersonality: e.target.value as AppSettings['agentPersonality'] }))}
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              >
                <option value="strategic">Strategic - Calculated, optimal moves</option>
                <option value="casual">Casual - Relaxed, exploration-focused</option>
                <option value="speedrun">Speedrun - Time-optimized strategies</option>
                <option value="explorer">Explorer - Discovery-focused, thorough</option>
              </select>
              <p className="text-xs text-neutral-500 mt-2">Defines the agent's decision-making approach</p>
            </div>

            {/* Agent Objectives */}
            <div>
              <label htmlFor="agent-objectives" className="block text-sm font-medium text-neutral-300 mb-2">
                🎯 Agent Objectives
              </label>
              <textarea
                id="agent-objectives"
                rows={3}
                value={settings.agentObjectives || ''}
                onChange={(e) => setSettings(prev => ({ ...prev, agentObjectives: e.target.value }))}
                placeholder="e.g., Complete Pokemon Red, defeat the Elite Four"
                className="w-full p-2 bg-neutral-800 border border-neutral-700 rounded-md text-sm placeholder:text-neutral-600 focus:ring-1 focus:ring-cyan-glow focus:border-cyan-glow outline-none"
              />
              <p className="text-xs text-neutral-500 mt-2">Current objectives for the autonomous agent</p>
            </div>
          </div>
        </div>

        <div className="flex justify-end p-4 bg-neutral-950/50 border-t border-neutral-800 rounded-b-lg">
          <button 
            onClick={handleSave}
            className="px-6 py-2 bg-cyan-glow text-neutral-950 font-bold rounded-md hover:bg-opacity-80 transition-all duration-200"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
