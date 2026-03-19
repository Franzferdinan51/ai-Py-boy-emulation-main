/**
 * Settings Modal - Simplified Settings Flow
 * Flow: Backend → OpenClaw → Model → Done
 */

import React, { useState, useEffect, useCallback } from 'react';

interface Settings {
  backendUrl: string;
  openclawMcpEndpoint: string;
  visionModel: string;
  autonomousLevel: string;
  agentPersonality: string;
  agentObjectives: string;
  agentMode: boolean;
  apiKey: string;
}

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: Settings;
  onSave: (newSettings: Settings) => void;
}

const DEFAULT_BACKEND = 'http://localhost:5002';
const DEFAULT_OPENCLAW = 'http://localhost:18789';

// Simplified model list - Bailian first (free)
const VISION_MODELS = [
  { id: 'kimi-k2.5', name: 'kimi-k2.5 (Bailian - Free)', badge: '🆓' },
  { id: 'qwen-vl-plus', name: 'qwen-vl-plus (Bailian - Free)', badge: '🆓' },
];

const TEXT_MODELS = [
  { id: 'MiniMax-M2.5', name: 'MiniMax-M2.5 (Bailian - Free)', badge: '🆓' },
  { id: 'qwen3.5-plus', name: 'qwen3.5-plus (Bailian)', badge: '🆓' },
  { id: 'glm-5', name: 'glm-5 (Bailian)', badge: '🆓' },
];

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [testingBackend, setTestingBackend] = useState(false);
  const [testingOpenClaw, setTestingOpenClaw] = useState(false);
  const [backendStatus, setBackendStatus] = useState<{ok: boolean; msg: string} | null>(null);
  const [openclawStatus, setOpenclawStatus] = useState<{ok: boolean; msg: string} | null>(null);

  // Initialize on open
  useEffect(() => {
    if (isOpen) {
      setLocalSettings(prev => ({
        ...prev,
        backendUrl: prev.backendUrl || DEFAULT_BACKEND,
        openclawMcpEndpoint: prev.openclawMcpEndpoint || DEFAULT_OPENCLAW,
        visionModel: prev.visionModel || 'kimi-k2.5',
        autonomousLevel: prev.autonomousLevel || 'moderate',
        agentPersonality: prev.agentPersonality || 'strategic',
        agentObjectives: prev.agentObjectives || 'Complete Pokemon Red',
      }));
    }
  }, [isOpen]);

  // Reset when settings change externally
  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  // Test backend connection
  const testBackend = useCallback(async () => {
    setTestingBackend(true);
    setBackendStatus(null);
    try {
      const res = await fetch(`${localSettings.backendUrl}/health`);
      if (res.ok) {
        setBackendStatus({ ok: true, msg: 'Connected' });
      } else {
        setBackendStatus({ ok: false, msg: `Error ${res.status}` });
      }
    } catch {
      setBackendStatus({ ok: false, msg: 'Connection failed' });
    }
    setTestingBackend(false);
  }, [localSettings.backendUrl]);

  // Test OpenClaw connection
  const testOpenClaw = useCallback(async () => {
    setTestingOpenClaw(true);
    setOpenclawStatus(null);
    try {
      const res = await fetch(`${localSettings.openclawMcpEndpoint}/health`);
      if (res.ok) {
        setOpenclawStatus({ ok: true, msg: 'Connected' });
      } else {
        setOpenclawStatus({ ok: false, msg: `Error ${res.status}` });
      }
    } catch {
      setOpenclawStatus({ ok: false, msg: 'Not running' });
    }
    setTestingOpenClaw(false);
  }, [localSettings.openclawMcpEndpoint]);

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  // Close on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className="relative bg-neutral-900 border border-neutral-700 rounded-xl w-full max-w-lg mx-4 max-h-[85vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h2 className="text-lg font-semibold text-neutral-200">Settings</h2>
          <button onClick={onClose} className="p-1 text-neutral-400 hover:text-white">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content - Simple single-column flow */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          
          {/* Step 1: Backend */}
          <section>
            <h3 className="text-sm font-medium text-cyan-400 mb-2 flex items-center gap-2">
              <span className="w-5 h-5 rounded bg-cyan-900/50 flex items-center justify-center text-xs">1</span>
              Game Backend
            </h3>
            <p className="text-xs text-neutral-500 mb-2">PyBoy MCP server URL</p>
            <div className="flex gap-2">
              <input
                type="text"
                value={localSettings.backendUrl}
                onChange={e => setLocalSettings(prev => ({ ...prev, backendUrl: e.target.value }))}
                placeholder="http://localhost:5002"
                className="flex-1 px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-sm text-neutral-200"
              />
              <button
                onClick={testBackend}
                disabled={testingBackend}
                className="px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg text-sm text-neutral-300"
              >
                {testingBackend ? '...' : 'Test'}
              </button>
            </div>
            {backendStatus && (
              <p className={`text-xs mt-1 ${backendStatus.ok ? 'text-green-400' : 'text-red-400'}`}>
                {backendStatus.ok ? '✓' : '✗'} {backendStatus.msg}
              </p>
            )}
          </section>

          {/* Step 2: OpenClaw */}
          <section>
            <h3 className="text-sm font-medium text-purple-400 mb-2 flex items-center gap-2">
              <span className="w-5 h-5 rounded bg-purple-900/50 flex items-center justify-center text-xs">2</span>
              OpenClaw Agent
            </h3>
            <p className="text-xs text-neutral-500 mb-2">OpenClaw Gateway MCP endpoint</p>
            <div className="flex gap-2">
              <input
                type="text"
                value={localSettings.openclawMcpEndpoint}
                onChange={e => setLocalSettings(prev => ({ ...prev, openclawMcpEndpoint: e.target.value }))}
                placeholder="http://localhost:18789"
                className="flex-1 px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-sm text-neutral-200"
              />
              <button
                onClick={testOpenClaw}
                disabled={testingOpenClaw}
                className="px-3 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg text-sm text-neutral-300"
              >
                {testingOpenClaw ? '...' : 'Test'}
              </button>
            </div>
            {openclawStatus && (
              <p className={`text-xs mt-1 ${openclawStatus.ok ? 'text-green-400' : 'text-red-400'}`}>
                {openclawStatus.ok ? '✓' : '✗'} {openclawStatus.msg}
              </p>
            )}
          </section>

          {/* Step 3: Model */}
          <section>
            <h3 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-2">
              <span className="w-5 h-5 rounded bg-green-900/50 flex items-center justify-center text-xs">3</span>
              Vision Model
            </h3>
            <p className="text-xs text-neutral-500 mb-2">AI model for screen analysis</p>
            <select
              value={localSettings.visionModel}
              onChange={e => setLocalSettings(prev => ({ ...prev, visionModel: e.target.value }))}
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-sm text-neutral-200"
            >
              {VISION_MODELS.map(m => (
                <option key={m.id} value={m.id}>{m.badge} {m.name}</option>
              ))}
            </select>
          </section>

          {/* Step 4: Agent Settings */}
          <section className="border-t border-neutral-700 pt-4">
            <h3 className="text-sm font-medium text-neutral-300 mb-3">Agent Behavior</h3>
            
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Autonomy</label>
                <select
                  value={localSettings.autonomousLevel}
                  onChange={e => setLocalSettings(prev => ({ ...prev, autonomousLevel: e.target.value }))}
                  className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-sm text-neutral-300"
                >
                  <option value="passive">Passive</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>
              
              <div>
                <label className="block text-xs text-neutral-500 mb-1">Personality</label>
                <select
                  value={localSettings.agentPersonality}
                  onChange={e => setLocalSettings(prev => ({ ...prev, agentPersonality: e.target.value }))}
                  className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-sm text-neutral-300"
                >
                  <option value="strategic">Strategic</option>
                  <option value="casual">Casual</option>
                  <option value="speedrun">Speedrun</option>
                </select>
              </div>
            </div>

            <div className="mt-3">
              <label className="block text-xs text-neutral-500 mb-1">Objectives</label>
              <input
                type="text"
                value={localSettings.agentObjectives}
                onChange={e => setLocalSettings(prev => ({ ...prev, agentObjectives: e.target.value }))}
                placeholder="Complete Pokemon Red"
                className="w-full px-3 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-sm text-neutral-200"
              />
            </div>

            <div className="mt-3 flex items-center justify-between">
              <span className="text-sm text-neutral-300">Agent Mode</span>
              <button
                onClick={() => setLocalSettings(prev => ({ ...prev, agentMode: !prev.agentMode }))}
                className={`relative w-10 h-5 rounded-full transition-colors ${localSettings.agentMode ? 'bg-green-600' : 'bg-neutral-600'}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${localSettings.agentMode ? 'left-5' : 'left-0.5'}`} />
              </button>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 p-4 border-t border-neutral-700">
          <button onClick={onClose} className="px-4 py-2 text-sm text-neutral-400 hover:text-white">
            Cancel
          </button>
          <button onClick={handleSave} className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm">
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;