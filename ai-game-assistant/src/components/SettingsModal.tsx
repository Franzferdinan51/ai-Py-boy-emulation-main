import React, { useEffect, useState } from 'react';
import type { AppSettings } from '../../services/apiService';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSave: (settings: AppSettings) => void;
}

interface ProbeStatus {
  ok: boolean;
  message: string;
}

const VISION_MODELS: Array<{ value: AppSettings['visionModel']; label: string }> = [
  { value: 'kimi-k2.5', label: 'kimi-k2.5' },
  { value: 'qwen-vl-plus', label: 'qwen-vl-plus' },
];

const PERSONALITIES: Array<{ value: AppSettings['agentPersonality']; label: string }> = [
  { value: 'strategic', label: 'Strategic' },
  { value: 'casual', label: 'Casual' },
  { value: 'speedrun', label: 'Speedrun' },
  { value: 'explorer', label: 'Explorer' },
];

const AUTONOMY_LEVELS: Array<{ value: AppSettings['autonomousLevel']; label: string }> = [
  { value: 'passive', label: 'Passive' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'aggressive', label: 'Aggressive' },
];

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [backendStatus, setBackendStatus] = useState<ProbeStatus | null>(null);
  const [openClawStatus, setOpenClawStatus] = useState<ProbeStatus | null>(null);
  const [testingBackend, setTestingBackend] = useState(false);
  const [testingOpenClaw, setTestingOpenClaw] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setDraft(settings);
      setBackendStatus(null);
      setOpenClawStatus(null);
    }
  }, [isOpen, settings]);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  const update = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  const testBackend = async () => {
    setTestingBackend(true);
    setBackendStatus(null);

    try {
      const response = await fetch(`${draft.backendUrl.replace(/\/+$/, '')}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      setBackendStatus({
        ok: true,
        message: payload.status === 'healthy' ? 'Backend healthy' : 'Backend responded',
      });
    } catch (error) {
      setBackendStatus({
        ok: false,
        message: error instanceof Error ? error.message : 'Connection failed',
      });
    } finally {
      setTestingBackend(false);
    }
  };

  const testOpenClaw = async () => {
    setTestingOpenClaw(true);
    setOpenClawStatus(null);

    try {
      const backendUrl = draft.backendUrl.replace(/\/+$/, '');
      const endpoint = encodeURIComponent(draft.openclawMcpEndpoint);
      const response = await fetch(`${backendUrl}/api/openclaw/health?endpoint=${endpoint}`);
      const payload = await response.json();

      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      setOpenClawStatus({
        ok: true,
        message: payload.service_status || 'OpenClaw reachable',
      });
    } catch (error) {
      setOpenClawStatus({
        ok: false,
        message: error instanceof Error ? error.message : 'Health check failed',
      });
    } finally {
      setTestingOpenClaw(false);
    }
  };

  const handleSave = () => {
    onSave(draft);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 px-4 backdrop-blur-sm">
      <div
        className="absolute inset-0"
        onClick={onClose}
      />

      <div className="relative z-10 flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-cyan-900/60 bg-neutral-950 shadow-2xl shadow-black/60">
        <div className="flex items-center justify-between border-b border-neutral-800 px-6 py-5">
          <div>
            <h2 className="text-xl font-semibold text-white">WebUI Settings</h2>
            <p className="mt-1 text-sm text-neutral-400">
              Keep this focused on connection details and default OpenClaw behavior.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-neutral-700 px-3 py-1.5 text-sm text-neutral-400 transition hover:border-neutral-500 hover:text-white"
          >
            Close
          </button>
        </div>

        <div className="space-y-6 overflow-y-auto px-6 py-6">
          <section className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5">
            <div className="mb-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Connection</h3>
              <p className="mt-1 text-sm text-neutral-400">Backend and OpenClaw endpoints are the only required integration points.</p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">Backend URL</label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={draft.backendUrl}
                    onChange={(event) => update('backendUrl', event.target.value)}
                    className="flex-1 rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                    placeholder="http://localhost:5002"
                  />
                  <button
                    onClick={testBackend}
                    disabled={testingBackend}
                    className="rounded-xl border border-neutral-700 px-4 text-sm text-neutral-200 transition hover:border-cyan-500 hover:text-white disabled:cursor-wait disabled:opacity-60"
                  >
                    {testingBackend ? 'Testing...' : 'Test'}
                  </button>
                </div>
                {backendStatus && (
                  <p className={`mt-2 text-xs ${backendStatus.ok ? 'text-green-400' : 'text-red-400'}`}>
                    {backendStatus.message}
                  </p>
                )}
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">OpenClaw Endpoint</label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={draft.openclawMcpEndpoint}
                    onChange={(event) => update('openclawMcpEndpoint', event.target.value)}
                    className="flex-1 rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                    placeholder="http://localhost:18789"
                  />
                  <button
                    onClick={testOpenClaw}
                    disabled={testingOpenClaw}
                    className="rounded-xl border border-neutral-700 px-4 text-sm text-neutral-200 transition hover:border-cyan-500 hover:text-white disabled:cursor-wait disabled:opacity-60"
                  >
                    {testingOpenClaw ? 'Testing...' : 'Test'}
                  </button>
                </div>
                {openClawStatus && (
                  <p className={`mt-2 text-xs ${openClawStatus.ok ? 'text-green-400' : 'text-red-400'}`}>
                    {openClawStatus.message}
                  </p>
                )}
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5">
            <div className="mb-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Startup Defaults</h3>
              <p className="mt-1 text-sm text-neutral-400">These are applied when the WebUI connects or when you resume OpenClaw control.</p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">Emulator Type</label>
                <select
                  value={draft.emulatorType}
                  onChange={(event) => update('emulatorType', event.target.value as AppSettings['emulatorType'])}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                >
                  <option value="gb">Game Boy / Game Boy Color</option>
                  <option value="gba">Game Boy Advance</option>
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">Vision Model</label>
                <select
                  value={draft.visionModel}
                  onChange={(event) => update('visionModel', event.target.value as AppSettings['visionModel'])}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                >
                  {VISION_MODELS.map((model) => (
                    <option key={model.value} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">Autonomy</label>
                <select
                  value={draft.autonomousLevel}
                  onChange={(event) => update('autonomousLevel', event.target.value as AppSettings['autonomousLevel'])}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                >
                  {AUTONOMY_LEVELS.map((level) => (
                    <option key={level.value} value={level.value}>
                      {level.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-neutral-200">Personality</label>
                <select
                  value={draft.agentPersonality}
                  onChange={(event) => update('agentPersonality', event.target.value as AppSettings['agentPersonality'])}
                  className="w-full rounded-xl border border-neutral-700 bg-neutral-950 px-3 py-2.5 text-sm text-white outline-none transition focus:border-cyan-500"
                >
                  {PERSONALITIES.map((personality) => (
                    <option key={personality.value} value={personality.value}>
                      {personality.label}
                    </option>
                  ))}
                </select>
              </div>

              <label className="flex items-center justify-between rounded-2xl border border-neutral-800 bg-neutral-950/70 px-4 py-3 text-sm text-neutral-200">
                Auto-connect on load
                <input
                  type="checkbox"
                  checked={draft.autoConnect}
                  onChange={(event) => update('autoConnect', event.target.checked)}
                  className="h-4 w-4 rounded border-neutral-700 bg-neutral-900 text-cyan-500"
                />
              </label>

              <label className="flex items-center justify-between rounded-2xl border border-neutral-800 bg-neutral-950/70 px-4 py-3 text-sm text-neutral-200">
                Launch emulator UI with ROM
                <input
                  type="checkbox"
                  checked={draft.launchUiOnRomLoad}
                  onChange={(event) => update('launchUiOnRomLoad', event.target.checked)}
                  className="h-4 w-4 rounded border-neutral-700 bg-neutral-900 text-cyan-500"
                />
              </label>
            </div>
          </section>
        </div>

        <div className="flex items-center justify-between border-t border-neutral-800 px-6 py-5">
          <p className="text-xs text-neutral-500">
            Objectives and live control stay in the main OpenClaw panel so runtime intent is visible.
          </p>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="rounded-xl border border-neutral-700 px-4 py-2 text-sm text-neutral-300 transition hover:border-neutral-500 hover:text-white"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="rounded-xl bg-cyan-500 px-4 py-2 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-400"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
