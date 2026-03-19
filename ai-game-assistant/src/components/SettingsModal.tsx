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

const VISION_MODELS: Array<{ value: AppSettings['visionModel']; label: string; note: string }> = [
  { value: 'kimi-k2.5', label: 'Kimi K2.5', note: 'Recommended default for OpenClaw vision.' },
  { value: 'qwen-vl-plus', label: 'Qwen VL Plus', note: 'Alternative vision profile for inspection-heavy runs.' },
];

const AI_PROVIDERS: Array<{ value: NonNullable<AppSettings['aiProvider']>; label: string; note: string }> = [
  { value: 'openclaw', label: 'OpenClaw MCP', note: 'Local OpenClaw integration (recommended).' },
  { value: 'lmstudio', label: 'LM Studio', note: 'Local models with custom endpoint support.' },
  { value: 'gemini', label: 'Google Gemini', note: 'Google AI models.' },
  { value: 'openrouter', label: 'OpenRouter', note: 'Multi-provider API gateway.' },
  { value: 'openai-compatible', label: 'OpenAI Compatible', note: 'Any OpenAI-compatible endpoint.' },
  { value: 'nvidia', label: 'NVIDIA NIM', note: 'NVIDIA AI models.' },
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

const normalizeProbeMessage = (error: unknown, fallback404: string) => {
  const message = error instanceof Error ? error.message : 'Connection failed';

  if (message.includes('404')) {
    return fallback404;
  }

  if (message.includes('429')) {
    return 'Rate limited. Wait a moment and try again.';
  }

  return message;
};

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [draft, setDraft] = useState<AppSettings>(settings);
  const [backendStatus, setBackendStatus] = useState<ProbeStatus | null>(null);
  const [openClawStatus, setOpenClawStatus] = useState<ProbeStatus | null>(null);
  const [lmStudioStatus, setLmStudioStatus] = useState<ProbeStatus | null>(null);
  const [testingBackend, setTestingBackend] = useState(false);
  const [testingOpenClaw, setTestingOpenClaw] = useState(false);
  const [testingLmStudio, setTestingLmStudio] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setDraft(settings);
      setBackendStatus(null);
      setOpenClawStatus(null);
      setLmStudioStatus(null);
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

  const activeVisionModel = VISION_MODELS.find((model) => model.value === draft.visionModel);

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
        message: normalizeProbeMessage(error, 'Backend responded, but /health is not exposed here.'),
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
        message: normalizeProbeMessage(error, 'Backend is reachable, but the OpenClaw health proxy is not exposed.'),
      });
    } finally {
      setTestingOpenClaw(false);
    }
  };

  const testLmStudio = async () => {
    setTestingLmStudio(true);
    setLmStudioStatus(null);

    try {
      const lmUrl = (draft.lmStudioUrl || 'http://localhost:1234/v1').replace(/\/+$/, '');
      // Test by fetching models list
      const response = await fetch(`${lmUrl}/models`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      const modelCount = payload.data?.length || 0;
      
      setLmStudioStatus({
        ok: true,
        message: `LM Studio connected - ${modelCount} model${modelCount !== 1 ? 's' : ''} available`,
      });
    } catch (error) {
      setLmStudioStatus({
        ok: false,
        message: normalizeProbeMessage(error, 'LM Studio not reachable at specified URL'),
      });
    } finally {
      setTestingLmStudio(false);
    }
  };

  const handleSave = () => {
    onSave(draft);
    onClose();
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal-scrim" onClick={onClose} />

      <div className="modal modal--settings" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <div className="modal__header">
          <div>
            <span className="modal__eyebrow">Session Setup</span>
            <h2 id="settings-title">Connection and OpenClaw defaults</h2>
            <p className="modal__subtitle">
              Keep setup here focused on endpoints and orchestration defaults. Gameplay chrome stays on the handheld.
            </p>
          </div>
          <button type="button" onClick={onClose} className="action-button action-button--ghost">
            Close
          </button>
        </div>

        <div className="modal__body">
          <section className="modal-section">
            <div className="modal-section__header">
              <span className="modal-section__eyebrow">Connection</span>
              <h3>Backend and OpenClaw links</h3>
            </div>

            <div className="settings-grid">
              <label className="form-field">
                <span className="form-field__label">Backend URL</span>
                <div className="field-row">
                  <input
                    type="text"
                    value={draft.backendUrl}
                    onChange={(event) => update('backendUrl', event.target.value)}
                    className="text-input"
                    placeholder="http://localhost:5002"
                  />
                  <button
                    type="button"
                    onClick={testBackend}
                    disabled={testingBackend}
                    className="field-button"
                  >
                    {testingBackend ? 'Testing...' : 'Test'}
                  </button>
                </div>
                <span className="form-field__help">Port `5002` matches the current backend runtime default.</span>
                {backendStatus && (
                  <span className={backendStatus.ok ? 'probe-badge probe-badge--ok' : 'probe-badge probe-badge--error'}>
                    {backendStatus.message}
                  </span>
                )}
              </label>

              <label className="form-field">
                <span className="form-field__label">OpenClaw endpoint</span>
                <div className="field-row">
                  <input
                    type="text"
                    value={draft.openclawMcpEndpoint}
                    onChange={(event) => update('openclawMcpEndpoint', event.target.value)}
                    className="text-input"
                    placeholder="http://localhost:18789"
                  />
                  <button
                    type="button"
                    onClick={testOpenClaw}
                    disabled={testingOpenClaw}
                    className="field-button"
                  >
                    {testingOpenClaw ? 'Testing...' : 'Test'}
                  </button>
                </div>
                <span className="form-field__help">This is the MCP endpoint the backend uses to reach OpenClaw.</span>
                {openClawStatus && (
                  <span className={openClawStatus.ok ? 'probe-badge probe-badge--ok' : 'probe-badge probe-badge--error'}>
                    {openClawStatus.message}
                  </span>
                )}
              </label>
            </div>
          </section>

          <section className="modal-section">
            <div className="modal-section__header">
              <span className="modal-section__eyebrow">AI Provider</span>
              <h3>Local models and LM Studio</h3>
            </div>

            <div className="settings-grid">
              <label className="form-field">
                <span className="form-field__label">AI Provider</span>
                <select
                  value={draft.aiProvider || 'openclaw'}
                  onChange={(event) => update('aiProvider', event.target.value as AppSettings['aiProvider'])}
                  className="select-input"
                >
                  {AI_PROVIDERS.map((provider) => (
                    <option key={provider.value} value={provider.value}>
                      {provider.label}
                    </option>
                  ))}
                </select>
                <span className="form-field__help">
                  {AI_PROVIDERS.find(p => p.value === (draft.aiProvider || 'openclaw'))?.note}
                </span>
              </label>
            </div>

            {(draft.aiProvider === 'lmstudio' || draft.aiProvider === 'openai-compatible') && (
              <>
                <div className="modal-section__divider" />
                
                <div className="settings-grid">
                  <label className="form-field">
                    <span className="form-field__label">Custom Endpoint URL</span>
                    <div className="field-row">
                      <input
                        type="text"
                        value={draft.lmStudioUrl || ''}
                        onChange={(event) => update('lmStudioUrl', event.target.value)}
                        className="text-input"
                        placeholder="http://localhost:1234/v1"
                      />
                      <button
                        type="button"
                        onClick={testLmStudio}
                        disabled={testingLmStudio}
                        className="field-button"
                      >
                        {testingLmStudio ? 'Testing...' : 'Test'}
                      </button>
                    </div>
                    <span className="form-field__help">
                      LM Studio default: http://localhost:1234/v1. Must be OpenAI-compatible.
                    </span>
                    {lmStudioStatus && (
                      <span className={lmStudioStatus.ok ? 'probe-badge probe-badge--ok' : 'probe-badge probe-badge--error'}>
                        {lmStudioStatus.message}
                      </span>
                    )}
                  </label>
                </div>

                <div className="settings-grid settings-grid--two-up">
                  <label className="form-field">
                    <span className="form-field__label">Thinking Model</span>
                    <input
                      type="text"
                      value={draft.lmStudioThinkingModel || ''}
                      onChange={(event) => update('lmStudioThinkingModel', event.target.value)}
                      className="text-input"
                      placeholder="qwen3.5-35b-a3b"
                    />
                    <span className="form-field__help">
                      Model for text reasoning and decision-making. Leave empty for auto-detect.
                    </span>
                  </label>

                  <label className="form-field">
                    <span className="form-field__label">Vision Model</span>
                    <input
                      type="text"
                      value={draft.lmStudioVisionModel || ''}
                      onChange={(event) => update('lmStudioVisionModel', event.target.value)}
                      className="text-input"
                      placeholder="qwen3-vl-8b"
                    />
                    <span className="form-field__help">
                      Model for screen/image analysis. Leave empty for auto-detect.
                    </span>
                  </label>
                </div>
              </>
            )}
          </section>

          <section className="modal-section">
            <div className="modal-section__header">
              <span className="modal-section__eyebrow">Defaults</span>
              <h3>Automation profile</h3>
            </div>

            <div className="settings-grid settings-grid--two-up">
              <label className="form-field">
                <span className="form-field__label">Default emulator</span>
                <select
                  value={draft.emulatorType}
                  onChange={(event) => update('emulatorType', event.target.value as AppSettings['emulatorType'])}
                  className="select-input"
                >
                  <option value="gb">Game Boy / Game Boy Color</option>
                  <option value="gba">Game Boy Advance</option>
                </select>
              </label>

              <label className="form-field">
                <span className="form-field__label">Vision profile</span>
                <select
                  value={draft.visionModel}
                  onChange={(event) => update('visionModel', event.target.value as AppSettings['visionModel'])}
                  className="select-input"
                >
                  {VISION_MODELS.map((model) => (
                    <option key={model.value} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </select>
                <span className="form-field__help">{activeVisionModel?.note}</span>
              </label>

              <label className="form-field">
                <span className="form-field__label">Autonomy</span>
                <select
                  value={draft.autonomousLevel}
                  onChange={(event) => update('autonomousLevel', event.target.value as AppSettings['autonomousLevel'])}
                  className="select-input"
                >
                  {AUTONOMY_LEVELS.map((level) => (
                    <option key={level.value} value={level.value}>
                      {level.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="form-field">
                <span className="form-field__label">Personality</span>
                <select
                  value={draft.agentPersonality}
                  onChange={(event) => update('agentPersonality', event.target.value as AppSettings['agentPersonality'])}
                  className="select-input"
                >
                  {PERSONALITIES.map((personality) => (
                    <option key={personality.value} value={personality.value}>
                      {personality.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="toggle-grid">
              <label className="toggle-card">
                <div className="toggle-card__copy">
                  <strong>Auto-connect and sync on open</strong>
                  <span>Applies the saved OpenClaw defaults as soon as the WebUI loads.</span>
                </div>
                <input
                  type="checkbox"
                  checked={draft.autoConnect}
                  onChange={(event) => update('autoConnect', event.target.checked)}
                />
              </label>

              <label className="toggle-card">
                <div className="toggle-card__copy">
                  <strong>Launch native emulator UI after ROM load</strong>
                  <span>Keeps the operator console and native emulator in sync when a cartridge is inserted.</span>
                </div>
                <input
                  type="checkbox"
                  checked={draft.launchUiOnRomLoad}
                  onChange={(event) => update('launchUiOnRomLoad', event.target.checked)}
                />
              </label>
            </div>

            <div className="modal-callout">
              Objective text and runtime controls stay in the main desk so the operator can see intent. Provider and endpoint details stay here.
            </div>
          </section>
        </div>

        <div className="modal__footer">
          <p className="modal__footer-copy">Saving updates the local defaults first, then attempts to sync the runtime.</p>
          <div className="modal__footer-actions">
            <button type="button" onClick={onClose} className="action-button action-button--ghost">
              Cancel
            </button>
            <button type="button" onClick={handleSave} className="action-button action-button--primary">
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
