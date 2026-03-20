import React, { useEffect, useState, useMemo } from 'react';
import type { AppSettings, ModelInfo } from '../../services/apiService';
import apiService from '../../services/apiService';

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

interface ModelOption {
  value: string;
  label: string;
  note: string;
  provider: string;
  isFree?: boolean;
  isVisionCapable?: boolean;
  role?: string;  // 'primary', 'vision', 'planning', 'fallback', 'general'
}

interface ModelGroup {
  provider: string;
  label: string;
  models: ModelOption[];
}

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

const PROVIDER_LABELS: Record<string, string> = {
  'bailian': 'Alibaba Bailian',
  'openai': 'OpenAI',
  'google': 'Google',
  'anthropic': 'Anthropic',
  'meta': 'Meta',
  'nvidia': 'NVIDIA',
  'mistral': 'Mistral',
  'cohere': 'Cohere',
  'lmstudio': 'LM Studio',
  'local': 'Local',
};

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

// Null-safe helper functions
const safeString = (value: unknown, fallback = ''): string => {
  if (value === null || value === undefined) return fallback;
  return String(value);
};

const safeBoolean = (value: unknown, fallback = false): boolean => {
  if (value === null || value === undefined) return fallback;
  return Boolean(value);
};

const safeNumber = (value: unknown, fallback = 0): number => {
  if (value === null || value === undefined || typeof value !== 'number') return fallback;
  return value;
};

// Model role badge helper
const getRoleBadge = (role?: string): string => {
  switch (role) {
    case 'primary': return '⭐';
    case 'vision': return '👁️';
    case 'planning': return '🧠';
    case 'fallback': return '↩️';
    default: return '';
  }
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
  
  // Dynamic model lists from OpenClaw
  const [allModels, setAllModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  
  // Model input mode (select from list vs manual input)
  const [visionInputMode, setVisionInputMode] = useState<'select' | 'manual'>('select');
  const [planningInputMode, setPlanningInputMode] = useState<'select' | 'manual'>('select');
  
  // Search filter for models
  const [visionSearch, setVisionSearch] = useState('');
  const [planningSearch, setPlanningSearch] = useState('');

  // Load models from OpenClaw when modal opens
  useEffect(() => {
    if (isOpen && !modelsLoaded) {
      void loadModelsFromOpenClaw();
    }
  }, [isOpen]);

  // Reset input mode based on current model value
  useEffect(() => {
    if (draft.visionModel && !allModels.find(m => m.id === draft.visionModel)) {
      setVisionInputMode('manual');
    }
    if (draft.planningModel && !allModels.find(m => m.id === draft.planningModel)) {
      setPlanningInputMode('manual');
    }
  }, [draft.visionModel, draft.planningModel, allModels]);

  const loadModelsFromOpenClaw = async () => {
    setLoadingModels(true);
    try {
      // Fetch all models
      const response = await apiService.getOpenClawModels().catch(() => ({ models: [] }));
      const models = response.models || [];
      setAllModels(models);
      setModelsLoaded(true);

      // Auto-select recommended models if not set
      if (models.length > 0) {
        const visionModels = models.filter(m => m.is_vision_capable);
        const planningModels = models.filter(m => !m.is_vision_capable || m.capabilities?.includes('reasoning'));
        
        if (!draft.visionModel && visionModels.length > 0) {
          const recommended = visionModels.find(m => m.is_free) || visionModels[0];
          setDraft(prev => ({ ...prev, visionModel: recommended.id }));
        }
        if (!draft.planningModel && planningModels.length > 0) {
          const recommended = planningModels.find(m => m.is_free) || planningModels[0];
          setDraft(prev => ({ ...prev, planningModel: recommended.id }));
        }
      }
    } catch (error) {
      console.error('Failed to load models from OpenClaw:', error);
      setAllModels([]);
      setModelsLoaded(true);
    } finally {
      setLoadingModels(false);
    }
  };

  // Group models by provider for cleaner display
  const modelGroups = useMemo(() => {
    const groups: Record<string, ModelGroup> = {};
    
    allModels.forEach(model => {
      const provider = safeString(model.provider, 'other');
      const providerLabel = PROVIDER_LABELS[provider] || provider.charAt(0).toUpperCase() + provider.slice(1);
      
      if (!groups[provider]) {
        groups[provider] = {
          provider,
          label: providerLabel,
          models: []
        };
      }
      
      groups[provider].models.push({
        value: model.id,
        label: safeString(model.label, model.name || model.id),
        note: model.description || (model.is_free ? 'FREE' : 'API credits'),
        provider,
        isFree: model.is_free,
        isVisionCapable: model.is_vision_capable,
        role: model.role  // Include role for badge display
      });
    });
    
    // Sort groups: free providers first, then alphabetically
    return Object.values(groups).sort((a, b) => {
      const aHasFree = a.models.some(m => m.isFree);
      const bHasFree = b.models.some(m => m.isFree);
      if (aHasFree && !bHasFree) return -1;
      if (!aHasFree && bHasFree) return 1;
      return a.label.localeCompare(b.label);
    });
  }, [allModels]);

  // Filter vision models
  const visionGroups = useMemo(() => {
    const groups: Record<string, ModelGroup> = {};
    const searchLower = visionSearch.toLowerCase();
    
    modelGroups.forEach(group => {
      const filteredModels = group.models.filter(m => 
        m.isVisionCapable && 
        (m.label.toLowerCase().includes(searchLower) || 
         m.value.toLowerCase().includes(searchLower))
      );
      
      if (filteredModels.length > 0) {
        groups[group.provider] = { ...group, models: filteredModels };
      }
    });
    
    return Object.values(groups);
  }, [modelGroups, visionSearch]);

  // Filter planning models
  const planningGroups = useMemo(() => {
    const groups: Record<string, ModelGroup> = {};
    const searchLower = planningSearch.toLowerCase();
    
    modelGroups.forEach(group => {
      const filteredModels = group.models.filter(m => 
        (m.isVisionCapable === false || m.provider === 'bailian') &&
        (m.label.toLowerCase().includes(searchLower) || 
         m.value.toLowerCase().includes(searchLower))
      );
      
      if (filteredModels.length > 0) {
        groups[group.provider] = { ...group, models: filteredModels };
      }
    });
    
    return Object.values(groups);
  }, [modelGroups, planningSearch]);

  // Fallback models if none discovered
  const fallbackVisionModels: ModelOption[] = [
    { value: 'bailian/kimi-k2.5', label: 'Kimi K2.5 (Vision)', note: 'Best for game screen analysis (FREE)', provider: 'bailian', isFree: true, isVisionCapable: true, role: 'vision' },
    { value: 'bailian/qwen-vl-plus', label: 'Qwen VL Plus (Vision)', note: 'High quality vision (quota)', provider: 'bailian', isFree: false, isVisionCapable: true, role: 'fallback' },
  ];

  const fallbackPlanningModels: ModelOption[] = [
    { value: 'bailian/glm-5', label: 'GLM-5 (Reasoning)', note: 'Fast decisions, great for games', provider: 'bailian', isFree: false, role: 'planning' },
    { value: 'bailian/MiniMax-M2.5', label: 'MiniMax M2.5 (FREE)', note: 'Unlimited, reliable (FREE)', provider: 'bailian', isFree: true, role: 'fallback' },
    { value: 'bailian/qwen3.5-plus', label: 'Qwen 3.5 Plus (Reasoning)', note: 'Best reasoning (quota)', provider: 'bailian', isFree: false, role: 'fallback' },
  ];

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

  const activeVisionModel = [...visionGroups.flatMap(g => g.models), ...fallbackVisionModels].find((model) => model.value === draft.visionModel);
  const activePlanningModel = [...planningGroups.flatMap(g => g.models), ...fallbackPlanningModels].find((model) => model.value === draft.planningModel);

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

  // Render model select dropdown with grouped options
  const renderModelSelect = (
    groups: ModelGroup[],
    fallbackModels: ModelOption[],
    value: string,
    onChange: (value: string) => void,
    search: string,
    onSearchChange: (value: string) => void,
    placeholder: string,
    isLoading: boolean
  ) => {
    const hasGroups = groups.length > 0;
    const displayGroups = hasGroups ? groups : [{ provider: 'fallback', label: 'Fallback Models', models: fallbackModels }];
    const totalModels = displayGroups.reduce((sum, g) => sum + g.models.length, 0);

    return (
      <div className="model-select-container">
        {/* Search input for filtering models */}
        {totalModels > 5 && (
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="text-input model-search"
            placeholder={`Search ${totalModels} models...`}
          />
        )}
        
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="select-input"
          disabled={isLoading}
        >
          {isLoading ? (
            <option>Loading models from OpenClaw...</option>
          ) : (
            displayGroups.map(group => (
              <optgroup key={group.provider} label={group.label}>
                {group.models.map(model => (
                  <option key={model.value} value={model.value}>
                    {getRoleBadge(model.role)} {model.label || model.value} {model.isFree ? '★' : ''} — {model.note}
                  </option>
                ))}
              </optgroup>
            ))
          )}
        </select>
      </div>
    );
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
              Configure endpoints and model preferences. Models are discovered from OpenClaw automatically.
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
                    value={safeString(draft.backendUrl)}
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
                    value={safeString(draft.openclawMcpEndpoint)}
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
                <span className="form-field__help">MCP endpoint for OpenClaw gateway (default: http://localhost:18789).</span>
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
              <h3>Model orchestration</h3>
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
                  {AI_PROVIDERS.find(p => p.value === (draft.aiProvider || 'openclaw'))?.note || 'Select an AI provider'}
                </span>
              </label>
            </div>

            {/* LM Studio / OpenAI-Compatible Settings */}
            {(draft.aiProvider === 'lmstudio' || draft.aiProvider === 'openai-compatible') && (
              <>
                <div className="modal-section__divider" />
                
                <div className="settings-grid">
                  <label className="form-field">
                    <span className="form-field__label">Custom Endpoint URL</span>
                    <div className="field-row">
                      <input
                        type="text"
                        value={safeString(draft.lmStudioUrl)}
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
                      value={safeString(draft.lmStudioThinkingModel)}
                      onChange={(event) => update('lmStudioThinkingModel', event.target.value)}
                      className="text-input"
                      placeholder="qwen3.5-35b-a3b"
                    />
                    <span className="form-field__help">
                      Model for text reasoning and decision-making.
                    </span>
                  </label>

                  <label className="form-field">
                    <span className="form-field__label">Vision Model</span>
                    <input
                      type="text"
                      value={safeString(draft.lmStudioVisionModel)}
                      onChange={(event) => update('lmStudioVisionModel', event.target.value)}
                      className="text-input"
                      placeholder="qwen3-vl-8b"
                    />
                    <span className="form-field__help">
                      Model for screen/image analysis.
                    </span>
                  </label>
                </div>
              </>
            )}
          </section>

          <section className="modal-section">
            <div className="modal-section__header">
              <span className="modal-section__eyebrow">OpenClaw Models</span>
              <h3>Vision and planning models</h3>
              <p className="modal-section__subtitle">
                {loadingModels 
                  ? 'Discovering models from OpenClaw...' 
                  : modelsLoaded 
                    ? `Found ${allModels.length} models from ${modelGroups.length} providers`
                    : 'Click refresh to discover models from OpenClaw'}
              </p>
            </div>

            <div className="settings-grid">
              {/* Vision Model */}
              <label className="form-field">
                <div className="form-field__header">
                  <span className="form-field__label">Vision Model</span>
                  <div className="input-mode-toggle">
                    <button
                      type="button"
                      className={`toggle-btn ${visionInputMode === 'select' ? 'toggle-btn--active' : ''}`}
                      onClick={() => setVisionInputMode('select')}
                    >
                      Select
                    </button>
                    <button
                      type="button"
                      className={`toggle-btn ${visionInputMode === 'manual' ? 'toggle-btn--active' : ''}`}
                      onClick={() => setVisionInputMode('manual')}
                    >
                      Manual
                    </button>
                  </div>
                </div>
                
                {visionInputMode === 'select' ? (
                  renderModelSelect(
                    visionGroups,
                    fallbackVisionModels,
                    safeString(draft.visionModel),
                    (value) => update('visionModel', value),
                    visionSearch,
                    setVisionSearch,
                    'Select vision model...',
                    loadingModels
                  )
                ) : (
                  <input
                    type="text"
                    value={safeString(draft.visionModel)}
                    onChange={(event) => update('visionModel', event.target.value)}
                    className="text-input"
                    placeholder="bailian/kimi-k2.5"
                  />
                )}
                <span className="form-field__help">
                  {activeVisionModel?.note || 'Model for screen/image analysis'}
                  {activeVisionModel?.isFree ? ' (FREE)' : ''}
                </span>
              </label>

              {/* Planning Model */}
              <label className="form-field">
                <div className="form-field__header">
                  <span className="form-field__label">Planning Model</span>
                  <div className="input-mode-toggle">
                    <button
                      type="button"
                      className={`toggle-btn ${planningInputMode === 'select' ? 'toggle-btn--active' : ''}`}
                      onClick={() => setPlanningInputMode('select')}
                    >
                      Select
                    </button>
                    <button
                      type="button"
                      className={`toggle-btn ${planningInputMode === 'manual' ? 'toggle-btn--active' : ''}`}
                      onClick={() => setPlanningInputMode('manual')}
                    >
                      Manual
                    </button>
                  </div>
                </div>
                
                {planningInputMode === 'select' ? (
                  renderModelSelect(
                    planningGroups,
                    fallbackPlanningModels,
                    safeString(draft.planningModel),
                    (value) => update('planningModel', value),
                    planningSearch,
                    setPlanningSearch,
                    'Select planning model...',
                    loadingModels
                  )
                ) : (
                  <input
                    type="text"
                    value={safeString(draft.planningModel)}
                    onChange={(event) => update('planningModel', event.target.value)}
                    className="text-input"
                    placeholder="bailian/glm-5"
                  />
                )}
                <span className="form-field__help">
                  {activePlanningModel?.note || 'Model for decision making and reasoning'}
                  {activePlanningModel?.isFree ? ' (FREE)' : ''}
                </span>
              </label>
            </div>

            <div className="settings-actions">
              <button
                type="button"
                onClick={() => void loadModelsFromOpenClaw()}
                disabled={loadingModels}
                className="action-button action-button--ghost"
              >
                {loadingModels ? 'Refreshing...' : 'Refresh Models'}
              </button>
            </div>
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
                <span className="form-field__label">Autonomy</span>
                <select
                  value={draft.autonomousLevel}
                  onChange={(event) => update('autonomousLevel', event.target.value as AppSettings['autonomousLevel'])}
                  className="select-input"
                >
                  {AUTONOMY_LEVELS.map((level) => (
                    <option key={safeString(level.value)} value={level.value}>
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
                    <option key={safeString(personality.value)} value={personality.value}>
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
                  checked={safeBoolean(draft.autoConnect)}
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
                  checked={safeBoolean(draft.launchUiOnRomLoad)}
                  onChange={(event) => update('launchUiOnRomLoad', event.target.checked)}
                />
              </label>

              <label className="toggle-card">
                <div className="toggle-card__copy">
                  <strong>Use dual-model routing</strong>
                  <span>Keep OpenClaw vision and planning split into separate model calls.</span>
                </div>
                <input
                  type="checkbox"
                  checked={safeBoolean(draft.useDualModel)}
                  onChange={(event) => update('useDualModel', event.target.checked)}
                />
              </label>
            </div>

            <div className="modal-callout">
              Provider and endpoint details stay here. Objective text and runtime controls are in the main desk.
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