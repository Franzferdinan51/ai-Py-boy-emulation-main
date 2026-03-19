import React, { useState, useCallback, useEffect } from 'react';
import { Database, RefreshCw, Plus, X, Eye, EyeOff } from 'lucide-react';

interface MemoryAddress {
  address: number;
  value: number;
  label?: string;
}

interface MemoryInspectorProps {
  backendUrl: string;
  className?: string;
  maxWatchAddresses?: number;
  isRomLoaded?: boolean;
}

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const MemoryInspector: React.FC<MemoryInspectorProps> = ({
  backendUrl,
  className = '',
  maxWatchAddresses = 10,
  isRomLoaded = true,
}) => {
  const [watchedAddresses, setWatchedAddresses] = useState<MemoryAddress[]>([]);
  const [newAddress, setNewAddress] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [memoryValues, setMemoryValues] = useState<Map<number, number>>(new Map());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<'hex' | 'dec' | 'binary'>('hex');
  const [autoRefresh, setAutoRefresh] = useState(true);

  const addWatchAddress = useCallback(() => {
    const addressStr = (newAddress ?? '').trim();
    if (!addressStr) {
      setError('Enter an address to watch.');
      return;
    }

    const address = addressStr.startsWith('0x') || addressStr.startsWith('0X')
      ? parseInt(addressStr, 16)
      : parseInt(addressStr, 10);

    if (isNaN(address) || address < 0 || address > 0xFFFF) {
      setError('Invalid address. Enter hex (e.g., 0xFF00) or decimal.');
      return;
    }

    if (watchedAddresses.length >= maxWatchAddresses) {
      setError(`Maximum ${maxWatchAddresses} watch addresses allowed.`);
      return;
    }

    if (watchedAddresses.some(a => a.address === address)) {
      setError('Address already being watched.');
      return;
    }

    const label = (newLabel ?? '').trim() || undefined;
    setWatchedAddresses(prev => [...prev, { address, value: 0, label }]);
    setNewAddress('');
    setNewLabel('');
    setError(null);
  }, [newAddress, newLabel, watchedAddresses, maxWatchAddresses]);

  const removeWatchAddress = useCallback((address: number) => {
    setWatchedAddresses(prev => prev.filter(a => a.address !== address));
  }, []);

  const updateLabel = useCallback((address: number, label: string) => {
    setWatchedAddresses(prev =>
      prev.map(a => a.address === address ? { ...a, label: label || undefined } : a)
    );
  }, []);

  const refreshMemory = useCallback(async () => {
    if (!backendUrl || watchedAddresses.length === 0) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const addresses = watchedAddresses.map(a => a.address);
      const response = await fetch(`${backendUrl}/api/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ addresses }),
      });

      if (!response.ok) {
        throw new Error(`Failed to read memory: ${response.statusText}`);
      }

      const data = await response.json();
      const newValues = new Map<number, number>();

      for (const addr of addresses) {
        const value = data[addr.toString(16)] ?? data[addr] ?? 0;
        newValues.set(addr, typeof value === 'number' ? value : 0);
      }

      setMemoryValues(newValues);
    } catch (err) {
      console.error('Memory read error:', err);
      setError(err instanceof Error ? err.message : 'Failed to read memory');
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl, watchedAddresses]);

  useEffect(() => {
    if (!autoRefresh || watchedAddresses.length === 0 || !isRomLoaded) {
      return;
    }

    void refreshMemory();
    const interval = setInterval(refreshMemory, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, watchedAddresses.length, refreshMemory, isRomLoaded]);

  const formatValue = (value: number | null | undefined): string => {
    if (value === null || value === undefined) {
      return '--';
    }

    const num = typeof value === 'number' ? value : 0;

    switch (displayMode) {
      case 'hex':
        return `0x${num.toString(16).toUpperCase().padStart(4, '0')}`;
      case 'dec':
        return num.toString().padStart(5, ' ');
      case 'binary':
        return num.toString(2).padStart(8, '0');
      default:
        return num.toString();
    }
  };

  const getByteColor = (value: number | null | undefined): string => {
    if (value === null || value === undefined) {
      return '';
    }
    return value > 127 ? 'memory-value--high' : 'memory-value--low';
  };

  const commonAddresses = [
    { address: 0x8000, label: 'VRAM Start' },
    { address: 0xC000, label: 'WRAM Start' },
    { address: 0xFF00, label: 'I/O Registers' },
    { address: 0xFF04, label: 'DIV Register' },
    { address: 0xFF44, label: 'LY (LCD Y)' },
  ];

  const quickAddAddress = (addr: number, label?: string) => {
    if (watchedAddresses.some(a => a.address === addr)) {
      return;
    }

    setWatchedAddresses(prev => [...prev, { address: addr, value: 0, label }]);
  };

  if (!isRomLoaded) {
    return (
      <section className={classNames('data-panel memory-panel', className)}>
        <div className="data-panel__header">
          <div>
            <span className="data-panel__eyebrow">Memory</span>
            <h3 className="data-panel__title">Memory Inspector</h3>
            <p className="data-panel__subtitle">Load a ROM to inspect memory addresses.</p>
          </div>
          <Database className="data-panel__icon" />
        </div>
        <div className="data-panel__body">
          <div className="empty-panel">Load a ROM to inspect memory addresses.</div>
        </div>
      </section>
    );
  }

  return (
    <section className={classNames('data-panel memory-panel', className)}>
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">Memory</span>
          <h3 className="data-panel__title">Memory Inspector</h3>
          <p className="data-panel__subtitle">
            {watchedAddresses.length > 0
              ? `${watchedAddresses.length}/${maxWatchAddresses} addresses watched`
              : 'Add addresses to watch memory values.'}
          </p>
        </div>
        <div className="memory-panel__actions">
          <button
            type="button"
            className="memory-panel__toggle"
            onClick={() => setAutoRefresh(!autoRefresh)}
            title={autoRefresh ? 'Auto-refresh enabled' : 'Auto-refresh disabled'}
          >
            {autoRefresh ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <button
            type="button"
            className="memory-panel__refresh"
            onClick={() => void refreshMemory()}
            disabled={isLoading || watchedAddresses.length === 0}
          >
            <RefreshCw size={16} className={isLoading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      <div className="data-panel__body memory-panel__body">
        {/* Address input */}
        <div className="memory-input-row">
          <input
            type="text"
            value={newAddress}
            onChange={e => setNewAddress(e.target.value)}
            placeholder="Address (e.g., 0xC000)"
            className="memory-input memory-input--address"
            onKeyDown={e => e.key === 'Enter' && addWatchAddress()}
          />
          <input
            type="text"
            value={newLabel}
            onChange={e => setNewLabel(e.target.value)}
            placeholder="Label (optional)"
            className="memory-input memory-input--label"
            onKeyDown={e => e.key === 'Enter' && addWatchAddress()}
          />
          <button
            type="button"
            onClick={addWatchAddress}
            disabled={!newAddress.trim()}
            className="memory-add-button"
          >
            <Plus size={16} />
          </button>
        </div>

        {/* Quick-add buttons */}
        <div className="memory-quick-add">
          <span className="memory-quick-add__label">Quick:</span>
          {commonAddresses.map(({ address, label }) => (
            <button
              key={address}
              type="button"
              onClick={() => quickAddAddress(address, label)}
              className="memory-quick-add__button"
            >
              {label ?? `0x${address.toString(16).toUpperCase()}`}
            </button>
          ))}
        </div>

        {/* Display mode toggle */}
        <div className="memory-display-toggle">
          {(['hex', 'dec', 'binary'] as const).map(mode => (
            <button
              key={mode}
              type="button"
              onClick={() => setDisplayMode(mode)}
              className={classNames(
                'memory-display-toggle__button',
                displayMode === mode && 'memory-display-toggle__button--active'
              )}
            >
              {mode.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Error */}
        {error && <div className="error-banner">{error}</div>}

        {/* Watch list */}
        <div className="memory-watch-list">
          {watchedAddresses.length === 0 ? (
            <div className="empty-panel empty-panel--compact">
              No addresses being watched. Add addresses above or use quick-add buttons.
            </div>
          ) : (
            watchedAddresses.map(({ address, label }) => {
              const value = memoryValues.get(address) ?? 0;
              return (
                <div key={address} className="memory-row">
                  <div className="memory-row__address">
                    <span className="memory-row__hex">0x{address.toString(16).toUpperCase().padStart(4, '0')}</span>
                    <input
                      type="text"
                      value={label ?? ''}
                      onChange={e => updateLabel(address, e.target.value)}
                      placeholder="Label..."
                      className="memory-row__label-input"
                    />
                  </div>
                  <div className="memory-row__value-group">
                    <span className={classNames('memory-row__value', getByteColor(value))}>
                      {formatValue(value)}
                    </span>
                    <span className="memory-row__binary">{value.toString(2).padStart(8, '0')}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeWatchAddress(address)}
                    className="memory-row__remove"
                    title="Remove"
                  >
                    <X size={14} />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
};

export default MemoryInspector;