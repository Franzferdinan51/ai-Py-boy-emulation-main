import React, { useState, useCallback, useEffect } from 'react';

interface MemoryAddress {
  address: number;
  value: number;
  label?: string;
}

interface MemoryInspectorProps {
  backendUrl: string;
  className?: string;
  maxWatchAddresses?: number;
}

const MemoryInspector: React.FC<MemoryInspectorProps> = ({
  backendUrl,
  className = '',
  maxWatchAddresses = 10,
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
    const address = parseInt(newAddress, 16);
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
    const label = newLabel.trim() || undefined;
    setWatchedAddresses(prev => [...prev, { address, value: 0, label }]);
    setNewAddress('');
    setNewLabel('');
    setError(null);
  }, [newAddress, newLabel, watchedAddresses.length, maxWatchAddresses]);

  const removeWatchAddress = useCallback((address: number) => {
    setWatchedAddresses(prev => prev.filter(a => a.address !== address));
  }, []);

  const updateLabel = useCallback((address: number, label: string) => {
    setWatchedAddresses(prev => prev.map(a => a.address === address ? { ...a, label: label || undefined } : a));
  }, []);

  const refreshMemory = useCallback(async () => {
    if (!backendUrl || watchedAddresses.length === 0) return;
    setIsLoading(true);
    setError(null);
    try {
      const addresses = watchedAddresses.map(a => a.address);
      const response = await fetch(`${backendUrl}/api/memory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ addresses }),
      });
      if (!response.ok) throw new Error(`Failed to read memory: ${response.statusText}`);
      const data = await response.json();
      const newValues = new Map<number, number>();
      for (const addr of addresses) {
        newValues.set(addr, data[addr.toString(16)] || data[addr] || 0);
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
    if (!autoRefresh || watchedAddresses.length === 0) return;
    refreshMemory();
    const interval = setInterval(refreshMemory, 1000);
    return () => clearInterval(interval);
  }, [autoRefresh, watchedAddresses.length, refreshMemory]);

  const formatValue = (value: number): string => {
    if (value === undefined || value === null) return '--';
    switch (displayMode) {
      case 'hex': return `0x${value.toString(16).toUpperCase().padStart(4, '0')}`;
      case 'dec': return value.toString().padStart(5, ' ');
      case 'binary': return value.toString(2).padStart(8, '0');
      default: return value.toString();
    }
  };

  const getByteColor = (value: number): string => value > 127 ? 'text-red-400' : 'text-green-400';

  const commonAddresses = [
    { address: 0x8000, label: 'VRAM Start' },
    { address: 0xC000, label: 'WRAM Start' },
    { address: 0xFF00, label: 'I/O Registers' },
    { address: 0xFF04, label: 'DIV Register' },
    { address: 0xFF44, label: 'LY (LCD Y)' },
    { address: 0xFF45, label: 'LYC' },
  ];

  const quickAddAddress = (addr: number, label?: string) => {
    if (!watchedAddresses.some(a => a.address === addr)) {
      setWatchedAddresses(prev => [...prev, { address: addr, value: 0, label }]);
    }
  };

  return (
    <div className={`bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden ${className}`}>
      <div className="flex items-center justify-between p-3 bg-neutral-800/50 border-b border-neutral-700">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-neutral-200">🔍 Memory Inspector</span>
          {isLoading && <div className="w-3 h-3 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex text-xs">
            {(['hex', 'dec', 'binary'] as const).map(mode => (
              <button key={mode} onClick={() => setDisplayMode(mode)}
                className={`px-2 py-1 ${displayMode === mode ? 'bg-cyan-600 text-white' : 'bg-neutral-700 text-neutral-400 hover:bg-neutral-600'} ${mode === 'hex' ? 'rounded-l' : ''} ${mode === 'binary' ? 'rounded-r' : ''}`}>
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
          <button onClick={() => setAutoRefresh(!autoRefresh)} className={`px-2 py-1 text-xs rounded ${autoRefresh ? 'bg-green-600 text-white' : 'bg-neutral-700 text-neutral-400'}`} title="Auto-refresh">🔄</button>
          <button onClick={refreshMemory} disabled={isLoading || watchedAddresses.length === 0} className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded hover:bg-neutral-600 disabled:opacity-50">Refresh</button>
        </div>
      </div>

      {error && <div className="p-2 bg-red-900/30 border-b border-red-800 text-red-400 text-sm">{error}</div>}

      <div className="p-3 bg-neutral-800/30 border-b border-neutral-700">
        <div className="flex gap-2">
          <input type="text" value={newAddress} onChange={e => setNewAddress(e.target.value)} placeholder="Address (e.g., 0xC000)"
            className="flex-1 px-3 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-sm text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500" onKeyDown={e => e.key === 'Enter' && addWatchAddress()} />
          <input type="text" value={newLabel} onChange={e => setNewLabel(e.target.value)} placeholder="Label (optional)"
            className="w-32 px-3 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-sm text-neutral-200 placeholder-neutral-500 focus:outline-none focus:border-cyan-500" onKeyDown={e => e.key === 'Enter' && addWatchAddress()} />
          <button onClick={addWatchAddress} disabled={!newAddress} className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed">+ Add</button>
        </div>
        <div className="flex flex-wrap gap-1 mt-2">
          <span className="text-xs text-neutral-500 mr-1">Quick:</span>
          {commonAddresses.map(({ address, label }) => (
            <button key={address} onClick={() => quickAddAddress(address, label)} className="text-xs px-2 py-0.5 bg-neutral-700 text-neutral-400 rounded hover:bg-neutral-600">{label || `0x${address.toString(16).toUpperCase()}`}</button>
          ))}
        </div>
      </div>

      <div className="p-3">
        {watchedAddresses.length === 0 ? (
          <div className="text-center py-8 text-neutral-500 text-sm">No addresses being watched.<br />Add addresses above or use quick-add buttons.</div>
        ) : (
          <div className="space-y-1">
            <div className="flex items-center text-xs text-neutral-500 px-2 py-1">
              <span className="w-12">Addr</span>
              <span className="w-24 ml-2">Label</span>
              <span className="flex-1 text-right">Value</span>
              <span className="w-16 text-right ml-4">Binary</span>
              <span className="w-8"></span>
            </div>
            {watchedAddresses.map(({ address, label }) => {
              const value = memoryValues.get(address) ?? 0;
              return (
                <div key={address} className="flex items-center px-2 py-1.5 bg-neutral-800/50 rounded hover:bg-neutral-800">
                  <span className="w-12 font-mono text-xs text-cyan-400">0x{address.toString(16).toUpperCase().padStart(4, '0')}</span>
                  <input type="text" value={label || ''} onChange={e => updateLabel(address, e.target.value)} placeholder="Label..."
                    className="w-24 ml-2 px-2 py-0.5 bg-neutral-900 border border-neutral-700 rounded text-xs text-neutral-300 placeholder-neutral-600 focus:outline-none focus:border-cyan-500" />
                  <span className={`flex-1 text-right font-mono text-sm ${getByteColor(value)}`}>{formatValue(value)}</span>
                  <span className="w-16 text-right font-mono text-xs text-neutral-500">{value.toString(2).padStart(8, '0')}</span>
                  <button onClick={() => removeWatchAddress(address)} className="w-8 text-center text-neutral-500 hover:text-red-400" title="Remove">×</button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default MemoryInspector;