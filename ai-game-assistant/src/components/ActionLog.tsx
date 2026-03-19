import React, { useState, useEffect, useRef, useCallback } from 'react';

export type ActionLogType = 'info' | 'action' | 'thought' | 'error' | 'success' | 'warning';

export interface ActionLogEntry {
  id: string;
  timestamp: Date;
  type: ActionLogType;
  message: string;
  details?: string;
}

interface ActionLogProps {
  entries?: ActionLogEntry[];
  className?: string;
  maxEntries?: number;
  autoScroll?: boolean;
  onClear?: () => void;
}

// Icons for different log types
const getTypeIcon = (type: ActionLogType): string => {
  switch (type) {
    case 'info': return 'ℹ️';
    case 'action': return '🎮';
    case 'thought': return '💭';
    case 'error': return '❌';
    case 'success': return '✅';
    case 'warning': return '⚠️';
    default: return '📝';
  }
};

const getTypeColor = (type: ActionLogType): string => {
  switch (type) {
    case 'info': return 'text-blue-400';
    case 'action': return 'text-cyan-400';
    case 'thought': return 'text-purple-400';
    case 'error': return 'text-red-400';
    case 'success': return 'text-green-400';
    case 'warning': return 'text-yellow-400';
    default: return 'text-neutral-400';
  }
};

const getTypeBgColor = (type: ActionLogType): string => {
  switch (type) {
    case 'info': return 'bg-blue-500/10';
    case 'action': return 'bg-cyan-500/10';
    case 'thought': return 'bg-purple-500/10';
    case 'error': return 'bg-red-500/10';
    case 'success': return 'bg-green-500/10';
    case 'warning': return 'bg-yellow-500/10';
    default: return 'bg-neutral-500/10';
  }
};

const ActionLog: React.FC<ActionLogProps> = ({
  entries: externalEntries,
  className = '',
  maxEntries = 100,
  autoScroll = true,
  onClear,
}) => {
  const [internalEntries, setInternalEntries] = useState<ActionLogEntry[]>([]);
  const [filter, setFilter] = useState<ActionLogType | 'all'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  // Use external entries if provided, otherwise use internal
  const entries = externalEntries || internalEntries;

  // Add a new log entry (for internal use)
  const addEntry = useCallback((type: ActionLogType, message: string, details?: string) => {
    const entry: ActionLogEntry = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      type,
      message,
      details,
    };

    setInternalEntries(prev => {
      const newEntries = [entry, ...prev];
      return newEntries.slice(0, maxEntries);
    });
  }, [maxEntries]);

  // Clear entries
  const clearEntries = useCallback(() => {
    setInternalEntries([]);
    onClear?.();
  }, [onClear]);

  // Filter entries
  const filteredEntries = entries.filter(entry => {
    if (filter !== 'all' && entry.type !== filter) return false;
    if (searchTerm && !entry.message.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  // Auto-scroll to top when new entries arrive
  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = 0;
    }
  }, [filteredEntries.length, autoScroll]);

  // Format timestamp
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // Format relative time
  const formatRelativeTime = (date: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    return formatTime(date);
  };

  return (
    <div className={`bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-neutral-800/50 border-b border-neutral-700">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-neutral-200">📋 Action Log</span>
          <span className="text-xs text-neutral-500">
            ({filteredEntries.length}/{entries.length})
          </span>
        </div>
        
        {/* Filter & Search */}
        <div className="flex items-center gap-2">
          <select
            value={filter}
            onChange={e => setFilter(e.target.value as ActionLogType | 'all')}
            className="px-2 py-1 bg-neutral-800 border border-neutral-600 rounded text-xs text-neutral-300 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All</option>
            <option value="action">Actions</option>
            <option value="thought">Thoughts</option>
            <option value="info">Info</option>
            <option value="success">Success</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
          
          <input
            type="text"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            placeholder="Search..."
            className="w-24 px-2 py-1 bg-neutral-800 border border-neutral-600 rounded text-xs text-neutral-300 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
          />
          
          <button
            onClick={clearEntries}
            disabled={entries.length === 0}
            className="px-2 py-1 text-xs bg-neutral-700 text-neutral-400 rounded hover:bg-neutral-600 disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Log Entries */}
      <div 
        ref={listRef}
        className="flex-1 overflow-y-auto p-2 space-y-1"
        style={{ maxHeight: '300px' }}
      >
        {filteredEntries.length === 0 ? (
          <div className="text-center py-8 text-neutral-500 text-sm">
            {entries.length === 0 ? 'No log entries yet.' : 'No entries match the filter.'}
          </div>
        ) : (
          filteredEntries.map((entry, index) => (
            <div
              key={entry.id}
              className={`flex items-start gap-2 p-2 rounded ${getTypeBgColor(entry.type)} ${
                index === 0 ? 'animate-highlight' : ''
              }`}
            >
              {/* Type Icon */}
              <span className="text-sm flex-shrink-0 mt-0.5" title={entry.type}>
                {getTypeIcon(entry.type)}
              </span>
              
              {/* Timestamp */}
              <span className="text-xs text-neutral-500 flex-shrink-0 w-16" title={formatTime(entry.timestamp)}>
                {formatRelativeTime(entry.timestamp)}
              </span>
              
              {/* Message */}
              <div className="flex-1 min-w-0">
                <p className={`text-sm ${getTypeColor(entry.type)} break-words`}>
                  {entry.message}
                </p>
                {entry.details && (
                  <p className="text-xs text-neutral-500 mt-0.5 font-mono truncate" title={entry.details}>
                    {entry.details}
                  </p>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Stats */}
      {entries.length > 0 && (
        <div className="p-2 bg-neutral-800/30 border-t border-neutral-700 text-xs text-neutral-500 flex justify-between">
          <div className="flex gap-3">
            <span>Actions: {entries.filter(e => e.type === 'action').length}</span>
            <span>Thoughts: {entries.filter(e => e.type === 'thought').length}</span>
            <span>Errors: {entries.filter(e => e.type === 'error').length}</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Export types and helper functions
export { getTypeIcon, getTypeColor, getTypeBgColor };

export default ActionLog;