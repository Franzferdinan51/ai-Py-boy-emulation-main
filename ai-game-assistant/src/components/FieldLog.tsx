import React, { useEffect, useRef, useState, useCallback } from 'react';
import apiService from '../../services/apiService';

export type EventKind =
  | 'THINK'
  | 'DECIDE'
  | 'ACT'
  | 'MILESTONE'
  | 'ALERT'
  | 'OBSERVE'
  | 'REFLECT';

export interface AgentEvent {
  id: string | number;
  kind: EventKind | string;
  session_id?: string | null;
  message: string;
  data?: Record<string, unknown>;
  timestamp: string;
}

interface FieldLogProps {
  /** If true, subscribes to the SSE event stream. Default false. */
  liveStream?: boolean;
  /** Max entries to keep in memory. Default 200. */
  maxEntries?: number;
  /** Optional filter — only show events whose kind is in this list. */
  kinds?: EventKind[];
  /** Optional session_id to filter by. */
  sessionId?: string;
  /** Polling interval when liveStream is false. Default 5000 ms. */
  pollIntervalMs?: number;
  className?: string;
  onClear?: () => void;
}

const KIND_META: Record<string, { icon: string; color: string; bg: string; label: string }> = {
  THINK:    { icon: '💭', color: 'text-purple-300',  bg: 'bg-purple-500/10',  label: 'Think' },
  DECIDE:   { icon: '🧭', color: 'text-blue-300',    bg: 'bg-blue-500/10',    label: 'Decide' },
  ACT:      { icon: '🎮', color: 'text-cyan-300',    bg: 'bg-cyan-500/10',    label: 'Act' },
  MILESTONE:{ icon: '🏆', color: 'text-yellow-300',  bg: 'bg-yellow-500/10',  label: 'Milestone' },
  ALERT:    { icon: '⚠️', color: 'text-red-300',     bg: 'bg-red-500/10',     label: 'Alert' },
  OBSERVE:  { icon: '👁',  color: 'text-emerald-300', bg: 'bg-emerald-500/10', label: 'Observe' },
  REFLECT:  { icon: '🔄', color: 'text-orange-300',  bg: 'bg-orange-500/10',  label: 'Reflect' },
};

function metaFor(kind: string) {
  return KIND_META[kind] ?? { icon: '📝', color: 'text-neutral-300', bg: 'bg-neutral-500/10', label: kind };
}

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return ts;
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return ts;
  }
}

/**
 * FieldLog — a streaming reasoning log for agent_features/events.py.
 *
 * Two modes:
 *  - Polling (default): GET /api/agent/events every pollIntervalMs.
 *  - Live (liveStream=true): GET /api/agent/events/stream (SSE).
 */
const FieldLog: React.FC<FieldLogProps> = ({
  liveStream = false,
  maxEntries = 200,
  kinds,
  sessionId,
  pollIntervalMs = 5000,
  className = '',
  onClear,
}) => {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [filterKind, setFilterKind] = useState<string>('ALL');
  const [search, setSearch] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [streamStatus, setStreamStatus] = useState<'idle' | 'live' | 'error'>('idle');
  const [streamError, setStreamError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const evtSourceRef = useRef<EventSource | null>(null);

  // ----- Filter helpers --------------------------------------------
  const filteredEvents = React.useMemo(() => {
    let list = events;
    if (filterKind !== 'ALL') list = list.filter((e) => e.kind === filterKind);
    if (kinds && kinds.length) list = list.filter((e) => kinds.includes(e.kind as EventKind));
    if (sessionId) list = list.filter((e) => !e.session_id || e.session_id === sessionId);
    if (search.trim()) {
      const needle = search.trim().toLowerCase();
      list = list.filter(
        (e) =>
          (e.message || '').toLowerCase().includes(needle) ||
          (e.kind || '').toLowerCase().includes(needle)
      );
    }
    return list;
  }, [events, filterKind, kinds, sessionId, search]);

  // ----- Auto-scroll ------------------------------------------------
  useEffect(() => {
    if (!autoScroll || !listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [filteredEvents, autoScroll]);

  // ----- Polling mode -----------------------------------------------
  useEffect(() => {
    if (liveStream) return;
    let cancelled = false;

    const fetchOnce = async () => {
      try {
        const resp = await apiService.listEvents({ limit: maxEntries, session_id: sessionId });
        if (cancelled) return;
        const sorted = (resp.events || []).slice().sort((a, b) => {
          const ta = new Date(a.timestamp).getTime() || 0;
          const tb = new Date(b.timestamp).getTime() || 0;
          return ta - tb;
        });
        setEvents(sorted);
      } catch (e) {
        // Swallow network errors; FieldLog is informational.
      }
    };

    fetchOnce();
    const interval = window.setInterval(fetchOnce, pollIntervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [liveStream, maxEntries, pollIntervalMs, sessionId]);

  // ----- SSE live stream mode --------------------------------------
  useEffect(() => {
    if (!liveStream) return;

    setStreamStatus('live');
    setStreamError(null);

    let es: EventSource | null = null;
    try {
      // SSE needs absolute URL with backend; EventSource doesn't support
      // custom headers, so we use the URL the apiService knows about.
      const backend = (apiService as any).getBaseUrl?.() ?? '';
      const url = `${backend}/api/agent/events/stream${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`;
      es = new EventSource(url);
      evtSourceRef.current = es;

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data && data.kind) {
            setEvents((prev) => {
              const next = [...prev, data as AgentEvent];
              if (next.length > maxEntries) next.splice(0, next.length - maxEntries);
              return next;
            });
          }
        } catch {
          // ignore malformed line
        }
      };

      es.onerror = () => {
        setStreamStatus('error');
        setStreamError('Stream disconnected — falling back to polling.');
        es?.close();
        evtSourceRef.current = null;
      };
    } catch (e: any) {
      setStreamStatus('error');
      setStreamError(e?.message ?? 'Failed to open stream');
    }

    return () => {
      es?.close();
      evtSourceRef.current = null;
    };
  }, [liveStream, maxEntries, sessionId]);

  // ----- Clear handler ----------------------------------------------
  const handleClear = useCallback(async () => {
    try {
      await apiService.clearEvents();
    } catch {
      // ignore
    }
    setEvents([]);
    if (onClear) onClear();
  }, [onClear]);

  const kindOptions = ['ALL', 'THINK', 'DECIDE', 'ACT', 'MILESTONE', 'ALERT', 'OBSERVE', 'REFLECT'];

  return (
    <div className={`flex flex-col h-full bg-neutral-900/70 border border-neutral-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between gap-2 p-2 border-b border-neutral-700 bg-neutral-800/70 rounded-t-lg">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-neutral-200">📡 Field Log</span>
          {liveStream && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${streamStatus === 'live' ? 'bg-emerald-700/30 text-emerald-300' : streamStatus === 'error' ? 'bg-red-700/30 text-red-300' : 'bg-neutral-700 text-neutral-300'}`}>
              {streamStatus === 'live' ? 'LIVE' : streamStatus === 'error' ? 'ERROR' : 'IDLE'}
            </span>
          )}
          <span className="text-[10px] text-neutral-500">{filteredEvents.length}/{events.length}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <input
            type="text"
            placeholder="search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-0.5 text-xs w-28 focus:outline-none focus:border-neutral-500"
          />
          <select
            value={filterKind}
            onChange={(e) => setFilterKind(e.target.value)}
            className="bg-neutral-900 border border-neutral-700 rounded px-1 py-0.5 text-xs focus:outline-none focus:border-neutral-500"
          >
            {kindOptions.map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setAutoScroll((v) => !v)}
            className={`text-[10px] px-1.5 py-0.5 rounded ${autoScroll ? 'bg-cyan-700/40 text-cyan-200' : 'bg-neutral-700 text-neutral-300'}`}
            title="Auto-scroll to latest event"
          >
            ↓
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-200"
            title="Clear events"
          >
            clear
          </button>
        </div>
      </div>

      {streamError && (
        <div className="px-3 py-1 text-[11px] text-amber-300 bg-amber-900/20 border-b border-amber-900/30">
          {streamError}
        </div>
      )}

      {/* Event list */}
      <div
        ref={listRef}
        className="flex-1 overflow-y-auto p-2 space-y-1 text-xs leading-snug"
        style={{ minHeight: 120 }}
      >
        {filteredEvents.length === 0 ? (
          <div className="text-neutral-500 italic text-center py-4">
            {liveStream ? 'Waiting for events…' : 'No events yet.'}
          </div>
        ) : (
          filteredEvents.map((e, i) => {
            const meta = metaFor(e.kind);
            return (
              <div
                key={`${e.id ?? 'idx'}-${i}`}
                className={`flex gap-2 px-2 py-1 rounded ${meta.bg} border border-neutral-800/50`}
              >
                <span className="text-base leading-none flex-shrink-0 mt-0.5">{meta.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-[10px] font-semibold uppercase tracking-wide ${meta.color}`}>
                      {meta.label}
                    </span>
                    <span className="text-[10px] text-neutral-500">{formatTimestamp(e.timestamp)}</span>
                    {e.session_id && (
                      <span className="text-[10px] text-neutral-600 truncate">
                        @{e.session_id.slice(0, 8)}
                      </span>
                    )}
                  </div>
                  <div className="text-neutral-200 break-words whitespace-pre-wrap">{e.message}</div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default FieldLog;
