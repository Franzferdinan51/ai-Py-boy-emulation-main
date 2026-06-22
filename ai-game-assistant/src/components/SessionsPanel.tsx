import React, { useEffect, useState, useCallback } from 'react';
import apiService from '../../services/apiService';

interface Session {
  id: string;
  name: string;
  rom_name?: string | null;
  emulator?: string | null;
  created_at?: string;
  updated_at?: string;
  active: boolean;
  milestones_count?: number;
  objectives_count?: number;
}

interface SessionsPanelProps {
  /** Polling interval for refreshing sessions list (ms). Default 8000. */
  pollIntervalMs?: number;
  className?: string;
}

/**
 * SessionsPanel — minimal UI for backend.agent_features.sessions.
 *
 * Endpoints used:
 *   GET    /api/games                  -> list sessions
 *   GET    /api/games/current          -> active session
 *   POST   /api/games/new              -> create session
 *   POST   /api/games/<id>/activate    -> set active
 *   DELETE /api/games/<id>             -> delete session
 *   POST   /api/games/<id>/save_state  -> save emulator state to disk
 *   GET    /api/games/<id>/save_state  -> load emulator state from disk
 */
const SessionsPanel: React.FC<SessionsPanelProps> = ({
  pollIntervalMs = 8000,
  className = '',
}) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await apiService.listSessions();
      setSessions(resp.sessions || []);
      setActiveId(resp.active_session_id ?? null);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load sessions');
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, pollIntervalMs);
    return () => window.clearInterval(interval);
  }, [refresh, pollIntervalMs]);

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return;
    setBusy('new');
    setError(null);
    try {
      await apiService.createSession({ name: newName.trim() });
      setNewName('');
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to create session');
    } finally {
      setBusy(null);
    }
  }, [newName, refresh]);

  const handleActivate = useCallback(async (id: string) => {
    setBusy(`activate:${id}`);
    setError(null);
    try {
      await apiService.activateSession(id);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to activate session');
    } finally {
      setBusy(null);
    }
  }, [refresh]);

  const handleSave = useCallback(async (id: string) => {
    setBusy(`save:${id}`);
    setError(null);
    try {
      await apiService.saveSessionState(id);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to save session state');
    } finally {
      setBusy(null);
    }
  }, [refresh]);

  const handleLoad = useCallback(async (id: string) => {
    setBusy(`load:${id}`);
    setError(null);
    try {
      await apiService.loadSessionState(id);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load session state');
    } finally {
      setBusy(null);
    }
  }, [refresh]);

  const handleDelete = useCallback(async (id: string) => {
    if (!window.confirm(`Delete session ${id.slice(0, 8)}…?`)) return;
    setBusy(`delete:${id}`);
    setError(null);
    try {
      await apiService.deleteSession(id);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to delete session');
    } finally {
      setBusy(null);
    }
  }, [refresh]);

  return (
    <div className={`flex flex-col bg-neutral-900/70 border border-neutral-700 rounded-lg ${className}`}>
      <div className="flex items-center justify-between gap-2 p-2 border-b border-neutral-700 bg-neutral-800/70 rounded-t-lg">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-neutral-200">🎮 Sessions</span>
          <span className="text-[10px] text-neutral-500">{sessions.length}</span>
          {activeId && (
            <span className="text-[10px] text-emerald-300 bg-emerald-700/20 px-1.5 py-0.5 rounded">
              active
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={refresh}
          className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-200"
          title="Refresh"
        >
          ↻
        </button>
      </div>

      {/* Create new session */}
      <div className="flex gap-1.5 p-2 border-b border-neutral-800">
        <input
          type="text"
          placeholder="new session name…"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleCreate();
          }}
          className="flex-1 bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs focus:outline-none focus:border-neutral-500"
        />
        <button
          type="button"
          onClick={handleCreate}
          disabled={!newName.trim() || busy === 'new'}
          className="text-xs px-2.5 py-1 rounded bg-emerald-700/50 hover:bg-emerald-700/70 text-emerald-100 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {busy === 'new' ? '…' : '+ new'}
        </button>
      </div>

      {error && (
        <div className="px-3 py-1 text-[11px] text-red-300 bg-red-900/20 border-b border-red-900/30">
          {error}
        </div>
      )}

      {/* Sessions list */}
      <div className="flex-1 overflow-y-auto p-1.5 space-y-1.5" style={{ minHeight: 80 }}>
        {sessions.length === 0 ? (
          <div className="text-neutral-500 italic text-center py-4 text-xs">
            No sessions yet. Create one above.
          </div>
        ) : (
          sessions.map((s) => {
            const isActive = s.id === activeId;
            return (
              <div
                key={s.id}
                className={`flex items-center gap-1.5 px-2 py-1.5 rounded border ${
                  isActive
                    ? 'bg-emerald-900/20 border-emerald-700/40'
                    : 'bg-neutral-900/40 border-neutral-800'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm text-neutral-100 truncate">{s.name}</span>
                    {isActive && <span className="text-[10px] text-emerald-300">●</span>}
                  </div>
                  <div className="text-[10px] text-neutral-500 truncate">
                    {s.rom_name ? `${s.rom_name}` : 'no ROM'}
                    {s.emulator ? ` · ${s.emulator}` : ''}
                    {typeof s.milestones_count === 'number'
                      ? ` · ${s.milestones_count}m / ${s.objectives_count ?? 0}o`
                      : ''}
                  </div>
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  {!isActive && (
                    <button
                      type="button"
                      onClick={() => handleActivate(s.id)}
                      disabled={busy === `activate:${s.id}`}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-blue-700/40 hover:bg-blue-700/60 text-blue-100 disabled:opacity-40"
                      title="Activate session"
                    >
                      ▶
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => handleSave(s.id)}
                    disabled={busy === `save:${s.id}` || !isActive}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-200 disabled:opacity-30"
                    title="Save emulator state to session"
                  >
                    ↓
                  </button>
                  <button
                    type="button"
                    onClick={() => handleLoad(s.id)}
                    disabled={busy === `load:${s.id}`}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-200 disabled:opacity-30"
                    title="Load emulator state from session"
                  >
                    ↑
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDelete(s.id)}
                    disabled={busy === `delete:${s.id}`}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-red-700/30 hover:bg-red-700/50 text-red-200 disabled:opacity-30"
                    title="Delete session"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default SessionsPanel;
