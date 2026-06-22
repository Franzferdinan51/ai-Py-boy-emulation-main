import React, { useEffect, useState, useCallback } from 'react';
import apiService from '../../services/apiService';

interface TelemetryWidgetProps {
  /** Polling interval (ms). Default 3000. */
  pollIntervalMs?: number;
  /** Optional session_id. */
  sessionId?: string;
  className?: string;
}

interface TelemetryData {
  session_id?: string | null;
  stuck_meter: number;
  position_history_len: number;
  last_positions: Array<{ x: number; y: number; map_id?: number; ts: number }>;
  actions_total: number;
  actions_success: number;
  actions_failure: number;
  battles_won: number;
  battles_lost: number;
  blackouts: number;
  party_hp_total?: number | null;
  party_hp_max?: number | null;
  ts: string;
}

function StuckBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const danger = pct >= 80;
  const warn = pct >= 50 && pct < 80;
  const color = danger ? 'bg-red-500' : warn ? 'bg-amber-500' : 'bg-emerald-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden border border-neutral-700">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-[10px] font-mono tabular-nums ${danger ? 'text-red-300' : warn ? 'text-amber-300' : 'text-emerald-300'}`}>
        {pct}
      </span>
    </div>
  );
}

function HpBar({ total, max }: { total: number; max: number }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (total / max) * 100)) : 0;
  const color = pct >= 60 ? 'bg-emerald-500' : pct >= 25 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden border border-neutral-700">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono tabular-nums text-neutral-300">
        {total}/{max}
      </span>
    </div>
  );
}

/**
 * TelemetryWidget — compact panel showing stuck-meter, battle stats,
 * blackouts, party HP, action success rate.
 *
 * Reads /api/agent/telemetry every pollIntervalMs.
 */
const TelemetryWidget: React.FC<TelemetryWidgetProps> = ({
  pollIntervalMs = 3000,
  sessionId,
  className = '',
}) => {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await apiService.getTelemetry(sessionId);
      setData(resp);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Telemetry unavailable');
    }
  }, [sessionId]);

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, pollIntervalMs);
    return () => window.clearInterval(interval);
  }, [refresh, pollIntervalMs]);

  const handleReset = useCallback(async () => {
    try {
      await apiService.resetTelemetry(sessionId);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? 'Failed to reset telemetry');
    }
  }, [sessionId, refresh]);

  if (!data) {
    return (
      <div className={`bg-neutral-900/70 border border-neutral-700 rounded-lg p-3 text-xs text-neutral-500 ${className}`}>
        {error ?? 'Loading telemetry…'}
      </div>
    );
  }

  const successRate = data.actions_total > 0
    ? Math.round((data.actions_success / data.actions_total) * 100)
    : null;

  return (
    <div className={`flex flex-col gap-2 bg-neutral-900/70 border border-neutral-700 rounded-lg p-2 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold text-neutral-200">📊 Telemetry</span>
        <button
          type="button"
          onClick={handleReset}
          className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-200"
          title="Reset counters"
        >
          reset
        </button>
      </div>

      {/* Stuck meter */}
      <div>
        <div className="flex items-center justify-between text-[10px] text-neutral-400 mb-0.5">
          <span>stuck-meter</span>
          <span>{data.position_history_len} positions tracked</span>
        </div>
        <StuckBar value={data.stuck_meter} />
      </div>

      {/* Actions */}
      <div>
        <div className="flex items-center justify-between text-[10px] text-neutral-400 mb-0.5">
          <span>actions</span>
          {successRate !== null && (
            <span className="font-mono tabular-nums text-neutral-300">
              {data.actions_success}/{data.actions_total} ({successRate}%)
            </span>
          )}
        </div>
        <div className="flex gap-1">
          <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden border border-neutral-700">
            <div
              className="h-full bg-cyan-500"
              style={{ width: `${successRate ?? 0}%` }}
            />
          </div>
        </div>
        {data.actions_failure > 0 && (
          <div className="text-[10px] text-red-300 mt-0.5">
            {data.actions_failure} failure{data.actions_failure !== 1 ? 's' : ''}
          </div>
        )}
      </div>

      {/* Battles + blackouts */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-neutral-900/40 border border-neutral-800 rounded p-1.5">
          <div className="text-[10px] text-neutral-500">battles</div>
          <div className="text-xs font-mono tabular-nums">
            <span className="text-emerald-300">{data.battles_won}W</span>
            {' / '}
            <span className="text-red-300">{data.battles_lost}L</span>
          </div>
        </div>
        <div className="bg-neutral-900/40 border border-neutral-800 rounded p-1.5">
          <div className="text-[10px] text-neutral-500">blackouts</div>
          <div className="text-xs font-mono tabular-nums text-red-300">
            {data.blackouts}
          </div>
        </div>
      </div>

      {/* Party HP */}
      {data.party_hp_max !== undefined && data.party_hp_max !== null && data.party_hp_total !== undefined && data.party_hp_total !== null && (
        <div>
          <div className="flex items-center justify-between text-[10px] text-neutral-400 mb-0.5">
            <span>party HP</span>
          </div>
          <HpBar total={data.party_hp_total} max={data.party_hp_max} />
        </div>
      )}

      {error && (
        <div className="text-[10px] text-amber-300 italic">{error}</div>
      )}
    </div>
  );
};

export default TelemetryWidget;
