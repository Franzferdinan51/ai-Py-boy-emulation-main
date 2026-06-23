import React from 'react';
import type { AgentRunEvent, AgentStateSnapshot } from '../../services/apiService';

interface AgentRunPanelProps {
  agentState: AgentStateSnapshot | null;
  events: AgentRunEvent[];
  error?: string | null;
  className?: string;
}

function formatTime(timestamp: string | null | undefined) {
  if (!timestamp) {
    return 'n/a';
  }

  const value = new Date(timestamp);
  if (Number.isNaN(value.getTime())) {
    return timestamp;
  }

  return value.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function extractText(value: unknown): string | null {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed || null;
  }

  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Record<string, unknown>;
  for (const key of ['summary', 'message', 'title', 'text', 'description', 'result']) {
    const next = record[key];
    const text = extractText(next);
    if (text) {
      return text;
    }
  }

  return null;
}

function describePayload(value: unknown) {
  const text = extractText(value);
  if (text) {
    return text;
  }

  if (!value || typeof value !== 'object') {
    return null;
  }

  try {
    const json = JSON.stringify(value);
    return json && json !== '{}' ? json : null;
  } catch {
    return null;
  }
}

function formatEventHeadline(event: AgentRunEvent) {
  const parts: string[] = [event.kind || 'event'];
  if (event.source) {
    parts.push(event.source);
  }
  if (event.success === true) {
    parts.push('ok');
  } else if (event.success === false) {
    parts.push('failed');
  }
  return parts.join(' · ');
}

function formatEventSummary(event: AgentRunEvent) {
  const action = describePayload(event.action);
  const observation = describePayload(event.observation);
  const changes = describePayload(event.changes);
  const pieces = [action ? `Action: ${action}` : null, observation ? `Observation: ${observation}` : null, changes ? `Changes: ${changes}` : null].filter(
    (piece): piece is string => Boolean(piece),
  );
  return pieces.length ? pieces.join(' · ') : 'No details recorded.';
}

function EventRow({ event, isLatest = false }: { event: AgentRunEvent; isLatest?: boolean }) {
  return (
    <div className={`rounded-md border px-2 py-1.5 ${isLatest ? 'border-amber-500/30 bg-amber-500/10' : 'border-neutral-800 bg-neutral-900/50'}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[11px] font-semibold text-neutral-100">
            {formatEventHeadline(event)}
          </div>
          <div className="truncate text-[10px] text-neutral-500">
            {formatTime(event.timestamp)}
            {event.session_id ? ` · ${event.session_id}` : ''}
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-neutral-300">
          {event.kind || 'event'}
        </span>
      </div>
      <div className="mt-1 text-[10px] leading-4 text-neutral-300">
        {formatEventSummary(event)}
      </div>
    </div>
  );
}

const AgentRunPanel: React.FC<AgentRunPanelProps> = ({
  agentState,
  events,
  error = null,
  className = '',
}) => {
  const recentEvents = Array.isArray(events) ? events.slice(0, 5) : [];
  const latestEvent = recentEvents[0] ?? null;
  const goal = (agentState?.current_goal || '').trim();
  const task = (agentState?.current_task || '').trim();
  const mode = (agentState?.mode || '').trim();
  const enabled = agentState?.enabled;

  return (
    <section className={`flex flex-col gap-2 rounded-lg border border-neutral-700 bg-neutral-900/70 p-2 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-neutral-100">Agent Run</div>
          <div className="text-[10px] text-neutral-500">Canonical run ledger and live agent state</div>
        </div>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${enabled ? 'bg-emerald-500/15 text-emerald-300' : 'bg-neutral-700/60 text-neutral-300'}`}>
          {enabled ? 'enabled' : 'disabled'}
        </span>
      </div>

      {error ? (
        <div className="rounded-md border border-red-900/40 bg-red-950/30 px-2 py-1 text-[10px] text-red-200">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Goal</div>
          <div className="mt-0.5 text-xs text-neutral-100">
            {goal || 'No goal set'}
          </div>
        </div>
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Task</div>
          <div className="mt-0.5 text-xs text-neutral-100">
            {task || 'No task set'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Mode</div>
          <div className="mt-0.5 text-xs text-neutral-100">
            {mode || 'manual'}
          </div>
        </div>
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">State</div>
          <div className="mt-0.5 text-xs text-neutral-100">
            {enabled ? 'enabled' : 'disabled'}
          </div>
        </div>
      </div>

      <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-neutral-500">Latest event</div>
            <div className="text-xs text-neutral-100">
              {latestEvent ? formatEventHeadline(latestEvent) : 'No run events yet.'}
            </div>
          </div>
          <div className="text-right text-[10px] text-neutral-500">
            <div>{latestEvent ? formatTime(latestEvent.timestamp) : 'Waiting'}</div>
            {latestEvent?.source ? <div>{latestEvent.source}</div> : null}
          </div>
        </div>
        {latestEvent ? (
          <div className="mt-1 text-[10px] leading-4 text-neutral-300">
            {formatEventSummary(latestEvent)}
          </div>
        ) : (
          <div className="mt-1 text-[10px] text-neutral-500">No ledger entries have been recorded yet.</div>
        )}
      </div>

      <div>
        <div className="mb-1 flex items-center justify-between gap-2 text-[10px] text-neutral-500">
          <span>Recent events</span>
          <span>{recentEvents.length}</span>
        </div>
        {recentEvents.length === 0 ? (
          <div className="rounded-md border border-dashed border-neutral-800 bg-neutral-900/30 px-2 py-2 text-[10px] text-neutral-500">
            No run events yet.
          </div>
        ) : (
          <div className="space-y-1">
            {recentEvents.map((event, index) => (
              <EventRow key={`${event.timestamp}-${event.kind}-${index}`} event={event} isLatest={index === 0} />
            ))}
          </div>
        )}
      </div>

      {!agentState ? (
        <div className="rounded-md border border-dashed border-neutral-800 bg-neutral-900/30 px-2 py-1 text-[10px] text-neutral-500">
          {error ? 'Agent state unavailable' : 'Waiting for agent state.'}
        </div>
      ) : null}
    </section>
  );
};

export default AgentRunPanel;
