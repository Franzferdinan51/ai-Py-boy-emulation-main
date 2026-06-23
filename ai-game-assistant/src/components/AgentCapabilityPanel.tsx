import React from 'react';
import { Bot, ChevronRight, ShieldAlert, Sparkles } from 'lucide-react';
import type {
  AgentCapabilityMemoryPattern,
  AgentCapabilityRoutine,
  AgentCapabilityRoutinesSnapshot,
  AgentCapabilityToolbeltSnapshot,
} from '../../services/apiService';

interface AgentCapabilityPanelProps {
  toolbelt: AgentCapabilityToolbeltSnapshot | null;
  routines: AgentCapabilityRoutinesSnapshot | null;
  className?: string;
}

function compactText(value: unknown) {
  if (typeof value !== 'string') {
    return null;
  }

  const text = value.trim();
  return text || null;
}

function formatCount(value: number | null | undefined) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '0';
  }

  return new Intl.NumberFormat().format(value);
}

function formatAction(nextAction: AgentCapabilityToolbeltSnapshot['next_recommended_action'] | null | undefined) {
  if (!nextAction) {
    return 'Observe';
  }

  const label = compactText(nextAction.action) || 'Observe';
  const target = compactText(nextAction.target);
  return target ? `${label} ${target}` : label;
}

function summarizeToolGroups(toolbelt: AgentCapabilityToolbeltSnapshot | null) {
  if (!toolbelt) {
    return [];
  }

  const groupedEntries = Object.entries(toolbelt.tool_groups || {});
  if (groupedEntries.length > 0) {
    return groupedEntries.map(([group, toolNames]) => ({
      group,
      count: Array.isArray(toolNames) ? toolNames.length : 0,
      tools: Array.isArray(toolNames) ? toolNames.slice(0, 3) : [],
    }));
  }

  const fallbackGroups = new Map<string, string[]>();
  for (const tool of toolbelt.available_tools || []) {
    const bucket = fallbackGroups.get(tool.category) || [];
    bucket.push(tool.name);
    fallbackGroups.set(tool.category, bucket);
  }

  return Array.from(fallbackGroups.entries()).map(([group, toolNames]) => ({
    group,
    count: toolNames.length,
    tools: toolNames.slice(0, 3),
  }));
}

function summarizeRoutineSteps(routine: AgentCapabilityRoutine) {
  const steps = routine.steps || [];
  if (steps.length === 0) {
    return 'No steps recorded';
  }

  return steps
    .slice(0, 3)
    .map((step) => step.action || 'NOOP')
    .join(' · ');
}

function summarizePattern(pattern: AgentCapabilityMemoryPattern) {
  const sequence = Array.isArray(pattern.sequence) ? pattern.sequence.filter(Boolean) : [];
  const outcome = compactText(pattern.outcome);
  const note = compactText(pattern.note);
  return note || outcome || (sequence.length ? sequence.join(' → ') : 'Pattern recorded');
}

function SectionCard({
  title,
  meta,
  children,
}: {
  title: string;
  meta?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
      <div className="mb-1 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[10px] uppercase tracking-wide text-neutral-500">{title}</div>
          {meta ? <div className="truncate text-[10px] text-neutral-400">{meta}</div> : null}
        </div>
      </div>
      {children}
    </div>
  );
}

const AgentCapabilityPanel: React.FC<AgentCapabilityPanelProps> = ({
  toolbelt,
  routines,
  className = '',
}) => {
  const availableTools = toolbelt?.available_tools || [];
  const toolGroups = summarizeToolGroups(toolbelt);
  const activeRoutine = compactText(toolbelt?.active_routine || routines?.active_routine) || 'No routine active';
  const nextAction = toolbelt?.planner_hint || toolbelt?.next_recommended_action || null;
  const suggestedRoutines = routines?.suggested_routines || [];
  const skillDrafts = routines?.skill_drafts || [];
  const learnedPatterns = toolbelt?.memory_summary?.learned_control_patterns || [];
  const learningSignals = toolbelt?.auto_learning_signals || null;
  const totalToolCount = availableTools.length;
  const groupedToolCount = toolGroups.reduce((sum, group) => sum + group.count, 0);
  const sessionId = compactText(toolbelt?.active_session_id || routines?.active_session_id);

  return (
    <section className={`flex flex-col gap-2 rounded-lg border border-neutral-700 bg-neutral-900/70 p-2 ${className}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <Bot className="h-3.5 w-3.5 text-neutral-300" />
            <div className="text-sm font-semibold text-neutral-100">Agent Capability</div>
          </div>
          <div className="text-[10px] text-neutral-500">
            Read-only Hermes-inspired toolbelt, routines, drafts, and guardrails
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-neutral-300">
          read-only
        </span>
      </div>

      {!toolbelt && !routines ? (
        <div className="rounded-md border border-dashed border-neutral-800 bg-neutral-900/30 px-2 py-2 text-[10px] text-neutral-500">
          Waiting for capability snapshots.
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Active routine</div>
          <div className="mt-0.5 truncate text-xs text-neutral-100">{activeRoutine}</div>
          {sessionId ? <div className="mt-0.5 truncate text-[10px] text-neutral-500">{sessionId}</div> : null}
        </div>
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Planner hint</div>
          <div className="mt-0.5 truncate text-xs text-neutral-100">{formatAction(nextAction)}</div>
          <div className="mt-0.5 line-clamp-2 text-[10px] leading-4 text-neutral-400">
            {compactText(nextAction?.reason) || 'No planner hint yet'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Tools</div>
          <div className="mt-0.5 text-xs text-neutral-100">{formatCount(totalToolCount)}</div>
          <div className="mt-0.5 text-[10px] text-neutral-500">{formatCount(toolGroups.length)} groups</div>
        </div>
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Grouped</div>
          <div className="mt-0.5 text-xs text-neutral-100">{formatCount(groupedToolCount)}</div>
          <div className="mt-0.5 text-[10px] text-neutral-500">by category</div>
        </div>
        <div className="rounded-md border border-neutral-800 bg-neutral-900/40 px-2 py-1.5">
          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Signals</div>
          <div className="mt-0.5 text-xs text-neutral-100">{formatCount(learningSignals?.control_patterns_observed)}</div>
          <div className="mt-0.5 text-[10px] text-neutral-500">patterns observed</div>
        </div>
      </div>

      <SectionCard
        title="Toolbelt"
        meta={toolGroups.length ? `${toolGroups.length} grouped tools` : `${formatCount(totalToolCount)} tools available`}
      >
        {toolGroups.length === 0 ? (
          <div className="text-[10px] text-neutral-500">No tool groups available yet.</div>
        ) : (
          <div className="space-y-1">
            {toolGroups.slice(0, 4).map((group) => (
              <div key={group.group} className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-[10px] font-semibold text-neutral-200">{group.group}</div>
                  <div className="text-[10px] text-neutral-500">{group.count}</div>
                </div>
                <div className="mt-0.5 truncate text-[10px] text-neutral-400">
                  {group.tools.length ? group.tools.join(' · ') : 'No tools in this group'}
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Routines"
        meta={suggestedRoutines.length ? `${suggestedRoutines.length} suggested` : 'No suggestions yet'}
      >
        <div className="space-y-1.5">
          {routines?.routines?.length ? (
            routines.routines.slice(0, 3).map((routine) => (
              <div key={routine.id} className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-[10px] font-semibold text-neutral-100">{routine.name}</div>
                  <span className="shrink-0 rounded-full bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-neutral-300">
                    {routine.status}
                  </span>
                </div>
                <div className="mt-0.5 truncate text-[10px] text-neutral-400">
                  {compactText(routine.description) || compactText(routine.summary) || summarizeRoutineSteps(routine)}
                </div>
              </div>
            ))
          ) : (
            <div className="text-[10px] text-neutral-500">No routines yet</div>
          )}

          {suggestedRoutines.length ? (
            <div className="space-y-1">
              {suggestedRoutines.slice(0, 3).map((routine) => (
                <div key={routine.id} className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <ChevronRight className="h-3 w-3 text-neutral-500" />
                    <div className="truncate text-[10px] font-semibold text-neutral-100">{routine.name}</div>
                  </div>
                  <div className="mt-0.5 truncate text-[10px] text-neutral-400">
                    {compactText(routine.summary) || summarizeRoutineSteps(routine)}
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </SectionCard>

      <SectionCard
        title="Skill drafts"
        meta={skillDrafts.length ? `${skillDrafts.length} draft${skillDrafts.length === 1 ? '' : 's'}` : 'No drafts yet'}
      >
        {skillDrafts.length === 0 ? (
          <div className="text-[10px] text-neutral-500">No skill drafts yet.</div>
        ) : (
          <div className="space-y-1">
            {skillDrafts.slice(0, 4).map((draft) => (
              <div key={draft.id} className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3 text-amber-300" />
                  <div className="truncate text-[10px] font-semibold text-neutral-100">{draft.name}</div>
                </div>
                <div className="mt-0.5 truncate text-[10px] text-neutral-400">
                  {compactText(draft.summary) || compactText(draft.source) || draft.status}
                </div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Guardrails"
        meta={learnedPatterns.length ? `${learnedPatterns.length} learned patterns` : 'No learned patterns yet'}
      >
        {learnedPatterns.length === 0 && !learningSignals ? (
          <div className="text-[10px] text-neutral-500">No failure-learning or guardrail signals yet.</div>
        ) : (
          <div className="space-y-1">
            {learningSignals ? (
              <div className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5 text-[10px] text-neutral-400">
                <div className="flex items-center gap-1.5 text-neutral-200">
                  <ShieldAlert className="h-3 w-3 text-rose-300" />
                  <span>Learning signals</span>
                </div>
                <div className="mt-0.5">
                  {formatCount(learningSignals.control_patterns_observed)} patterns, {formatCount(learningSignals.suggested_routine_count)} suggestions, {formatCount(learningSignals.skill_draft_count)} drafts
                </div>
              </div>
            ) : null}

            {learnedPatterns.slice(0, 3).map((pattern, index) => (
              <div key={`${pattern.timestamp || 'pattern'}-${index}`} className="rounded border border-neutral-800/80 bg-neutral-950/30 px-2 py-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate text-[10px] font-semibold text-neutral-100">Failure learning</div>
                  <div className="text-[10px] text-neutral-500">{pattern.sequence?.length || 0} steps</div>
                </div>
                <div className="mt-0.5 truncate text-[10px] text-neutral-400">{summarizePattern(pattern)}</div>
              </div>
            ))}
          </div>
        )}
      </SectionCard>
    </section>
  );
};

export default AgentCapabilityPanel;
