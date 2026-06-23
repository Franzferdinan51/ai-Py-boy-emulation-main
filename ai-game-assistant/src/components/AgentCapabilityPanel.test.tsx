import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, test } from 'vitest';
import AgentCapabilityPanel from './AgentCapabilityPanel';

test('renders capability toolbelt, routines, drafts, and guardrails', () => {
  const installs: string[] = [];
  render(
    <AgentCapabilityPanel
      toolbelt={{
        active_session_id: 'session-123',
        active_routine: 'heal_party',
        available_tools: [
          {
            name: 'get_agent_context',
            access: 'read-only',
            category: 'context',
            backend_route: '/api/agent/context',
            mcp_tool: 'get_agent_context',
            description: 'Get a full structured gameplay snapshot for planning.',
          },
          {
            name: 'act_and_observe',
            access: 'mutating',
            category: 'actions',
            backend_route: '/api/agent/act',
            mcp_tool: 'act_and_observe',
            description: 'Execute one canonical action and return the resulting observation.',
          },
        ],
        tool_groups: {
          context: ['get_agent_context'],
          actions: ['act_and_observe'],
        },
        memory_summary: {
          total_records: 4,
          by_type: { note: 2, control_pattern: 2 },
          latest_by_type: {},
          recent_notes: [{ type: 'note', text: 'Heal before exploring.' }],
          learned_control_patterns: [
            {
              sequence: ['UP', 'A'],
              outcome: 'Enter a doorway from the overworld',
              note: 'Use UP then A near entrances',
              timestamp: '2026-06-23T10:00:00.000Z',
            },
          ],
        },
        next_recommended_action: {
          action: 'FOLLOW_ROUTINE',
          target: 'heal_party',
          reason: 'Continue the active routine: heal_party.',
          source: 'session.active_routine',
        },
        auto_learning_signals: {
          control_patterns_observed: 1,
          suggested_routine_count: 1,
          skill_draft_count: 1,
        },
        timestamp: '2026-06-23T10:00:00.000Z',
      }}
      routines={{
        active_session_id: 'session-123',
        active_routine: 'heal_party',
        routines: [
          {
            id: 'routine-1',
            name: 'heal_party',
            description: 'Visit the Pokemon Center and restore the party.',
            kind: 'playbook',
            origin: 'operator',
            status: 'ready',
            tags: ['healing'],
            steps: [{ action: 'UP', frames: 1 }],
            updated_at: '2026-06-23T10:00:00.000Z',
          },
        ],
        suggested_routines: [
          {
            id: 'learned-1',
            name: 'enter_a_doorway_from_the_overworld',
            kind: 'generated_playbook',
            origin: 'memory',
            status: 'suggested',
            steps: [{ action: 'UP', frames: 1 }, { action: 'A', frames: 1 }],
            summary: 'Enter a doorway from the overworld',
          },
        ],
        skill_drafts: [
          {
            id: 'skill-1',
            name: 'heal_party',
            source: 'routine.upsert',
            status: 'draft',
            summary: 'Visit the Pokemon Center and restore the party.',
          },
        ],
        timestamp: '2026-06-23T10:00:00.000Z',
      }}
      workshop={{
        active_session_id: 'session-123',
        active_routine: 'heal_party',
        workspace_precedence: 'repo-local',
        workspace_skills_root: '/tmp/workspace/skills',
        install_route: '/api/agent/skills/workshop/install',
        draft_count: 1,
        drafts: [
          {
            id: 'skill-1',
            name: 'heal_party',
            source: 'routine.upsert',
            status: 'draft',
            summary: 'Visit the Pokemon Center and restore the party.',
            artifact: {
              frontmatter: {
                name: 'heal-party',
                description: 'Visit the Pokemon Center and restore the party.',
              },
              content: '# heal_party',
              relative_install_dir: 'generated/heal-party',
              install_path: '/tmp/workspace/skills/generated/heal-party/SKILL.md',
              installed: false,
            },
            preview_markdown: '# heal_party',
            preview_excerpt: '# heal_party',
            installed: false,
          },
        ],
        timestamp: '2026-06-23T10:00:00.000Z',
      }}
      onInstallSkillDraft={(draftId) => {
        installs.push(draftId);
      }}
    />,
  );

  expect(screen.getByText('Agent Capability')).toBeDefined();
  expect(screen.getAllByText('heal_party').length).toBeGreaterThanOrEqual(2);
  expect(screen.getByText('FOLLOW_ROUTINE heal_party')).toBeDefined();
  expect(screen.getByText('2 grouped tools')).toBeDefined();
  expect(screen.getByText('enter_a_doorway_from_the_overworld')).toBeDefined();
  expect(screen.getByText('Enter a doorway from the overworld')).toBeDefined();
  expect(screen.getByText('Guardrails')).toBeDefined();
  expect(screen.getByText('Use UP then A near entrances')).toBeDefined();
  expect(screen.getByText('Skill workshop')).toBeDefined();
  expect(screen.getByText('generated/heal-party')).toBeDefined();

  fireEvent.click(screen.getByRole('button', { name: 'Install' }));
  expect(installs).toEqual(['skill-1']);
});

test('renders an empty state without crashing', () => {
  render(<AgentCapabilityPanel toolbelt={null} routines={null} workshop={null} actionError="Install failed" />);

  expect(screen.getByText('Agent Capability')).toBeDefined();
  expect(screen.getByText('Waiting for capability snapshots.')).toBeDefined();
  expect(screen.getByText('No routines yet')).toBeDefined();
  expect(screen.getByText('Install failed')).toBeDefined();
});

afterEach(() => {
  cleanup();
});
