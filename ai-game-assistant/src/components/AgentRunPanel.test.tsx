import React from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, test } from 'vitest';
import AgentRunPanel from './AgentRunPanel';
import type { AgentRunEvent, AgentStateSnapshot } from '../../services/apiService';

const agentState: AgentStateSnapshot = {
  mode: 'autonomous',
  enabled: true,
  current_goal: 'Defeat Brock',
  current_task: 'Navigate to Pewter City',
  timestamp: '2026-06-23T10:00:00.000Z',
};

const events: AgentRunEvent[] = [
  {
    kind: 'run_event',
    timestamp: '2026-06-23T10:05:00.000Z',
    source: 'ai',
    success: true,
    action: { action: 'A', frames: 1 },
    observation: { summary: 'Pressed A' },
    changes: { hp: -1 },
    data: { action: { action: 'A', frames: 1 }, success: true },
  },
  {
    kind: 'observation',
    timestamp: '2026-06-23T10:04:00.000Z',
    source: 'system',
    observation: { summary: 'Entered battle' },
    data: { observation: { summary: 'Entered battle' } },
  },
];

test('renders the agent run ledger with state, latest event, and recent events', () => {
  render(
    <AgentRunPanel
      agentState={agentState}
      events={events}
    />,
  );

  expect(screen.getByText('Agent Run')).toBeDefined();
  expect(screen.getByText('Defeat Brock')).toBeDefined();
  expect(screen.getByText('Navigate to Pewter City')).toBeDefined();
  expect(screen.getByText('autonomous')).toBeDefined();
  expect(screen.getAllByText('enabled').length).toBeGreaterThanOrEqual(2);
  expect(screen.getAllByText('run_event').length).toBeGreaterThanOrEqual(1);
  expect(screen.getByText('Latest event')).toBeDefined();
  expect(screen.queryAllByText(/Pressed A/).length).toBeGreaterThanOrEqual(1);
  expect(screen.queryAllByText(/Entered battle/).length).toBeGreaterThanOrEqual(1);
});

test('renders empty and error states clearly', () => {
  const { rerender } = render(
    <AgentRunPanel
      agentState={null}
      events={[]}
      error="Unable to load run events"
    />,
  );

  expect(screen.getByText('Unable to load run events')).toBeDefined();
  expect(screen.getAllByText('No run events yet.').length).toBeGreaterThanOrEqual(2);
  expect(screen.getByText('Agent state unavailable')).toBeDefined();

  rerender(
    <AgentRunPanel
      agentState={null}
      events={[]}
    />,
  );

  expect(screen.getByText('Waiting for agent state.')).toBeDefined();
});

afterEach(() => {
  cleanup();
});
