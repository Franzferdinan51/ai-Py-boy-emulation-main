import React from 'react';
import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import GameCanvas from './GameCanvas';

class MockEventSource {
  static instances: MockEventSource[] = [];

  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(public readonly url: string) {
    MockEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  emitOpen() {
    this.onopen?.();
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent<string>);
  }
}

class MockImage {
  onload: (() => void) | null = null;
  private _src = '';

  set src(value: string) {
    this._src = value;
    this.onload?.();
  }

  get src() {
    return this._src;
  }
}

test('renders a server SSE frame without a synthetic type field', async () => {
  const clearRect = vi.fn();
  const drawImage = vi.fn();
  const getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    clearRect,
    drawImage,
  } as unknown as CanvasRenderingContext2D);

  MockEventSource.instances = [];
  vi.stubGlobal('EventSource', MockEventSource as unknown as typeof EventSource);
  vi.stubGlobal('Image', MockImage as unknown as typeof Image);

  render(<GameCanvas backendUrl="http://localhost:5002" showStats={false} />);

  const source = MockEventSource.instances[0];
  expect(source).toBeDefined();

  act(() => {
    source.emitOpen();
  });

  await act(async () => {
    source.emitMessage({
      image: 'ZmFrZS1mcmFtZQ==',
      timestamp: '2026-06-22T10:00:00.000Z',
      frame: 42,
      fps: 59.7,
    });
  });

  expect(clearRect).toHaveBeenCalledWith(0, 0, 160, 144);
  expect(drawImage).toHaveBeenCalledTimes(1);
  expect(source.closed).toBe(false);

  getContext.mockRestore();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

beforeEach(() => {
  MockEventSource.instances = [];
});
