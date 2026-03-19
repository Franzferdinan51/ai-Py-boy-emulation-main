import React, { useState, useCallback, useEffect } from 'react';
import type { GameAction } from '../../types';

interface ControllerProps {
  onAction: (action: GameAction) => void;
  disabled?: boolean;
  className?: string;
  layout?: 'standard' | 'compact';
}

const DPad: React.FC<{ onAction: (action: GameAction) => void; disabled?: boolean }> = ({ onAction, disabled }) => {
  const [pressed, setPressed] = useState<GameAction | null>(null);

  const handlePress = useCallback((action: GameAction) => {
    if (disabled) return;
    setPressed(action);
    onAction(action);
  }, [onAction, disabled]);

  const handleRelease = useCallback(() => setPressed(null), []);

  useEffect(() => { if (disabled) setPressed(null); }, [disabled]);

  const dpadButtonClass = (action: GameAction) => `
    absolute flex items-center justify-center w-12 h-12 bg-neutral-800 border-2 border-neutral-600 
    text-neutral-300 transition-all duration-75 select-none
    ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer hover:bg-neutral-700 active:bg-neutral-900'}
    ${pressed === action ? 'bg-cyan-600 border-cyan-400 text-white shadow-lg shadow-cyan-500/30 scale-95' : ''}
  `;

  return (
    <div className="relative w-36 h-36">
      <button className={dpadButtonClass('UP')} style={{ top: 0, left: '50%', transform: 'translateX(-50%)' }}
        onMouseDown={() => handlePress('UP')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Up">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg>
      </button>
      <button className={dpadButtonClass('DOWN')} style={{ bottom: 0, left: '50%', transform: 'translateX(-50%)' }}
        onMouseDown={() => handlePress('DOWN')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Down">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
      </button>
      <button className={dpadButtonClass('LEFT')} style={{ left: 0, top: '50%', transform: 'translateY(-50%)' }}
        onMouseDown={() => handlePress('LEFT')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Left">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
      </button>
      <button className={dpadButtonClass('RIGHT')} style={{ right: 0, top: '50%', transform: 'translateY(-50%)' }}
        onMouseDown={() => handlePress('RIGHT')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Right">
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
      </button>
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ transform: 'translate(25%, 25%)' }}>
        <div className="w-12 h-12 bg-neutral-900 rounded-full border-2 border-neutral-700" />
      </div>
    </div>
  );
};

const ActionButtons: React.FC<{ onAction: (action: GameAction) => void; disabled?: boolean }> = ({ onAction, disabled }) => {
  const [pressed, setPressed] = useState<GameAction | null>(null);
  const handlePress = useCallback((action: GameAction) => { if (disabled) return; setPressed(action); onAction(action); }, [onAction, disabled]);
  const handleRelease = useCallback(() => setPressed(null), []);

  const buttonClass = (action: GameAction, color: string) => `
    absolute flex items-center justify-center w-14 h-14 rounded-full border-2 font-bold text-lg
    transition-all duration-75 select-none
    ${disabled ? 'opacity-40 cursor-not-allowed' : `cursor-pointer hover:brightness-110 active:scale-95`}
    ${pressed === action ? `border-${color}-400 shadow-lg shadow-${color}-500/30 scale-95` : `border-neutral-600`}
  `;

  return (
    <div className="relative w-32 h-28">
      <button className={`${buttonClass('A', 'green')} bg-green-600 text-white`} style={{ right: 0, bottom: 0, borderColor: pressed === 'A' ? '#4ade80' : undefined, boxShadow: pressed === 'A' ? '0 0 15px rgba(74, 222, 128, 0.5)' : undefined }}
        onMouseDown={() => handlePress('A')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="A Button">A</button>
      <button className={`${buttonClass('B', 'red')} bg-red-600 text-white`} style={{ right: 60, bottom: 25, borderColor: pressed === 'B' ? '#f87171' : undefined, boxShadow: pressed === 'B' ? '0 0 15px rgba(248, 113, 113, 0.5)' : undefined }}
        onMouseDown={() => handlePress('B')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="B Button">B</button>
    </div>
  );
};

const MenuButtons: React.FC<{ onAction: (action: GameAction) => void; disabled?: boolean }> = ({ onAction, disabled }) => {
  const [pressed, setPressed] = useState<GameAction | null>(null);
  const handlePress = useCallback((action: GameAction) => { if (disabled) return; setPressed(action); onAction(action); }, [onAction, disabled]);
  const handleRelease = useCallback(() => setPressed(null), []);

  const smallButtonClass = (action: GameAction) => `
    absolute flex items-center justify-center w-10 h-6 bg-neutral-800 border border-neutral-600 
    rounded text-neutral-400 text-xs font-medium transition-all duration-75 select-none
    ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer hover:bg-neutral-700 active:bg-neutral-900'}
    ${pressed === action ? 'bg-cyan-700 border-cyan-500 text-white' : ''}
  `;

  return (
    <div className="relative w-32 h-8">
      <button className={smallButtonClass('SELECT')} style={{ left: 10 }} onMouseDown={() => handlePress('SELECT')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Select">SELECT</button>
      <button className={smallButtonClass('START')} style={{ right: 10 }} onMouseDown={() => handlePress('START')} onMouseUp={handleRelease} onMouseLeave={handleRelease} disabled={disabled} aria-label="Start">START</button>
    </div>
  );
};

const Controller: React.FC<ControllerProps> = ({ onAction, disabled = false, className = '', layout = 'standard' }) => {
  useEffect(() => {
    const keyMap: Record<string, GameAction> = { ArrowUp: 'UP', ArrowDown: 'DOWN', ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', z: 'A', x: 'B', Enter: 'START', Shift: 'SELECT' };
    const handleKeyDown = (e: KeyboardEvent) => { if (disabled) return; const action = keyMap[e.key]; if (action) { e.preventDefault(); onAction(action); } };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onAction, disabled]);

  if (layout === 'compact') {
    return (
      <div className={`flex items-center justify-center gap-8 p-4 bg-neutral-900/80 rounded-lg border border-neutral-700 ${className}`}>
        <DPad onAction={onAction} disabled={disabled} />
        <div className="flex flex-col gap-2">
          <ActionButtons onAction={onAction} disabled={disabled} />
          <MenuButtons onAction={onAction} disabled={disabled} />
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center gap-6 p-6 bg-neutral-900/80 rounded-lg border border-neutral-700 ${className}`}>
      <div className="text-neutral-400 text-sm font-medium">Virtual Controller</div>
      <div className="flex items-center justify-center gap-12">
        <DPad onAction={onAction} disabled={disabled} />
        <div className="flex flex-col items-center gap-4">
          <ActionButtons onAction={onAction} disabled={disabled} />
          <MenuButtons onAction={onAction} disabled={disabled} />
        </div>
      </div>
      <div className="text-neutral-500 text-xs mt-2">
        <span className="text-neutral-400">Keyboard:</span> Arrow Keys: D-Pad | Z: A | X: B | Enter: Start | Shift: Select
      </div>
    </div>
  );
};

export default Controller;