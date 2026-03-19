import React from 'react';
import type { EmulatorMode } from '../types';
import { EmulatorMode as EmulatorModeEnum } from '../types';

interface HeaderProps {
  emulatorMode: EmulatorMode;
  onModeChange: (mode: EmulatorMode) => void;
  onOpenSettings: () => void;
  useEnhancedUI?: boolean;
  onToggleEnhancedUI?: () => void;
  agentMode?: boolean;
}

const Header: React.FC<HeaderProps> = ({ emulatorMode, onModeChange, onOpenSettings, useEnhancedUI = false, onToggleEnhancedUI, agentMode = true }) => {
  const buttonBaseClasses = 'px-4 py-1 rounded-md text-sm font-semibold transition-all duration-300';
  const activeButtonClasses = 'bg-cyan-glow text-neutral-900 shadow-lg shadow-cyan-glow/30';
  const inactiveButtonClasses = 'bg-neutral-800 hover:bg-neutral-700';

  return (
    <header className="flex items-center justify-between p-4 bg-neutral-900/50 backdrop-blur-sm border-b border-neutral-800">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-bold font-display text-gray-100 tracking-wider">
          AI Game <span className="text-cyan-glow">Assistant</span>
        </h1>
        {/* Agent Mode Badge - Dynamic */}
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border ${
          agentMode 
            ? 'bg-green-500/20 border-green-500/40' 
            : 'bg-neutral-600/20 border-neutral-500/40'
        }`}>
          <div className={`w-2 h-2 rounded-full ${agentMode ? 'bg-green-500 animate-pulse' : 'bg-neutral-500'}`} />
          <span className={`text-xs font-semibold ${agentMode ? 'text-green-400' : 'text-neutral-400'}`}>
            {agentMode ? '🤖 Agent Active' : '👤 Manual Mode'}
          </span>
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 p-1 bg-neutral-900 rounded-lg">
          <button
            onClick={() => onModeChange(EmulatorModeEnum.GB)}
            className={`${buttonBaseClasses} ${emulatorMode === EmulatorModeEnum.GB ? activeButtonClasses : inactiveButtonClasses}`}
          >
            Game Boy
          </button>
          <button
            onClick={() => onModeChange(EmulatorModeEnum.GBA)}
            className={`${buttonBaseClasses} ${emulatorMode === EmulatorModeEnum.GBA ? activeButtonClasses : inactiveButtonClasses}`}
          >
            GBA
          </button>
        </div>

        {/* Enhanced UI Toggle */}
        {onToggleEnhancedUI && (
          <div className="flex items-center space-x-2">
            <span className="text-sm text-neutral-400">Enhanced UI</span>
            <button
              onClick={onToggleEnhancedUI}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                useEnhancedUI ? 'bg-cyan-500' : 'bg-neutral-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  useEnhancedUI ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        )}

        <button onClick={onOpenSettings} className="text-neutral-400 hover:text-white transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.438.995s.145.755.438.995l1.003.827c.48.398.668 1.03.26 1.431l-1.296 2.247a1.125 1.125 0 01-1.37.49l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.063-.374-.313-.686-.645-.87a6.52 6.52 0 01-.22-.127c-.324-.196-.72-.257-1.075-.124l-1.217.456a1.125 1.125 0 01-1.37-.49l-1.296-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.437-.995s-.145-.755-.437-.995l-1.004-.827a1.125 1.125 0 01-.26-1.431l1.296-2.247a1.125 1.125 0 011.37-.49l1.217.456c.355.133.75.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>
    </header>
  );
};

export default Header;