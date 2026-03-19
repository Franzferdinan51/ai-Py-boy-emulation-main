import React from 'react';
import type { AgentStatus } from '../types';

interface AgentStatusPanelProps {
  status: AgentStatus;
  isAgentMode: boolean;
  onTakeover: () => void;
  onOverride: () => void;
}

const AgentStatusPanel: React.FC<AgentStatusPanelProps> = ({ 
  status, 
  isAgentMode,
  onTakeover,
  onOverride 
}) => {
  const getStatusColor = () => {
    if (!isAgentMode) return 'bg-neutral-600';
    if (!status.connected) return 'bg-red-500';
    return 'bg-green-500';
  };

  const getStatusText = () => {
    if (!isAgentMode) return 'Manual Mode';
    if (!status.connected) return 'Disconnected';
    return 'Agent Active';
  };

  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden">
      {/* Header with Agent Mode Indicator */}
      <div className="flex items-center justify-between p-3 bg-neutral-800/50 border-b border-neutral-700">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`} />
          <span className="font-semibold text-neutral-200">🤖 Agent Status</span>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
          isAgentMode ? 'bg-green-500/20 text-green-400' : 'bg-neutral-600/20 text-neutral-400'
        }`}>
          {getStatusText()}
        </span>
      </div>

      {/* Agent Info */}
      <div className="p-4 space-y-3">
        {/* Connected Agent */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-400">Connected Agent</span>
          <span className="text-sm font-medium text-neutral-200">{status.agentName || 'OpenClaw Agent'}</span>
        </div>

        {/* Current Action */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-400">Current Action</span>
          <span className="text-sm font-medium text-cyan-glow">{status.currentAction || 'Idle'}</span>
        </div>

        {/* Heartbeat */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-400">Heartbeat</span>
          <span className={`text-sm font-mono ${status.heartbeat > 0 ? 'text-green-400' : 'text-neutral-500'}`}>
            {status.heartbeat > 0 ? `${status.heartbeat}ms ago` : 'No data'}
          </span>
        </div>

        {/* Last Decision */}
        <div className="mt-3">
          <span className="text-sm text-neutral-400 block mb-1">Last Decision</span>
          <div className="p-2 bg-neutral-800 rounded text-xs text-neutral-300 font-mono">
            {status.lastDecision || 'Awaiting first decision...'}
          </div>
        </div>

        {/* Decision History */}
        <div className="mt-3">
          <span className="text-sm text-neutral-400 block mb-2">Decision History</span>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {status.decisionHistory.length === 0 ? (
              <span className="text-xs text-neutral-600">No decisions yet</span>
            ) : (
              status.decisionHistory.map((decision, index) => (
                <div key={index} className="text-xs text-neutral-400 font-mono border-l-2 border-neutral-700 pl-2">
                  {decision}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex space-x-2 mt-4 pt-3 border-t border-neutral-700">
          <button
            onClick={onTakeover}
            disabled={isAgentMode}
            className={`flex-1 px-3 py-2 rounded-md text-sm font-semibold transition-all ${
              isAgentMode 
                ? 'bg-neutral-700 text-neutral-500 cursor-not-allowed' 
                : 'bg-cyan-600 hover:bg-cyan-500 text-white'
            }`}
          >
            ⚡ Agent Takeover
          </button>
          <button
            onClick={onOverride}
            disabled={!isAgentMode}
            className={`flex-1 px-3 py-2 rounded-md text-sm font-semibold transition-all ${
              !isAgentMode 
                ? 'bg-neutral-700 text-neutral-500 cursor-not-allowed' 
                : 'bg-orange-600 hover:bg-orange-500 text-white'
            }`}
          >
            🛑 Manual Override
          </button>
        </div>
      </div>
    </div>
  );
};

export default AgentStatusPanel;