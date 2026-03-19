import React, { useState, useCallback, useEffect } from 'react';
import { Target, Lightbulb, Zap, Clock, Loader2 } from 'lucide-react';

interface StrategyGoal {
  id: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  status: 'active' | 'pending' | 'completed';
  progress?: number;
}

interface StrategyData {
  current_objective: string;
  recommended_action: string;
  confidence: number;
  goals: StrategyGoal[];
  last_update: string;
  timestamp: string;
}

interface StrategyPanelProps {
  backendUrl: string;
  isRomLoaded: boolean;
  agentEnabled?: boolean;
}

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const StrategyPanel: React.FC<StrategyPanelProps> = ({ backendUrl, isRomLoaded, agentEnabled }) => {
  const [strategyData, setStrategyData] = useState<StrategyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStrategyData = useCallback(async () => {
    if (!isRomLoaded || !backendUrl) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/agent/strategy`);
      if (!response.ok) {
        throw new Error(`Failed to fetch strategy: ${response.status}`);
      }
      const data = await response.json();
      setStrategyData(data);
    } catch (err) {
      // If endpoint doesn't exist, show default state
      setStrategyData({
        current_objective: agentEnabled ? 'Explore and progress' : 'Manual control active',
        recommended_action: 'Awaiting input',
        confidence: 0,
        goals: [],
        last_update: new Date().toISOString(),
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  }, [backendUrl, isRomLoaded, agentEnabled]);

  useEffect(() => {
    if (!isRomLoaded) {
      setStrategyData(null);
      setError(null);
      return undefined;
    }

    void fetchStrategyData();
    const intervalId = window.setInterval(fetchStrategyData, 5000);

    return () => window.clearInterval(intervalId);
  }, [isRomLoaded, fetchStrategyData]);

  const getPriorityColor = (priority: StrategyGoal['priority']) => {
    switch (priority) {
      case 'high':
        return 'var(--berry-strong)';
      case 'medium':
        return 'var(--amber)';
      case 'low':
        return 'var(--olive-strong)';
      default:
        return 'var(--slate)';
    }
  };

  const getStatusIcon = (status: StrategyGoal['status']) => {
    switch (status) {
      case 'active':
        return <Zap size={14} className="strategy-goal__icon strategy-goal__icon--active" />;
      case 'completed':
        return <span className="strategy-goal__check">✓</span>;
      default:
        return <Clock size={14} className="strategy-goal__icon" />;
    }
  };

  const confidencePercent = Math.round((strategyData?.confidence ?? 0) * 100);

  return (
    <section className="data-panel strategy-panel">
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">AI Strategy</span>
          <h3 className="data-panel__title">Current Plan</h3>
          <p className="data-panel__subtitle">
            {isRomLoaded
              ? strategyData?.current_objective || 'Analyzing game state...'
              : 'Load a ROM to see AI strategy.'}
          </p>
        </div>
        {loading ? (
          <Loader2 className="data-panel__icon data-panel__icon--spin" />
        ) : (
          <Target className="data-panel__icon" />
        )}
      </div>

      <div className="data-panel__body strategy-panel__body">
        {!isRomLoaded ? (
          <div className="empty-panel">Load a ROM to view AI strategy.</div>
        ) : (
          <>
            {error && <div className="error-banner">{error}</div>}

            {/* Current objective */}
            {strategyData?.current_objective && (
              <div className="strategy-objective">
                <div className="strategy-objective__header">
                  <Lightbulb size={16} />
                  <span className="strategy-objective__label">Current Objective</span>
                </div>
                <p className="strategy-objective__text">{strategyData.current_objective}</p>
              </div>
            )}

            {/* Recommended action */}
            {strategyData?.recommended_action && (
              <div className="strategy-action">
                <span className="strategy-action__label">Recommended</span>
                <strong className="strategy-action__value">{strategyData.recommended_action}</strong>

                {/* Confidence bar */}
                <div className="strategy-confidence">
                  <span className="strategy-confidence__label">Confidence</span>
                  <div className="strategy-confidence__bar">
                    <div
                      className="strategy-confidence__fill"
                      style={{ width: `${confidencePercent}%` }}
                    />
                  </div>
                  <span className="strategy-confidence__value">{confidencePercent}%</span>
                </div>
              </div>
            )}

            {/* Goals list */}
            {strategyData?.goals && strategyData.goals.length > 0 && (
              <div className="strategy-goals">
                <span className="strategy-goals__label">Goals</span>
                <div className="strategy-goals__list">
                  {strategyData.goals.map((goal) => (
                    <div
                      key={goal.id}
                      className={classNames(
                        'strategy-goal',
                        `strategy-goal--${goal.status}`
                      )}
                    >
                      <div className="strategy-goal__header">
                        {getStatusIcon(goal.status)}
                        <strong className="strategy-goal__description">{goal.description}</strong>
                      </div>

                      {goal.status === 'active' && typeof goal.progress === 'number' && (
                        <div className="strategy-goal__progress">
                          <div className="strategy-goal__progress-bar">
                            <div
                              className="strategy-goal__progress-fill"
                              style={{
                                width: `${Math.max(0, Math.min(100, goal.progress))}%`,
                                backgroundColor: getPriorityColor(goal.priority),
                              }}
                            />
                          </div>
                          <span className="strategy-goal__progress-value">{goal.progress}%</span>
                        </div>
                      )}

                      <span
                        className="strategy-goal__priority"
                        style={{ backgroundColor: getPriorityColor(goal.priority) }}
                      >
                        {goal.priority}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Last update */}
            {strategyData?.last_update && (
              <div className="strategy-meta">
                <span>Last updated: {new Date(strategyData.last_update).toLocaleTimeString()}</span>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
};

export default StrategyPanel;