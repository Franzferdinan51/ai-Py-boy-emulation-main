import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import AIPanel from './components/AIPanel';
import AgentStatusPanel from './components/AgentStatusPanel';
import GamePanel from './components/GamePanel';
import StatePanel from './components/StatePanel';
import SettingsModal from './components/SettingsModal';
import { ErrorBoundary } from './components/ErrorBoundary';
// FIX: Added PlayerStats to the import list to resolve "Cannot find name" error.
import type { Achievement, PartyMember, Objective, Item, GameAction, AIState, AILog, ChatMessage, AppSettings, AiModel, PlayerStats, MapData, AgentStatus } from './types';
import { AIState as AIStateEnum } from './types';
import { fetchAvailableModels, getAIThoughtAndAction, getChatResponse } from './services/geminiService';
import { getScreen, sendAction, loadRom, saveState as saveGameState, loadState as loadGameState, checkBackendStatus } from './services/backendService';
import { configService } from './services/configService';
import { useInterval } from './hooks/useInterval';

const SAVE_KEY = 'ai-game-assistant-save-v1';
const STREAM_INTERVAL_MS = 250; // approx 4 FPS for the screen stream
const IDLE_STATE_REFRESH_MS = 2000;

const App: React.FC = () => {
  const [aiState, setAiState] = useState<AIState>(AIStateEnum.IDLE);
  const [gameScreenUrl, setGameScreenUrl] = useState<string | null>(null);
  const [actionHistory, setActionHistory] = useState<GameAction[]>([]);
  const [isRomLoaded, setIsRomLoaded] = useState(false);
  
  // Dynamic Game State
  const [party, setParty] = useState<PartyMember[]>([]);
  const [mapInfo, setMapInfo] = useState<MapData>({ name: 'Loading...', coords: [0, 0], pointsOfInterest: []});
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [objectives, setObjectives] = useState<Objective[]>([
    { id: 1, text: 'Complete the tutorial quest', completed: false },
    { id: 2, text: 'Visit the capital city', completed: false },
    { id: 3, text: 'Find the hidden ancient ruins', completed: false },
  ]);
  const [inventory, setInventory] = useState<Item[]>([]);
  const [playerStats, setPlayerStats] = useState<PlayerStats>({ runtime: '00:00:00', steps: 0, money: 0 });
  const [dialogue, setDialogue] = useState('');
  
  // State for AIPanel
  const [aiGoal, setAiGoal] = useState<string>('Defeat the final boss and save the world.');
  const [currentReasoning, setCurrentReasoning] = useState<string>('Please load a ROM to begin.');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  
  // State for Settings
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings>({
    aiActionInterval: 4000,
    backendUrl: 'http://localhost:5000',
    aiProvider: 'google',
    googleApiKey: '',
    openrouterApiKey: '',
    lmStudioUrl: 'http://localhost:1234',
    selectedModel: '',
    // OpenClaw Agent Defaults - Agent is DEFAULT
    agentMode: true, // Agent controls by default
    openclawMcpEndpoint: 'http://localhost:3000/mcp',
    visionModel: 'kimi-k2.5',
    autonomousLevel: 'moderate',
    agentObjectives: 'Complete Pokemon Red, defeat the Elite Four',
    agentPersonality: 'strategic',
  });
  const [availableModels, setAvailableModels] = useState<AiModel[]>([]);
  const [isModelListLoading, setIsModelListLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'saving' | 'saved' | ''>('');
  const [justCompletedObjectiveIds, setJustCompletedObjectiveIds] = useState<Set<number>>(new Set());

  // New state for backend connection
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // OpenClaw Agent Status
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({
    connected: true,
    agentName: 'OpenClaw Agent',
    currentAction: 'Idle',
    lastDecision: 'Initializing...',
    heartbeat: 0,
    decisionHistory: ['Agent initialized in default mode'],
  });


  const currentScreenBlob = useRef<Blob | null>(null);
  const messageIdCounter = useRef(0);
  const prevObjectivesRef = useRef<Objective[]>([]);
  const nextObjectiveId = useRef(objectives.length > 0 ? Math.max(...objectives.map(o => o.id)) + 1 : 1);
  const gameScreenUrlRef = useRef<string | null>(null);
  
  const addChatMessage = useCallback((text: string, sender: 'user' | 'ai' | 'system') => {
    const message: ChatMessage = { id: messageIdCounter.current++, sender, text };
    setChatHistory(prev => [...prev, message]);
  }, []);

  // Effect to detect and handle objective completion with race condition protection
  useEffect(() => {
    const prevObjectives = prevObjectivesRef.current;

    // Prevent race conditions by checking if objectives have actually changed
    const objectivesChanged = JSON.stringify(prevObjectives) !== JSON.stringify(objectives);

    if (!objectivesChanged) return;

    if (prevObjectives.length > 0 || objectives.some(o => o.completed)) {
        const newlyCompletedObjectives = objectives.filter(currentObj => {
            const prevObj = prevObjectives.find(p => p.id === currentObj.id);
            return currentObj.completed && (!prevObj || !prevObj.completed);
        });

        if (newlyCompletedObjectives.length > 0) {
            // Batch all objective completion notifications to prevent race conditions
            const completionTimeouts: NodeJS.Timeout[] = [];

            newlyCompletedObjectives.forEach(obj => {
                addChatMessage(`Objective complete: '${obj.text}'`, 'system');
                setJustCompletedObjectiveIds(prev => new Set(prev).add(obj.id));

                const timeout = setTimeout(() => {
                    setJustCompletedObjectiveIds(prev => {
                        const newSet = new Set(prev);
                        newSet.delete(obj.id);
                        return newSet;
                    });
                }, 2000);

                completionTimeouts.push(timeout);
            });

            // Cleanup function to clear timeouts if component unmounts or dependencies change
            return () => {
                completionTimeouts.forEach(timeout => clearTimeout(timeout));
            };
        }
    }

    prevObjectivesRef.current = objectives;
  }, [objectives, addChatMessage]);

  const handleReorderObjectives = (newObjectives: Objective[]) => {
    setObjectives(newObjectives);
  };

  const handleToggleObjective = (id: number) => {
    setObjectives(prev => prev.map(obj => obj.id === id ? { ...obj, completed: !obj.completed } : obj));
  };

  const handleDeleteObjective = (id: number) => {
    setObjectives(prev => prev.filter(obj => obj.id !== id));
  };

  const handleCreateObjective = (text: string) => {
    if (!text.trim()) return;
    const newObjective: Objective = {
      id: nextObjectiveId.current++,
      text,
      completed: false
    };
    setObjectives(prev => [...prev, newObjective]);
  };

  // Cleanup URL objects on unmount and when dependencies change to prevent memory leaks
  useEffect(() => {
    return () => {
      if (gameScreenUrlRef.current) {
        URL.revokeObjectURL(gameScreenUrlRef.current);
        gameScreenUrlRef.current = null;
      }
      // Also cleanup current blob if it exists
      if (currentScreenBlob.current) {
        currentScreenBlob.current = null;
      }
    };
  }, []);

  // Enhanced cleanup for URL objects when game screen URL changes
  useEffect(() => {
    return () => {
      if (gameScreenUrl) {
        URL.revokeObjectURL(gameScreenUrl);
      }
    };
  }, [gameScreenUrl]);

  // Load state from localStorage on initial mount
  useEffect(() => {
    try {
        const savedStateJSON = localStorage.getItem(SAVE_KEY);
        if (savedStateJSON) {
            const savedState = JSON.parse(savedStateJSON);

            if (savedState.appSettings) setAppSettings(savedState.appSettings);
            if (savedState.aiGoal) setAiGoal(savedState.aiGoal);
            if (savedState.objectives) {
              setObjectives(savedState.objectives);
              nextObjectiveId.current = savedState.objectives.length > 0 ? Math.max(...savedState.objectives.map((o: Objective) => o.id)) + 1 : 1
            }
            if (savedState.chatHistory) setChatHistory(savedState.chatHistory);
            if (savedState.actionHistory) setActionHistory(savedState.actionHistory);
            if (savedState.achievements) setAchievements(savedState.achievements);
            if (savedState.party) setParty(savedState.party);
            if (savedState.inventory) setInventory(savedState.inventory);
            if (savedState.mapInfo) setMapInfo(savedState.mapInfo);
            if (savedState.playerStats) setPlayerStats(savedState.playerStats);

            addChatMessage('Session restored from previous state.', 'system');
        }
    } catch (error) {
        console.error("Failed to load state from localStorage", error);
        addChatMessage('Could not restore previous session.', 'system');
    }
  }, [addChatMessage]);

  // Initialize dynamic configuration and backend URL
  useEffect(() => {
    const initializeConfiguration = async () => {
      try {
        // Try to get optimal backend URL with dynamic port allocation
        const optimalUrl = await configService.getOptimalBackendUrl();

        // Update settings if the URL changed
        if (optimalUrl !== appSettings.backendUrl) {
          setAppSettings(prev => ({
            ...prev,
            backendUrl: optimalUrl
          }));

          addChatMessage(`Connected to backend at: ${optimalUrl}`, 'system');
        }

        // Validate configuration
        try {
          const configResponse = await configService.validateConfiguration(optimalUrl);
          const summary = configService.getConfigurationSummary(configResponse.configuration);

          if (!summary.hasValidConfig) {
            addChatMessage('Configuration validation failed. Check server logs.', 'system');
          } else {
            addChatMessage(`Configuration validated. ${summary.apiKeyCount} API key(s) configured.`, 'system');
          }

          if (summary.warnings.length > 0) {
            summary.warnings.forEach(warning => {
              addChatMessage(`Configuration warning: ${warning}`, 'system');
            });
          }
        } catch (configError) {
          console.warn('Configuration validation failed, using fallback:', configError);
          addChatMessage('Using fallback configuration', 'system');
        }

        // Check backend status with the updated URL
        const status = await checkBackendStatus(optimalUrl);
        if (status.success) {
          setBackendStatus('connected');
          setConnectionError(null);
        } else {
          setBackendStatus('disconnected');
          setConnectionError(status.message);
        }
      } catch (error) {
        console.error('Failed to initialize configuration:', error);
        addChatMessage('Failed to initialize dynamic configuration', 'system');
        setBackendStatus('disconnected');
        setConnectionError('Configuration initialization failed');
      }
    };

    initializeConfiguration();
  }, []); // Run once on mount

  // Autosave state to localStorage
  const saveState = useCallback(() => {
    setSaveStatus('saving');
    try {
        const stateToSave = {
            appSettings,
            aiGoal,
            objectives,
            chatHistory,
            actionHistory,
            achievements,
            party,
            inventory,
            mapInfo,
            playerStats
        };
        localStorage.setItem(SAVE_KEY, JSON.stringify(stateToSave));
        setSaveStatus('saved');
        setTimeout(() => setSaveStatus(''), 2000);
    } catch (error) {
        console.error("Failed to save state to localStorage", error);
        setSaveStatus('');
    }
  }, [appSettings, aiGoal, objectives, chatHistory, actionHistory, achievements, party, inventory, mapInfo, playerStats]);

  useInterval(saveState, 15000); // Save every 15 seconds

  // Global error handler for unhandled promise rejections
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      addChatMessage('An unexpected error occurred. Please try again.', 'system');
      event.preventDefault(); // Prevent the default browser behavior
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, [addChatMessage]);

  const refreshFullGameState = useCallback(async () => {
    if (!isRomLoaded || backendStatus !== 'connected') return;
    try {
      const screenBlob = await getScreen(appSettings.backendUrl);

      // Cleanup previous blob if it exists
      if (currentScreenBlob.current) {
        currentScreenBlob.current = null;
      }

      currentScreenBlob.current = screenBlob;
      const newUrl = URL.createObjectURL(screenBlob);

      setGameScreenUrl(prevUrl => {
        if (prevUrl && prevUrl !== gameScreenUrlRef.current) {
          URL.revokeObjectURL(prevUrl);
        }
        if (gameScreenUrlRef.current && gameScreenUrlRef.current !== prevUrl) {
          URL.revokeObjectURL(gameScreenUrlRef.current);
        }
        gameScreenUrlRef.current = newUrl;
        return newUrl;
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
      setBackendStatus('disconnected');
      setConnectionError(errorMessage);
      setAiState(AIStateEnum.IDLE);
    }
  }, [isRomLoaded, appSettings.backendUrl, backendStatus]);

  // Initial and on-change check for backend connection
  useEffect(() => {
    const connectToBackend = async () => {
        if (!appSettings.backendUrl) {
            setBackendStatus('disconnected');
            setConnectionError("Pyboy Server URL is not set in settings.");
            return;
        }

        setBackendStatus('checking');
        setConnectionError(null);
        
        const result = await checkBackendStatus(appSettings.backendUrl);
        
        if (result.success) {
            setBackendStatus('connected');
            addChatMessage(result.message, 'system');
            // After connecting, try a state refresh in case a ROM is already loaded on the backend.
            refreshFullGameState();
        } else {
            setBackendStatus('disconnected');
            setConnectionError(result.message);
        }
    };
    connectToBackend();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appSettings.backendUrl, addChatMessage]);


  useEffect(() => {
    const loadModels = async () => {
      if (
        (appSettings.aiProvider === 'google' && !appSettings.googleApiKey) ||
        (appSettings.aiProvider === 'openrouter' && !appSettings.openrouterApiKey) ||
        (appSettings.aiProvider === 'lmstudio' && !appSettings.lmStudioUrl)
      ) {
        setAvailableModels([]);
        return;
      }

      setIsModelListLoading(true);
      setAvailableModels([]);
      try {
        const models = await fetchAvailableModels(appSettings);
        setAvailableModels(models);
        if (models.length > 0 && !models.find(m => m.id === appSettings.selectedModel)) {
            setAppSettings(prev => ({...prev, selectedModel: models[0].id }));
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
        console.error("Failed to load models:", error);
        addChatMessage(`Failed to load models for ${appSettings.aiProvider}: ${errorMessage}`, 'system');
      } finally {
        setIsModelListLoading(false);
      }
    };

    loadModels();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appSettings.aiProvider, appSettings.googleApiKey, appSettings.openrouterApiKey, appSettings.lmStudioUrl]);

  
  const refreshScreen = useCallback(async () => {
    if (!isRomLoaded || backendStatus !== 'connected') return;
    try {
        const screenBlob = await getScreen(appSettings.backendUrl);

        // Cleanup previous blob if it exists
        if (currentScreenBlob.current) {
            currentScreenBlob.current = null;
        }

        currentScreenBlob.current = screenBlob;
        const newUrl = URL.createObjectURL(screenBlob);

        setGameScreenUrl(prevUrl => {
            if (prevUrl && prevUrl !== gameScreenUrlRef.current) {
                URL.revokeObjectURL(prevUrl);
            }
            if (gameScreenUrlRef.current && gameScreenUrlRef.current !== prevUrl) {
                URL.revokeObjectURL(gameScreenUrlRef.current);
            }
            gameScreenUrlRef.current = newUrl;
            return newUrl;
        });
    } catch (error) {
        // Only transition to disconnected state on first failure to avoid log spam
        if (backendStatus === 'connected') {
            const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
            setBackendStatus('disconnected');
            setConnectionError(errorMessage);
            addChatMessage(`Connection lost. Screen stream failed.`, 'system');
        }
    }
  }, [isRomLoaded, appSettings.backendUrl, backendStatus, addChatMessage]);

  const aiTick = useCallback(async () => {
    if (aiState !== AIStateEnum.RUNNING || !isRomLoaded) return;

    setAiState(AIStateEnum.THINKING);

    try {
        await refreshFullGameState();

        if (!currentScreenBlob.current) {
            console.error("No screen data available for AI to process.");
            addChatMessage("Could not retrieve game screen.", 'system');
            setAiState(AIStateEnum.RUNNING);
            return;
        }

        const currentGoal = aiGoal || "Explore the world.";
        const { reasoning, action } = await getAIThoughtAndAction(appSettings, currentScreenBlob.current, currentGoal, objectives, actionHistory, mapInfo, dialogue);

        setCurrentReasoning(reasoning);
        setActionHistory(prev => [...prev, action]);

        await sendAction(appSettings.backendUrl, action);

        setAiState(AIStateEnum.RUNNING);

    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
        console.error('AI tick operation failed:', error);
        addChatMessage(`AI operation failed: ${errorMessage}`, 'system');
        setCurrentReasoning(`Error: ${errorMessage}`);
        setAiState(AIStateEnum.RUNNING);

        // Prevent unhandled promise rejection
        return Promise.resolve();
    }
  }, [aiState, actionHistory, refreshFullGameState, aiGoal, objectives, addChatMessage, isRomLoaded, appSettings, mapInfo, dialogue]);

  // Main AI action loop
  useInterval(aiTick, aiState === AIStateEnum.RUNNING && isRomLoaded ? appSettings.aiActionInterval : null);
  
  // Constant screen refresh for visual feedback
  useInterval(refreshScreen, isRomLoaded && aiState !== AIStateEnum.THINKING ? STREAM_INTERVAL_MS : null);

  // Slower state refresh for panels when AI is idle
  useInterval(refreshFullGameState, aiState === AIStateEnum.IDLE && isRomLoaded ? IDLE_STATE_REFRESH_MS : null);

  const handleToggleAI = () => {
    if (!isRomLoaded) return;
    if (aiState === AIStateEnum.IDLE) {
      setActionHistory([]);
      setCurrentReasoning("AI is starting...");
      addChatMessage('AI starting...', 'system');
      setAiState(AIStateEnum.RUNNING);
    } else {
      addChatMessage('AI stopped by user.', 'system');
      setCurrentReasoning("AI stopped by user.");
      setAiState(AIStateEnum.IDLE);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim() || isChatting) return;

    const userMessage = chatInput;
    addChatMessage(userMessage, 'user');
    setChatInput('');
    setIsChatting(true);

    try {
        // FIX: Corrected variable name from 'user_message' to 'userMessage'.
        const aiResponseText = await getChatResponse(appSettings, userMessage);
        addChatMessage(aiResponseText, 'ai');
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
        console.error('Failed to get AI chat response:', error);
        addChatMessage(`AI chat failed: ${errorMessage}`, 'system');
    } finally {
        setIsChatting(false);
    }
  };

  const handleSaveSettings = (newSettings: AppSettings) => {
    if (newSettings.aiProvider !== appSettings.aiProvider) {
      newSettings.selectedModel = '';
    }
    
    // Update agent status when settings change
    if (newSettings.agentMode !== appSettings.agentMode) {
      setAgentStatus(prev => ({
        ...prev,
        lastDecision: newSettings.agentMode ? 'Agent mode enabled' : 'Manual mode enabled',
        decisionHistory: [...prev.decisionHistory, `(${new Date().toLocaleTimeString()}) ${newSettings.agentMode ? 'Agent mode enabled' : 'Manual override'}`].slice(-10),
      }));
      addChatMessage(newSettings.agentMode ? '🤖 Agent Mode enabled' : '🛑 Manual mode enabled', 'system');
    }
    
    setAppSettings(newSettings);
    setIsSettingsOpen(false);
    addChatMessage(`Settings updated. AI Provider: ${newSettings.apiProvider || newSettings.aiProvider}.`, 'system');
  };

  const handleLoadRom = async (file: File) => {
    try {
        await loadRom(appSettings.backendUrl, file);
        setIsRomLoaded(true);
        setCurrentReasoning("Successfully loaded ROM. Ready for instructions.");
        addChatMessage(`Successfully loaded ${file.name}.`, 'system');
        await refreshFullGameState(); // Fetch the first screen and state immediately after loading.
    } catch (error) {
        const message = error instanceof Error ? error.message : "An unknown error occurred.";
        setBackendStatus('disconnected');
        setConnectionError(message);
        addChatMessage(`Failed to load ROM: ${message}`, 'system');
        setIsRomLoaded(false);
        throw error; // Re-throw to let RomLoader component handle its UI state
    }
  };
  
  const handleSaveState = async () => {
    if (!isRomLoaded) return;
    try {
      await saveGameState(appSettings.backendUrl);
      addChatMessage('Game state saved.', 'system');
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred.";
      addChatMessage(`Failed to save state: ${message}`, 'system');
    }
  };

  const handleLoadState = async () => {
    if (!isRomLoaded) return;
    try {
      await loadGameState(appSettings.backendUrl);
      addChatMessage('Game state loaded.', 'system');
      await refreshFullGameState(); // Refresh screen immediately after loading
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred.";
      addChatMessage(`Failed to load state: ${message}`, 'system');
    }
  };

  // Agent control handlers
  const handleAgentTakeover = () => {
    setAppSettings(prev => ({ ...prev, agentMode: true }));
    setAgentStatus(prev => ({
      ...prev,
      currentAction: 'Agent takeover activated',
      lastDecision: 'Agent mode enabled by user',
      decisionHistory: [...prev.decisionHistory, `(${new Date().toLocaleTimeString()}) Agent takeover activated`].slice(-10),
    }));
    addChatMessage('🤖 Agent Takeover: Agent now has full control', 'system');
  };

  const handleManualOverride = () => {
    setAppSettings(prev => ({ ...prev, agentMode: false }));
    setAgentStatus(prev => ({
      ...prev,
      currentAction: 'Manual control',
      lastDecision: 'Human took manual control',
      decisionHistory: [...prev.decisionHistory, `(${new Date().toLocaleTimeString()}) Manual override activated`].slice(-10),
    }));
    addChatMessage('🛑 Manual Override: Human now in control', 'system');
  };

  // Update agent status heartbeat
  useEffect(() => {
    const interval = setInterval(() => {
      setAgentStatus(prev => ({
        ...prev,
        heartbeat: prev.heartbeat + 1000,
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);


  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        console.error('App ErrorBoundary caught an error:', error, errorInfo);
        // Optionally log to error reporting service
      }}
    >
      <div className="h-screen w-full flex flex-col bg-neutral-950 font-sans overflow-hidden">
        <Header
          achievements={achievements}
          onOpenSettings={() => setIsSettingsOpen(true)}
          runtime={playerStats.runtime}
          steps={playerStats.steps}
          saveStatus={saveStatus}
          agentMode={appSettings.agentMode}
        />
        <main className="flex-grow grid grid-cols-1 md:grid-cols-12 gap-4 p-4 min-h-0 overflow-y-auto">
          <div className="md:col-span-4 lg:col-span-3 min-h-0 space-y-4">
            <ErrorBoundary>
              <AIPanel
                aiState={aiState}
                aiGoal={aiGoal}
                currentReasoning={currentReasoning}
                actionHistory={actionHistory}
                chatHistory={chatHistory}
                chatInput={chatInput}
                isChatting={isChatting}
                isRomLoaded={isRomLoaded}
                onScreenText={dialogue}
                onGoalChange={setAiGoal}
                onStart={handleToggleAI}
                onStop={handleToggleAI}
                onChatInputChange={setChatInput}
                onSendMessage={handleSendMessage}
              />
            </ErrorBoundary>
            {/* OpenClaw Agent Status Panel */}
            <ErrorBoundary>
              <AgentStatusPanel
                status={agentStatus}
                isAgentMode={appSettings.agentMode}
                onTakeover={handleAgentTakeover}
                onOverride={handleManualOverride}
              />
            </ErrorBoundary>
          </div>
          <div className="md:col-span-8 lg:col-span-5 min-h-0">
            <ErrorBoundary>
              <GamePanel
                party={party}
                gameScreenUrl={gameScreenUrl}
                lastAction={actionHistory.length > 0 ? actionHistory[actionHistory.length - 1] : null}
                isRomLoaded={isRomLoaded}
                onLoadRom={handleLoadRom}
                onSaveState={handleSaveState}
                onLoadState={handleLoadState}
                backendStatus={backendStatus}
                connectionError={connectionError}
                onOpenSettings={() => setIsSettingsOpen(true)}
              />
            </ErrorBoundary>
          </div>
          <div className="md:col-span-12 lg:col-span-4 min-h-0">
            <ErrorBoundary>
              <StatePanel
                objectives={objectives}
                inventory={inventory}
                inventoryTitle="INVENTORY (Key Items)"
                mapInfo={mapInfo}
                money={playerStats.money}
                onReorderObjectives={handleReorderObjectives}
                achievements={achievements}
                playerStats={playerStats}
                justCompletedObjectiveIds={justCompletedObjectiveIds}
                onToggleObjective={handleToggleObjective}
                onDeleteObjective={handleDeleteObjective}
                onCreateObjective={handleCreateObjective}
              />
            </ErrorBoundary>
          </div>
        </main>
        <ErrorBoundary>
          <SettingsModal
            isOpen={isSettingsOpen}
            onClose={() => setIsSettingsOpen(false)}
            currentSettings={appSettings}
            onSave={handleSaveSettings}
            availableModels={availableModels}
            isModelListLoading={isModelListLoading}
          />
        </ErrorBoundary>
      </div>
    </ErrorBoundary>
  );
};

export default App;