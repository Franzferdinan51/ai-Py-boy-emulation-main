import React, { useEffect, useState, useCallback } from 'react';
import { Users, MessageCircle, Loader2 } from 'lucide-react';

interface NPCData {
  npcs: NPC[];
  timestamp: string;
}

interface NPC {
  id: number;
  name: string;
  type: string;
  location: { x: number; y: number };
  dialogue?: string;
  interactable: boolean;
}

interface NPCPanelProps {
  backendUrl: string;
  isRomLoaded: boolean;
}

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const NPCPanel: React.FC<NPCPanelProps> = ({ backendUrl, isRomLoaded }) => {
  const [npcData, setNpcData] = useState<NPCData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNpc, setSelectedNpc] = useState<NPC | null>(null);

  const fetchNPCData = useCallback(async () => {
    if (!isRomLoaded || !backendUrl) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/npc/nearby`);
      if (!response.ok) {
        throw new Error(`Failed to fetch NPC data: ${response.status}`);
      }
      const data = await response.json();
      setNpcData(data);
    } catch (err) {
      // If endpoint doesn't exist, show empty state
      setNpcData({
        npcs: [],
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  }, [backendUrl, isRomLoaded]);

  useEffect(() => {
    if (!isRomLoaded) {
      setNpcData(null);
      setError(null);
      setSelectedNpc(null);
      return undefined;
    }

    void fetchNPCData();
    const intervalId = window.setInterval(fetchNPCData, 5000);

    return () => window.clearInterval(intervalId);
  }, [isRomLoaded, fetchNPCData]);

  const getNpcTypeLabel = (type: string) => {
    switch (type.toLowerCase()) {
      case 'trainer':
        return 'Trainer';
      case 'shopkeeper':
        return 'Shop';
      case 'npc':
        return 'NPC';
      case 'healer':
        return 'Healer';
      default:
        return type || 'Character';
    }
  };

  return (
    <section className="data-panel npc-panel">
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">Characters</span>
          <h3 className="data-panel__title">Nearby NPCs</h3>
          <p className="data-panel__subtitle">
            {isRomLoaded
              ? npcData?.npcs?.length
                ? `${npcData.npcs.length} character${npcData.npcs.length === 1 ? '' : 's'} detected`
                : 'Scanning for NPCs...'
              : 'Load a ROM to detect NPCs.'}
          </p>
        </div>
        {loading ? (
          <Loader2 className="data-panel__icon data-panel__icon--spin" />
        ) : (
          <Users className="data-panel__icon" />
        )}
      </div>

      <div className="data-panel__body npc-panel__body">
        {!isRomLoaded ? (
          <div className="empty-panel">Load a ROM to detect nearby NPCs.</div>
        ) : (
          <>
            {error && <div className="error-banner">{error}</div>}

            {!error && (!npcData?.npcs || npcData.npcs.length === 0) && (
              <div className="empty-panel">No NPCs detected nearby.</div>
            )}

            <div className="npc-list">
              {npcData?.npcs?.map((npc) => (
                <button
                  key={`npc-${npc.id}-${npc.name}`}
                  type="button"
                  className={classNames(
                    'npc-card',
                    selectedNpc?.id === npc.id && 'npc-card--selected',
                    !npc.interactable && 'npc-card--disabled'
                  )}
                  onClick={() => setSelectedNpc(selectedNpc?.id === npc.id ? null : npc)}
                  disabled={!npc.interactable}
                >
                  <div className="npc-card__header">
                    <div className="npc-card__info">
                      <strong className="npc-card__name">{npc.name || 'Unknown'}</strong>
                      <span className="npc-card__type">{getNpcTypeLabel(npc.type)}</span>
                    </div>
                    {npc.interactable && (
                      <MessageCircle size={16} className="npc-card__icon" />
                    )}
                  </div>

                  <div className="npc-card__location">
                    <span>X: {npc.location?.x ?? '--'}, Y: {npc.location?.y ?? '--'}</span>
                  </div>

                  {selectedNpc?.id === npc.id && npc.dialogue && (
                    <div className="npc-card__dialogue">
                      <p>"{npc.dialogue}"</p>
                    </div>
                  )}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
};

export default NPCPanel;