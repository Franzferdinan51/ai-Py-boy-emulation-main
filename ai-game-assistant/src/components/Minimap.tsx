import React, { useEffect, useState, useCallback } from 'react';
import { Map, Navigation, Loader2 } from 'lucide-react';

interface MapLocation {
  name: string;
  x: number;
  y: number;
  type?: 'town' | 'route' | 'cave' | 'building';
}

interface MinimapData {
  current_map: string;
  current_location: { x: number; y: number };
  locations?: MapLocation[];
  timestamp: string;
}

interface MinimapProps {
  backendUrl: string;
  isRomLoaded: boolean;
}

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const Minimap: React.FC<MinimapProps> = ({ backendUrl, isRomLoaded }) => {
  const [mapData, setMapData] = useState<MinimapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMapData = useCallback(async () => {
    if (!isRomLoaded || !backendUrl) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/map/location`);
      if (!response.ok) {
        throw new Error(`Failed to fetch map data: ${response.status}`);
      }
      const data = await response.json();
      setMapData(data);
    } catch (err) {
      // If endpoint doesn't exist, show placeholder
      setMapData({
        current_map: 'Unknown Location',
        current_location: { x: 0, y: 0 },
        locations: [],
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  }, [backendUrl, isRomLoaded]);

  useEffect(() => {
    if (!isRomLoaded) {
      setMapData(null);
      setError(null);
      return undefined;
    }

    void fetchMapData();
    const intervalId = window.setInterval(fetchMapData, 5000);

    return () => window.clearInterval(intervalId);
  }, [isRomLoaded, fetchMapData]);

  const getLocationTypeColor = (type?: string) => {
    switch (type) {
      case 'town':
        return 'var(--olive-strong)';
      case 'route':
        return 'var(--amber)';
      case 'cave':
        return 'var(--slate)';
      case 'building':
        return 'var(--berry-strong)';
      default:
        return 'var(--text-muted)';
    }
  };

  return (
    <section className="data-panel minimap-panel">
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">Navigation</span>
          <h3 className="data-panel__title">World Map</h3>
          <p className="data-panel__subtitle">
            {isRomLoaded
              ? mapData?.current_map || 'Detecting location...'
              : 'Load a ROM to view the map.'}
          </p>
        </div>
        {loading ? (
          <Loader2 className="data-panel__icon data-panel__icon--spin" />
        ) : (
          <Map className="data-panel__icon" />
        )}
      </div>

      <div className="data-panel__body minimap-panel__body">
        {!isRomLoaded ? (
          <div className="empty-panel">Load a ROM to view the world map.</div>
        ) : (
          <>
            {error && <div className="error-banner">{error}</div>}

            <div className="minimap-grid">
              {/* Simplified map visualization */}
              <div className="minimap-canvas">
                <div className="minimap-grid-overlay">
                  {Array.from({ length: 64 }).map((_, i) => (
                    <div key={i} className="minimap-cell" />
                  ))}
                </div>

                {/* Player position indicator */}
                <div
                  className="minimap-player"
                  style={{
                    left: `${((mapData?.current_location?.x ?? 0) % 8) * 12.5}%`,
                    top: `${((mapData?.current_location?.y ?? 0) % 8) * 12.5}%`,
                  }}
                >
                  <Navigation size={12} />
                </div>
              </div>

              {/* Location info */}
              <div className="minimap-info">
                <div className="minimap-info__row">
                  <span className="minimap-info__label">Current Location</span>
                  <strong className="minimap-info__value">
                    {mapData?.current_map || 'Unknown'}
                  </strong>
                </div>

                <div className="minimap-info__row">
                  <span className="minimap-info__label">Coordinates</span>
                  <span className="minimap-info__coords">
                    X: {mapData?.current_location?.x ?? '--'}, Y: {mapData?.current_location?.y ?? '--'}
                  </span>
                </div>
              </div>

              {/* Nearby locations */}
              {mapData?.locations && mapData.locations.length > 0 && (
                <div className="minimap-locations">
                  <span className="minimap-locations__label">Nearby</span>
                  <div className="minimap-locations__list">
                    {mapData.locations.slice(0, 4).map((loc, idx) => (
                      <div
                        key={`loc-${idx}-${loc.name}`}
                        className="minimap-location-chip"
                        style={{ '--location-color': getLocationTypeColor(loc.type) } as React.CSSProperties}
                      >
                        {loc.name}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
};

export default Minimap;