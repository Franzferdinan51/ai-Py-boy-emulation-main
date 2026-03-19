import React, { useState, useEffect } from 'react';
import { Heart, Zap, Shield, Sword, Activity } from 'lucide-react';
import type { PartyData, Pokemon } from '../../services/apiService';
import apiService from '../../services/apiService';

interface PartyPanelProps {
  isRomLoaded: boolean;
  onPartyUpdate?: (party: PartyData) => void;
}

const PartyPanel: React.FC<PartyPanelProps> = ({ isRomLoaded, onPartyUpdate }) => {
  const [partyData, setPartyData] = useState<PartyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchParty = async () => {
    if (!isRomLoaded) return;
    
    setLoading(true);
    try {
      const data = await apiService.getParty();
      setPartyData(data);
      setError(null);
      onPartyUpdate?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch party');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isRomLoaded) {
      setPartyData(null);
      return;
    }

    fetchParty();
    const interval = setInterval(fetchParty, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [isRomLoaded]);

  const getHPColor = (hpPercent: number) => {
    if (hpPercent > 50) return 'text-green-400';
    if (hpPercent > 20) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getHPBarColor = (hpPercent: number) => {
    if (hpPercent > 50) return 'bg-green-500';
    if (hpPercent > 20) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getStatusColor = (status: number) => {
    if (status === 0) return 'text-green-400';
    return 'text-purple-400';
  };

  if (!isRomLoaded) {
    return (
      <div className="p-4 text-center text-neutral-500">
        <Heart className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Load a ROM to view party</p>
      </div>
    );
  }

  if (loading && !partyData) {
    return (
      <div className="p-4 text-center text-neutral-400">
        <Activity className="w-6 h-6 mx-auto mb-2 animate-spin" />
        <p className="text-sm">Loading party...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center text-red-400">
        <p className="text-sm">{error}</p>
        <button onClick={fetchParty} className="mt-2 text-xs underline">Retry</button>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-400 uppercase">Party ({partyData?.party_count || 0}/6)</h3>
        <button 
          onClick={fetchParty}
          className="p-1 hover:bg-neutral-800 rounded"
          title="Refresh"
        >
          <Activity className="w-3 h-3" />
        </button>
      </div>

      {partyData && partyData.party.length > 0 ? (
        <div className="space-y-2">
          {partyData.party.map((pokemon) => (
            <PokemonCard key={pokemon.slot} pokemon={pokemon} />
          ))}
        </div>
      ) : (
        <div className="text-center text-neutral-500 py-8">
          <p className="text-sm">No Pokemon in party</p>
          <p className="text-xs mt-1">Start a new game to get your first Pokemon!</p>
        </div>
      )}
    </div>
  );
};

interface PokemonCardProps {
  pokemon: Pokemon;
}

const PokemonCard: React.FC<PokemonCardProps> = ({ pokemon }) => {
  const hpPercent = pokemon.hp_percent || 0;

  return (
    <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700 hover:border-neutral-600 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500">#{pokemon.slot}</span>
          <span className="font-medium text-white">{pokemon.species_name || 'Unknown'}</span>
          <span className="text-xs text-neutral-400">Lv.{pokemon.level}</span>
        </div>
        <div className="flex items-center gap-1">
          {pokemon.type1 && (
            <span className="text-xs px-2 py-0.5 bg-neutral-700 rounded">{pokemon.type1}</span>
          )}
          {pokemon.type2 && (
            <span className="text-xs px-2 py-0.5 bg-neutral-700 rounded">{pokemon.type2}</span>
          )}
        </div>
      </div>

      {/* HP Bar */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-neutral-400 flex items-center gap-1">
            <Heart className="w-3 h-3" /> HP
          </span>
          <span className={getHPColor(hpPercent)}>
            {pokemon.hp}/{pokemon.max_hp} ({hpPercent}%)
          </span>
        </div>
        <div className="h-2 bg-neutral-700 rounded-full overflow-hidden">
          <div 
            className={`h-full ${getHPBarColor(hpPercent)} transition-all duration-300`}
            style={{ width: `${hpPercent}%` }}
          />
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center justify-between text-xs mb-2">
        <span className="text-neutral-400 flex items-center gap-1">
          <Shield className="w-3 h-3" /> Status
        </span>
        <span className={getStatusColor(pokemon.status || 0)}>
          {pokemon.status_text || 'Healthy'}
        </span>
      </div>

      {/* Moves */}
      {pokemon.moves.length > 0 && (
        <div className="grid grid-cols-2 gap-1">
          {pokemon.moves.map((move, idx) => (
            <div key={idx} className="text-xs px-2 py-1 bg-neutral-700 rounded text-neutral-300">
              {move.name}
            </div>
          ))}
        </div>
      )}

      {/* OT ID */}
      {pokemon.ot_id && (
        <div className="mt-2 text-xs text-neutral-500">
          OT ID: {pokemon.ot_id}
        </div>
      )}
    </div>
  );
};

export default PartyPanel;
