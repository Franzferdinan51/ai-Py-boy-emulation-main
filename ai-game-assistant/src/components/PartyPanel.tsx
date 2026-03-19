import React, { useEffect, useState } from 'react';
import { Activity, Heart, Shield } from 'lucide-react';
import apiService, { type PartyData, type Pokemon } from '../../services/apiService';

interface PartyPanelProps {
  isRomLoaded: boolean;
  onPartyUpdate?: (party: PartyData) => void;
}

const PARTY_REFRESH_MS = 5000;

const getHpPercent = (pokemon: Pokemon) => {
  if (typeof pokemon.hp_percent === 'number') {
    return pokemon.hp_percent;
  }

  if (!pokemon.hp || !pokemon.max_hp) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round((pokemon.hp / pokemon.max_hp) * 100)));
};

const getHpTextColor = (hpPercent: number) => {
  if (hpPercent > 50) {
    return 'text-green-400';
  }

  if (hpPercent > 20) {
    return 'text-yellow-400';
  }

  return 'text-red-400';
};

const getHpBarColor = (hpPercent: number) => {
  if (hpPercent > 50) {
    return 'bg-green-500';
  }

  if (hpPercent > 20) {
    return 'bg-yellow-500';
  }

  return 'bg-red-500';
};

const getStatusTextColor = (status: number | null) => (status && status > 0 ? 'text-purple-400' : 'text-green-400');

const PartyPanel: React.FC<PartyPanelProps> = ({ isRomLoaded, onPartyUpdate }) => {
  const [partyData, setPartyData] = useState<PartyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isRomLoaded) {
      setPartyData(null);
      setError(null);
      return undefined;
    }

    let cancelled = false;

    const fetchParty = async () => {
      setLoading(true);

      try {
        const data = await apiService.getParty();
        if (cancelled) {
          return;
        }

        setPartyData(data);
        setError(null);
        onPartyUpdate?.(data);
      } catch (fetchError) {
        if (cancelled) {
          return;
        }

        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load party');
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchParty();
    const intervalId = window.setInterval(fetchParty, PARTY_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isRomLoaded, onPartyUpdate]);

  if (!isRomLoaded) {
    return (
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5 text-center text-neutral-500">
        <Heart className="mx-auto mb-3 h-8 w-8 opacity-50" />
        <p className="text-sm">Load a ROM to inspect the active party.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70">
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Party</h3>
          <p className="mt-1 text-xs text-neutral-500">
            {partyData ? `${partyData.party_count}/6 detected` : 'Reading party data'}
          </p>
        </div>
        {loading && <Activity className="h-4 w-4 animate-spin text-neutral-500" />}
      </div>

      <div className="space-y-3 p-4">
        {error && (
          <div className="rounded-xl border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        {!error && partyData && partyData.party.length === 0 && (
          <div className="rounded-xl border border-neutral-800 bg-neutral-950/70 px-4 py-6 text-center text-sm text-neutral-500">
            No Pokemon detected in the party yet.
          </div>
        )}

        {partyData?.party.map((pokemon) => (
          <PartyCard key={pokemon.slot} pokemon={pokemon} />
        ))}
      </div>
    </div>
  );
};

const PartyCard: React.FC<{ pokemon: Pokemon }> = ({ pokemon }) => {
  const hpPercent = getHpPercent(pokemon);
  const typeLabels = [pokemon.type1, pokemon.type2].filter(Boolean) as string[];

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[11px] uppercase tracking-wide text-neutral-500">
              Slot {pokemon.slot}
            </span>
            <span className="text-base font-semibold text-white">
              {pokemon.species_name || 'Unknown'}
            </span>
          </div>
          <p className="mt-1 text-sm text-neutral-400">Level {pokemon.level ?? '--'}</p>
        </div>

        <div className="flex flex-wrap justify-end gap-1">
          {typeLabels.map((typeName) => (
            <span
              key={`${pokemon.slot}-${typeName}`}
              className="rounded-full border border-neutral-700 bg-neutral-900 px-2 py-0.5 text-[11px] uppercase tracking-wide text-neutral-300"
            >
              {typeName}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4">
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 text-neutral-400">
            <Heart className="h-3.5 w-3.5" />
            HP
          </span>
          <span className={getHpTextColor(hpPercent)}>
            {pokemon.hp ?? '--'} / {pokemon.max_hp ?? '--'} ({hpPercent}%)
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
          <div
            className={`h-full ${getHpBarColor(hpPercent)}`}
            style={{ width: `${hpPercent}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs">
        <span className="flex items-center gap-1 text-neutral-400">
          <Shield className="h-3.5 w-3.5" />
          Status
        </span>
        <span className={getStatusTextColor(pokemon.status)}>
          {pokemon.status_text || 'Healthy'}
        </span>
      </div>

      <div className="mt-4">
        <p className="mb-2 text-xs uppercase tracking-[0.2em] text-neutral-500">Moves</p>
        <div className="grid grid-cols-2 gap-2">
          {pokemon.moves.length > 0 ? (
            pokemon.moves.map((move) => (
              <div
                key={`${pokemon.slot}-${move.id}-${move.name}`}
                className="rounded-xl border border-neutral-800 bg-neutral-900 px-2 py-1.5 text-xs text-neutral-300"
              >
                {move.name}
              </div>
            ))
          ) : (
            <div className="col-span-2 rounded-xl border border-dashed border-neutral-800 px-3 py-2 text-xs text-neutral-500">
              No move data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PartyPanel;
