import React, { useEffect, useState } from 'react';
import { Activity, Heart, Shield } from 'lucide-react';
import apiService, { type PartyData, type Pokemon } from '../../services/apiService';

interface PartyPanelProps {
  isRomLoaded: boolean;
  onPartyUpdate?: (party: PartyData) => void;
}

const PARTY_REFRESH_MS = 5000;

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const getHpPercent = (pokemon: Pokemon | null | undefined): number => {
  if (!pokemon) {
    return 0;
  }

  if (typeof pokemon.hp_percent === 'number') {
    return Math.max(0, Math.min(100, pokemon.hp_percent));
  }

  const hp = pokemon.hp ?? 0;
  const maxHp = pokemon.max_hp ?? 1;

  if (maxHp <= 0) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round((hp / maxHp) * 100)));
};

const getHpTone = (hpPercent: number): 'good' | 'warn' | 'bad' => {
  if (hpPercent > 50) {
    return 'good';
  }

  if (hpPercent > 20) {
    return 'warn';
  }

  return 'bad';
};

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

    void fetchParty();
    const intervalId = window.setInterval(() => {
      void fetchParty();
    }, PARTY_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isRomLoaded, onPartyUpdate]);

  const partyCount = partyData?.party_count ?? 0;
  const partyMembers = partyData?.party ?? [];

  return (
    <section className="data-panel party-panel">
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">Party</span>
          <h3 className="data-panel__title">Active Team</h3>
          <p className="data-panel__subtitle">
            {isRomLoaded
              ? partyData
                ? `${partyCount}/6 Pokémon detected`
                : 'Reading party data...'
              : 'Load a ROM to inspect the active party.'}
          </p>
        </div>
        {loading ? <Activity className="data-panel__icon data-panel__icon--spin" /> : <Heart className="data-panel__icon" />}
      </div>

      <div className="data-panel__body party-panel__body">
        {!isRomLoaded ? (
          <div className="empty-panel">Load a ROM to inspect the active party.</div>
        ) : (
          <>
            {error && <div className="error-banner">{error}</div>}

            {!error && partyMembers.length === 0 && (
              <div className="empty-panel">No Pokémon detected in the party yet.</div>
            )}

            <div className="pokemon-list">
              {partyMembers.map((pokemon) => (
                <PartyCard key={`pokemon-${pokemon.slot}-${pokemon.species_name ?? 'unknown'}`} pokemon={pokemon} />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
};

const PartyCard: React.FC<{ pokemon: Pokemon }> = ({ pokemon }) => {
  const hpPercent = getHpPercent(pokemon);
  const hpTone = getHpTone(hpPercent);
  const typeLabels = [pokemon.type1, pokemon.type2].filter((t): t is string => Boolean(t));
  const moves = pokemon.moves ?? [];
  const speciesName = pokemon.species_name ?? 'Unknown';
  const level = pokemon.level ?? '--';
  const hp = pokemon.hp ?? '--';
  const maxHp = pokemon.max_hp ?? '--';
  const statusText = pokemon.status_text ?? 'Healthy';
  const hasStatus = pokemon.status != null && pokemon.status > 0;

  return (
    <article className="pokemon-card">
      <div className="pokemon-card__top">
        <div>
          <div className="pokemon-card__name-row">
            <span className="pokemon-card__slot">Slot {pokemon.slot}</span>
            <strong className="pokemon-card__name">{speciesName}</strong>
          </div>
          <span className="pokemon-card__level">Level {level}</span>
        </div>

        {typeLabels.length > 0 && (
          <div className="pokemon-card__types">
            {typeLabels.map((typeName) => (
              <span key={`type-${pokemon.slot}-${typeName}`} className="type-chip">
                {typeName}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="pokemon-card__meter">
        <div className="pokemon-card__meter-row">
          <span className="pokemon-card__meter-label">
            <Heart className="pokemon-card__meter-icon" />
            HP
          </span>
          <strong className={classNames('pokemon-card__meter-value', `pokemon-card__meter-value--${hpTone}`)}>
            {hp} / {maxHp} ({hpPercent}%)
          </strong>
        </div>
        <div className="hp-meter">
          <div
            className={classNames('hp-meter__fill', `hp-meter__fill--${hpTone}`)}
            style={{ width: `${hpPercent}%` }}
          />
        </div>
      </div>

      <div className="pokemon-card__status">
        <span className="pokemon-card__meter-label">
          <Shield className="pokemon-card__meter-icon" />
          Status
        </span>
        <strong className={classNames(
          'pokemon-card__status-value',
          hasStatus && 'pokemon-card__status-value--alert'
        )}>
          {statusText}
        </strong>
      </div>

      <div className="pokemon-card__moves">
        <span className="pokemon-card__moves-label">Moves</span>
        <div className="move-grid">
          {moves.length > 0 ? (
            moves.map((move) => (
              <span key={`move-${pokemon.slot}-${move.id}-${move.name ?? 'unknown'}`} className="move-pill">
                {move.name ?? 'Unknown'}
              </span>
            ))
          ) : (
            <span className="move-pill move-pill--empty">No move data available</span>
          )}
        </div>
      </div>
    </article>
  );
};

export default PartyPanel;