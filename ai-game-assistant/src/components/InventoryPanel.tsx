import React, { useEffect, useState } from 'react';
import { Activity, Backpack, Coins, Package } from 'lucide-react';
import apiService, { type InventoryData, type InventoryItem } from '../../services/apiService';

interface InventoryPanelProps {
  isRomLoaded: boolean;
  onInventoryUpdate?: (inventory: InventoryData) => void;
}

const INVENTORY_REFRESH_MS = 10000;

const classNames = (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' ');

const getCategory = (itemName: string) => {
  const label = itemName.toLowerCase();

  if (label.includes('ball')) {
    return 'pokeball';
  }

  if (label.includes('potion') || label.includes('heal') || label.includes('revive') || label.includes('antidote')) {
    return 'medicine';
  }

  if (label.includes('badge')) {
    return 'badge';
  }

  if (label.includes('key') || label.includes('ticket') || label.includes('pass')) {
    return 'key';
  }

  if (label.includes('rod')) {
    return 'fishing';
  }

  if (label.includes('stone')) {
    return 'stone';
  }

  return 'general';
};

const getCategoryLabel = (category: string) => {
  switch (category) {
    case 'pokeball':
      return 'Poke Ball';
    case 'medicine':
      return 'Medicine';
    case 'badge':
      return 'Badge';
    case 'key':
      return 'Key item';
    case 'fishing':
      return 'Fishing';
    case 'stone':
      return 'Stone';
    default:
      return 'General';
  }
};

const InventoryPanel: React.FC<InventoryPanelProps> = ({ isRomLoaded, onInventoryUpdate }) => {
  const [inventoryData, setInventoryData] = useState<InventoryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isRomLoaded) {
      setInventoryData(null);
      setError(null);
      return undefined;
    }

    let cancelled = false;

    const fetchInventory = async () => {
      setLoading(true);

      try {
        const data = await apiService.getInventory();
        if (cancelled) {
          return;
        }

        setInventoryData(data);
        setError(null);
        onInventoryUpdate?.(data);
      } catch (fetchError) {
        if (cancelled) {
          return;
        }

        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load inventory');
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void fetchInventory();
    const intervalId = window.setInterval(() => {
      void fetchInventory();
    }, INVENTORY_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isRomLoaded, onInventoryUpdate]);

  return (
    <section className="data-panel">
      <div className="data-panel__header">
        <div>
          <span className="data-panel__eyebrow">Inventory</span>
          <h3 className="data-panel__title">Bag contents</h3>
          <p className="data-panel__subtitle">
            {isRomLoaded
              ? inventoryData
                ? `${inventoryData.item_count} tracked items`
                : 'Reading bag data'
              : 'Load a ROM to inspect the bag.'}
          </p>
        </div>
        {loading ? <Activity className="data-panel__icon data-panel__icon--spin" /> : <Backpack className="data-panel__icon" />}
      </div>

      <div className="data-panel__body">
        {!isRomLoaded ? (
          <div className="empty-panel">Load a ROM to inspect the bag.</div>
        ) : (
          <>
            {inventoryData && (
              <div className="money-card">
                <div>
                  <span className="money-card__label">
                    <Coins className="money-card__icon" />
                    Cash on hand
                  </span>
                  <strong className="money-card__value">
                    {inventoryData.money_formatted || `¥${inventoryData.money.toLocaleString()}`}
                  </strong>
                </div>
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}

            {!error && inventoryData && inventoryData.items.length === 0 && (
              <div className="empty-panel">No inventory items detected yet.</div>
            )}

            <div className="inventory-list">
              {inventoryData?.items.map((item) => (
                <InventoryRow key={item.slot} item={item} />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
};

const InventoryRow: React.FC<{ item: InventoryItem }> = ({ item }) => {
  const category = getCategory(item.name);

  return (
    <article className="inventory-row">
      <div className="inventory-row__meta">
        <div className="inventory-row__title">
          <span className="inventory-row__slot">#{item.slot}</span>
          <strong>{item.name}</strong>
        </div>
        <span className={classNames('inventory-tag', `inventory-tag--${category}`)}>
          {getCategoryLabel(category)}
        </span>
      </div>

      <div className="inventory-quantity">
        <Package className="inventory-quantity__icon" />
        <span>x{item.quantity}</span>
      </div>
    </article>
  );
};

export default InventoryPanel;
