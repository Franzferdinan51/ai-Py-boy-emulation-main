import React, { useEffect, useState } from 'react';
import { Activity, Backpack, Coins, Package } from 'lucide-react';
import apiService, { type InventoryData, type InventoryItem } from '../../services/apiService';

interface InventoryPanelProps {
  isRomLoaded: boolean;
  onInventoryUpdate?: (inventory: InventoryData) => void;
}

const INVENTORY_REFRESH_MS = 10000;

const getCategory = (itemName: string) => {
  const label = itemName.toLowerCase();

  if (label.includes('ball')) {
    return 'Pokeball';
  }

  if (label.includes('potion') || label.includes('heal') || label.includes('revive') || label.includes('antidote')) {
    return 'Medicine';
  }

  if (label.includes('badge')) {
    return 'Badge';
  }

  if (label.includes('key') || label.includes('ticket') || label.includes('pass')) {
    return 'Key item';
  }

  if (label.includes('rod')) {
    return 'Fishing';
  }

  if (label.includes('stone')) {
    return 'Stone';
  }

  return 'General';
};

const getCategoryClasses = (category: string) => {
  switch (category) {
    case 'Pokeball':
      return 'border-red-900/60 bg-red-950/40 text-red-300';
    case 'Medicine':
      return 'border-green-900/60 bg-green-950/40 text-green-300';
    case 'Badge':
      return 'border-yellow-900/60 bg-yellow-950/40 text-yellow-300';
    case 'Key item':
      return 'border-blue-900/60 bg-blue-950/40 text-blue-300';
    case 'Fishing':
      return 'border-cyan-900/60 bg-cyan-950/40 text-cyan-300';
    case 'Stone':
      return 'border-purple-900/60 bg-purple-950/40 text-purple-300';
    default:
      return 'border-neutral-800 bg-neutral-900 text-neutral-300';
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

    fetchInventory();
    const intervalId = window.setInterval(fetchInventory, INVENTORY_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isRomLoaded, onInventoryUpdate]);

  if (!isRomLoaded) {
    return (
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70 p-5 text-center text-neutral-500">
        <Backpack className="mx-auto mb-3 h-8 w-8 opacity-50" />
        <p className="text-sm">Load a ROM to inspect the bag.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900/70">
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">Inventory</h3>
          <p className="mt-1 text-xs text-neutral-500">
            {inventoryData ? `${inventoryData.item_count} tracked items` : 'Reading bag data'}
          </p>
        </div>
        {loading && <Activity className="h-4 w-4 animate-spin text-neutral-500" />}
      </div>

      <div className="space-y-3 p-4">
        {inventoryData && (
          <div className="rounded-2xl border border-neutral-800 bg-neutral-950/80 px-4 py-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-sm text-neutral-300">
                <Coins className="h-4 w-4 text-yellow-400" />
                Cash on hand
              </span>
              <span className="text-lg font-semibold text-yellow-300">
                {inventoryData.money_formatted || `¥${inventoryData.money.toLocaleString()}`}
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-900/60 bg-red-950/40 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        {!error && inventoryData && inventoryData.items.length === 0 && (
          <div className="rounded-xl border border-neutral-800 bg-neutral-950/70 px-4 py-6 text-center text-sm text-neutral-500">
            No inventory items detected yet.
          </div>
        )}

        <div className="space-y-2">
          {inventoryData?.items.map((item) => (
            <InventoryRow key={item.slot} item={item} />
          ))}
        </div>
      </div>
    </div>
  );
};

const InventoryRow: React.FC<{ item: InventoryItem }> = ({ item }) => {
  const category = getCategory(item.name);

  return (
    <div className="flex items-center justify-between rounded-2xl border border-neutral-800 bg-neutral-950/80 px-4 py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wide text-neutral-500">#{item.slot}</span>
          <span className="truncate text-sm font-medium text-white">{item.name}</span>
        </div>
        <span className={`mt-2 inline-flex rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-wide ${getCategoryClasses(category)}`}>
          {category}
        </span>
      </div>

      <div className="ml-4 flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-3 py-1 text-sm text-neutral-300">
        <Package className="h-3.5 w-3.5" />
        x{item.quantity}
      </div>
    </div>
  );
};

export default InventoryPanel;
