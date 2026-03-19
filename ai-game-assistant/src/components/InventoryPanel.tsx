import React, { useState, useEffect } from 'react';
import { Backpack, Coins, Package } from 'lucide-react';
import type { InventoryData, InventoryItem } from '../../services/apiService';
import apiService from '../../services/apiService';

interface InventoryPanelProps {
  isRomLoaded: boolean;
  onInventoryUpdate?: (inventory: InventoryData) => void;
}

const InventoryPanel: React.FC<InventoryPanelProps> = ({ isRomLoaded, onInventoryUpdate }) => {
  const [inventoryData, setInventoryData] = useState<InventoryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchInventory = async () => {
    if (!isRomLoaded) return;
    
    setLoading(true);
    try {
      const data = await apiService.getInventory();
      setInventoryData(data);
      setError(null);
      onInventoryUpdate?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch inventory');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isRomLoaded) {
      setInventoryData(null);
      return;
    }

    fetchInventory();
    const interval = setInterval(fetchInventory, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [isRomLoaded]);

  if (!isRomLoaded) {
    return (
      <div className="p-4 text-center text-neutral-500">
        <Backpack className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">Load a ROM to view inventory</p>
      </div>
    );
  }

  if (loading && !inventoryData) {
    return (
      <div className="p-4 text-center text-neutral-400">
        <Package className="w-6 h-6 mx-auto mb-2 animate-spin" />
        <p className="text-sm">Loading inventory...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-center text-red-400">
        <p className="text-sm">{error}</p>
        <button onClick={fetchInventory} className="mt-2 text-xs underline">Retry</button>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-400 uppercase flex items-center gap-2">
          <Backpack className="w-4 h-4" /> Inventory
        </h3>
        <button 
          onClick={fetchInventory}
          className="p-1 hover:bg-neutral-800 rounded"
          title="Refresh"
        >
          <Package className="w-3 h-3" />
        </button>
      </div>

      {/* Money Display */}
      {inventoryData && (
        <div className="bg-neutral-800 rounded-lg p-3 border border-neutral-700">
          <div className="flex items-center justify-between">
            <span className="text-neutral-400 text-sm flex items-center gap-2">
              <Coins className="w-4 h-4 text-yellow-400" /> Money
            </span>
            <span className="text-lg font-bold text-yellow-400">
              ₽{inventoryData.money.toLocaleString()}
            </span>
          </div>
          <div className="mt-1 text-xs text-neutral-500">
            {inventoryData.item_count} items in bag
          </div>
        </div>
      )}

      {/* Items List */}
      {inventoryData && inventoryData.items.length > 0 ? (
        <div className="space-y-1 max-h-96 overflow-y-auto">
          {inventoryData.items.map((item) => (
            <ItemRow key={item.slot} item={item} />
          ))}
        </div>
      ) : (
        <div className="text-center text-neutral-500 py-8">
          <Backpack className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No items in bag</p>
          <p className="text-xs mt-1">Visit a Poké Mart to buy items!</p>
        </div>
      )}
    </div>
  );
};

interface ItemRowProps {
  item: InventoryItem;
}

const ItemRow: React.FC<ItemRowProps> = ({ item }) => {
  // Categorize items by type for better visualization
  const getItemCategory = (itemName: string): string => {
    const name = itemName.toLowerCase();
    if (name.includes('ball')) return 'Poké Ball';
    if (name.includes('potion') || name.includes('heal') || name.includes('antidote') || name.includes('revive')) return 'Medicine';
    if (name.includes('stone')) return 'Evolution Stone';
    if (name.includes('badge')) return 'Badge';
    if (name.includes('key') || name.includes('ticket') || name.includes('pass')) return 'Key Item';
    if (name.includes('rod')) return 'Fishing Rod';
    if (name.includes('repel')) return 'Repel';
    if (name.includes('ether') || name.includes('elixir') || name.includes('pp up')) return 'PP Restore';
    if (name.includes('attack') || name.includes('defend') || name.includes('speed') || name.includes('special')) return 'Stat Boost';
    if (name.includes('candy') || name.includes('protein') || name.includes('iron') || name.includes('calcium') || name.includes('carbos') || name.includes('hp up')) return 'Rare Candy/Stats';
    return 'Other';
  };

  const getCategoryColor = (category: string): string => {
    switch (category) {
      case 'Poké Ball': return 'text-red-400 bg-red-900/20';
      case 'Medicine': return 'text-green-400 bg-green-900/20';
      case 'Evolution Stone': return 'text-purple-400 bg-purple-900/20';
      case 'Badge': return 'text-yellow-400 bg-yellow-900/20';
      case 'Key Item': return 'text-blue-400 bg-blue-900/20';
      case 'Fishing Rod': return 'text-cyan-400 bg-cyan-900/20';
      case 'Repel': return 'text-gray-400 bg-gray-900/20';
      case 'PP Restore': return 'text-pink-400 bg-pink-900/20';
      case 'Stat Boost': return 'text-orange-400 bg-orange-900/20';
      case 'Rare Candy/Stats': return 'text-amber-400 bg-amber-900/20';
      default: return 'text-neutral-400 bg-neutral-900/20';
    }
  };

  const category = getItemCategory(item.name);
  const colorClass = getCategoryColor(category);

  return (
    <div className="flex items-center justify-between p-2 bg-neutral-800/50 hover:bg-neutral-800 rounded transition-colors">
      <div className="flex items-center gap-2">
        <span className="text-xs text-neutral-500 w-6">#{item.slot}</span>
        <div>
          <div className="text-sm text-white">{item.name}</div>
          <div className={`text-xs px-1.5 py-0.5 rounded inline-block mt-0.5 ${colorClass}`}>
            {category}
          </div>
        </div>
      </div>
      <div className="text-right">
        <div className="text-sm font-medium text-neutral-300">x{item.quantity}</div>
      </div>
    </div>
  );
};

export default InventoryPanel;
