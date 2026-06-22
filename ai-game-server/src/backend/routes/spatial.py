"""
routes/spatial — Spatial / position / minimap / NPCs / strategy endpoints.

Extracted from server.py. Exposes:

  GET  /api/spatial/position  — player position + map_id
  GET  /api/spatial/minimap   — sparse 2D tile layout + player
  GET  /api/spatial/npcs      — nearby NPCs (battle-only for now)
  GET  /api/spatial/strategy  — strategic analysis (heal, battle, money)
  GET  /api/agent/strategy    — alias for /api/spatial/strategy

All endpoints return a stable empty / fallback shape when no ROM is loaded
so the frontend can render the placeholder UI without conditional logic.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict

from flask import jsonify

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_spatial_routes(
    app,
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
) -> None:
    """Register spatial / position / minimap / NPCs / strategy endpoints."""

    def _emulators():
        return emulators_getter() or {}

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/spatial/position", methods=["GET"])
    def api_spatial_position():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")

        empty_response = {
            "x": 0,
            "y": 0,
            "map_id": 0,
            "map_name": "none",
            "timestamp": datetime.now().isoformat(),
            "loaded": False,
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]

            if hasattr(emulator, "get_position"):
                pos = emulator.get_position()
                pos["timestamp"] = datetime.now().isoformat()
                pos["loaded"] = True
                return jsonify(pos), 200

            # Fallback: read memory directly for Pokemon
            if hasattr(emulator, "_read_byte"):
                x = emulator._read_byte(0xD062)
                y = emulator._read_byte(0xD063)
                map_id = emulator._read_byte(0xD35E)
                return jsonify(
                    {
                        "x": x,
                        "y": y,
                        "map_id": map_id,
                        "map_name": f"Map {map_id}",
                        "timestamp": datetime.now().isoformat(),
                        "loaded": True,
                    }
                ), 200

            return jsonify(empty_response), 200
        except Exception as e:
            logger.debug(f"Error getting position: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200

    @app.route("/api/spatial/minimap", methods=["GET"])
    def api_spatial_minimap():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")

        empty_response = {
            "width": 0,
            "height": 0,
            "tiles": [],
            "player": {"x": 0, "y": 0},
            "timestamp": datetime.now().isoformat(),
            "loaded": False,
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]
            player_pos = {"x": 0, "y": 0}
            if hasattr(emulator, "get_position"):
                pos = emulator.get_position()
                player_pos = {"x": pos.get("x", 0), "y": pos.get("y", 0)}

            # Sparse minimap — full tilemap reading is not yet implemented
            return jsonify(
                {
                    "width": 20,
                    "height": 18,
                    "tiles": [],
                    "player": player_pos,
                    "timestamp": datetime.now().isoformat(),
                    "loaded": True,
                    "note": "Sparse minimap - tile data not implemented",
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting minimap: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200

    @app.route("/api/spatial/npcs", methods=["GET"])
    def api_spatial_npcs():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")

        empty_response = {
            "npcs": [],
            "count": 0,
            "timestamp": datetime.now().isoformat(),
            "loaded": False,
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]
            npcs = []

            if hasattr(emulator, "get_battle_info"):
                battle = emulator.get_battle_info()
                if battle.get("in_battle") and battle.get("enemy"):
                    enemy = battle["enemy"]
                    npcs.append(
                        {
                            "id": enemy.get("species_id", 0),
                            "name": enemy.get("species_name", "Unknown"),
                            "x": -1,
                            "y": -1,
                            "type": "enemy_pokemon",
                            "level": enemy.get("level", 0),
                            "hp_percent": enemy.get("hp_percent", 0),
                        }
                    )

            return jsonify(
                {
                    "npcs": npcs,
                    "count": len(npcs),
                    "timestamp": datetime.now().isoformat(),
                    "loaded": True,
                    "note": "Battle NPCs only - sprite memory reading not implemented",
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting NPCs: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200

    @app.route("/api/agent/strategy", methods=["GET"])
    @app.route("/api/spatial/strategy", methods=["GET"])
    def api_spatial_strategy():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")

        empty_response = {
            "status": "no_rom",
            "health": {
                "party_healthy": True,
                "lowest_hp_percent": 100,
                "needs_healing": False,
            },
            "battle": {"in_battle": False, "recommendation": "none"},
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
            "loaded": False,
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]

            party = []
            if hasattr(emulator, "get_party_info"):
                party = emulator.get_party_info() or []

            lowest_hp = 100
            needs_healing = False
            for mon in party:
                hp_pct = mon.get("hp_percent", 100)
                if hp_pct < lowest_hp:
                    lowest_hp = hp_pct
                if hp_pct < 30:
                    needs_healing = True

            party_healthy = not needs_healing

            battle_info = {"in_battle": False, "recommendation": "none"}
            battle_rec = "none"

            if hasattr(emulator, "get_battle_info"):
                battle = emulator.get_battle_info()
                battle_info["in_battle"] = battle.get("in_battle", False)
                if battle.get("in_battle"):
                    enemy = battle.get("enemy", {})
                    enemy_hp_pct = enemy.get("hp_percent", 100)
                    if lowest_hp < 20:
                        battle_rec = "run"
                    elif enemy_hp_pct < 30 and lowest_hp > 50:
                        battle_rec = "catch"
                    else:
                        battle_rec = "attack"
                    battle_info["recommendation"] = battle_rec
                    battle_info["enemy"] = enemy

            recommendations = []
            if needs_healing:
                recommendations.append("Heal party at Pokemon Center")
            if battle_info["in_battle"]:
                recommendations.append(f"Battle: {battle_rec}")
            if not party:
                recommendations.append("Get first Pokemon")

            inv = {}
            if hasattr(emulator, "get_inventory_info"):
                inv = emulator.get_inventory_info() or {}
            money = inv.get("money", 0)
            if money > 0:
                recommendations.insert(0, f"Money: ¥{money:,}")

            if battle_info["in_battle"]:
                status = (
                    f"In battle vs {battle_info.get('enemy', {}).get('species_name', 'unknown')}"
                )
            elif needs_healing:
                status = f"Party needs healing ({lowest_hp:.0f}% HP)"
            elif party:
                status = f"Exploring with {len(party)} Pokemon"
            else:
                status = "Ready"

            return jsonify(
                {
                    "status": status,
                    "health": {
                        "party_healthy": party_healthy,
                        "lowest_hp_percent": round(lowest_hp, 1),
                        "needs_healing": needs_healing,
                    },
                    "battle": battle_info,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat(),
                    "loaded": True,
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting strategy: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200
