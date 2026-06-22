"""
Collision Map — RAM-derived walkability grid.

Inspired by NousResearch/pokemon-agent's `/map/ascii` endpoint and
`/screenshot/grid` A1..J9 grid overlay.

This module provides a base interface for collision-map extraction that can
be specialized per game (Pokemon Red, etc.). The base implementation derives
a coarse walkability grid from the visible tilemap:

  - Each tile is marked WALKABLE / BLOCKED / UNKNOWN based on tile ID range.
  - Grid can be exported as ASCII (PIL/numpy not required) or as a labeled
    coordinate grid (A1..J9, etc.) suitable for screenshot overlay.

Per-game specializations can be added by subclassing `CollisionMapProvider`
and registering it with `register_collision_provider()`.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

_lock = threading.RLock()
_providers: Dict[str, "CollisionMapProvider"] = {}
_default_provider = "pokemon_red"


class CollisionMapProvider:
    """Base collision-map provider."""

    name: str = "base"

    def get_walkable_tile_ids(self) -> List[int]:
        """Return list of tile IDs considered walkable. Default heuristic."""
        # Generic Game Boy: tile 0x00 is "blank", common walkable grass floors
        return [0x00, 0x01, 0x02, 0x04, 0x05, 0x10, 0x12, 0x14, 0x15, 0x16, 0x18, 0x1A, 0x1C, 0x1E]

    def is_walkable(self, tile_id: int) -> bool:
        return int(tile_id) in self.get_walkable_tile_ids()

    def extract_tilemap(
        self,
        emulator,
        width: int = 32,
        height: int = 32,
    ) -> List[List[int]]:
        """Pull a tilemap from the emulator. Returns 2D list of tile ids."""
        try:
            # PyBoy's tilemap access
            if hasattr(emulator, "pyboy") and emulator.pyboy is not None:
                pyboy = emulator.pyboy
                if hasattr(pyboy, "tilemap") and hasattr(pyboy.tilemap, "search"):
                    # PyBoy 2.x API
                    screen_width = getattr(pyboy, "screen", None)
                    raw = pyboy.tilemap.search(range(0, 384))
                    # Pad/crop to width*height
                    if not isinstance(raw, (list, tuple)):
                        raw = list(raw)
                    needed = width * height
                    if len(raw) < needed:
                        raw = raw + [0] * (needed - len(raw))
                    raw = raw[:needed]
                    return [list(raw[i * width:(i + 1) * width]) for i in range(height)]
        except Exception:  # noqa: BLE001
            pass
        return [[0] * width for _ in range(height)]

    def build_collision_grid(
        self,
        emulator,
        width: int = 32,
        height: int = 32,
    ) -> Dict[str, Any]:
        """Build a 2D walkability grid from the emulator's tilemap."""
        tilemap = self.extract_tilemap(emulator, width, height)
        grid: List[List[str]] = []
        for row in tilemap:
            grid_row = []
            for tile_id in row:
                if tile_id == 0:
                    grid_row.append(".")  # unknown/blank
                elif self.is_walkable(tile_id):
                    grid_row.append("·")  # walkable
                else:
                    grid_row.append("#")  # blocked
            grid.append(grid_row)
        return {
            "provider": self.name,
            "width": width,
            "height": height,
            "grid": grid,
            "tilemap": tilemap,
        }

    def to_ascii(self, grid: List[List[str]]) -> str:
        return "\n".join("".join(row) for row in grid)

    def to_labeled_grid(
        self,
        grid: List[List[str]],
        col_labels: Optional[List[str]] = None,
        row_labels: Optional[List[str]] = None,
    ) -> str:
        """Add coordinate labels like A1..J9, useful for screenshot overlay."""
        height = len(grid)
        width = len(grid[0]) if height else 0
        col_labels = col_labels or [chr(ord("A") + i) for i in range(width)]
        row_labels = row_labels or [str(i + 1) for i in range(height)]
        lines = []
        # Header
        lines.append("   " + " ".join(col_labels[:width]))
        for i, row in enumerate(grid[:height]):
            label = row_labels[i] if i < len(row_labels) else str(i + 1)
            lines.append(f"{label:>2} " + " ".join(row))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pokemon Red specialization stub
# ---------------------------------------------------------------------------

class PokemonRedCollisionProvider(CollisionMapProvider):
    name = "pokemon_red"

    def get_walkable_tile_ids(self) -> List[int]:
        # Coarse Pokemon Red walkable set (grass, doors, paths, water edges).
        # Real implementations would parse map blocks more carefully.
        return [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
            0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
            0x52, 0x53, 0x54, 0x55, 0x5C, 0x5D, 0x5E, 0x5F,
        ]


# Register default providers (deferred to bottom to avoid forward ref)
_default_providers_to_register: List[CollisionMapProvider] = [PokemonRedCollisionProvider()]


def register_collision_provider(provider: CollisionMapProvider) -> None:
    with _lock:
        _providers[provider.name] = provider


def get_provider(name: Optional[str] = None) -> CollisionMapProvider:
    with _lock:
        return _providers.get(name or _default_provider, _providers[_default_provider])


# Register deferred defaults
for _p in _default_providers_to_register:
    register_collision_provider(_p)
_default_providers_to_register.clear()


# ---------------------------------------------------------------------------
# Flask registration
# ---------------------------------------------------------------------------

def register_collision_routes(app, *, emulator_ref_getter=None):
    """Register collision-map routes.

    emulator_ref_getter: callable returning the current emulator instance.
    Wire it to the same getter the rest of the server uses to find the active
    emulator (e.g. lambda: get_active_emulator()).
    """
    from flask import jsonify, request, Response

    if emulator_ref_getter is None:
        emulator_ref_getter = lambda: None  # noqa: E731

    @app.route("/api/spatial/collision", methods=["GET"])
    def _collision():
        width = int(request.args.get("width", 32))
        height = int(request.args.get("height", 32))
        provider_name = request.args.get("provider")
        provider = get_provider(provider_name)
        emu = emulator_ref_getter()
        if emu is None:
            return jsonify({
                "ok": False,
                "error": "no emulator loaded",
                "ascii": "",
                "labeled": "",
            }), 200
        result = provider.build_collision_grid(emu, width, height)
        result["ascii"] = provider.to_ascii(result["grid"])
        result["labeled"] = provider.to_labeled_grid(result["grid"])
        return jsonify({"ok": True, **result})

    @app.route("/api/spatial/grid", methods=["GET"])
    def _grid():
        width = int(request.args.get("width", 20))
        height = int(request.args.get("height", 18))
        provider_name = request.args.get("provider")
        provider = get_provider(provider_name)
        emu = emulator_ref_getter()
        if emu is None:
            return jsonify({
                "ok": False,
                "error": "no emulator loaded",
                "ascii": "",
                "labeled": "",
            }), 200
        result = provider.build_collision_grid(emu, width, height)
        result["ascii"] = provider.to_ascii(result["grid"])
        result["labeled"] = provider.to_labeled_grid(result["grid"])
        return jsonify({"ok": True, **result})

    @app.route("/api/spatial/grid/text", methods=["GET"])
    def _grid_text():
        """Plain-text labeled grid, ideal for prompt injection."""
        width = int(request.args.get("width", 20))
        height = int(request.args.get("height", 18))
        provider_name = request.args.get("provider")
        provider = get_provider(provider_name)
        emu = emulator_ref_getter()
        if emu is None:
            return "(no emulator loaded)", 200
        result = provider.build_collision_grid(emu, width, height)
        return Response(provider.to_labeled_grid(result["grid"]), mimetype="text/plain")
