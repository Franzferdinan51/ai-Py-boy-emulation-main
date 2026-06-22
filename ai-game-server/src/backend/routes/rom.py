"""
routes/rom — ROM upload + load + game-state query endpoints.

Extracted from server.py. Exposes:

  POST /api/upload-rom         — multipart upload + security validation
  POST /api/load_rom           — legacy alias (path or multipart)
  POST /api/rom/load           — modern alias for /api/load_rom
  GET  /api/game/state         — game_state dict snapshot
  GET  /api/party              — party Pokemon list (slot/level/HP/...)
  GET  /api/inventory          — money + item list
  GET  /api/memory/watch       — sparse RAM watch (placeholder)

All endpoints degrade gracefully when no emulator / no ROM is loaded. The
ROM upload pipeline reuses the existing security helpers
(`validate_file_upload`, `validate_string_input`, `sanitize_filename`) and
state-synchronization helpers (`sync_loaded_rom_state`,
`ensure_emulation_loop_running`) which are injected as dependencies so the
blueprint can be wired without importing the legacy ``server`` module.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from flask import jsonify, request

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_rom_routes(
    app,
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    # ROM upload dependencies (all required when /api/upload-rom is used)
    validate_file_upload: Optional[Callable[..., Any]] = None,
    validate_string_input: Optional[Callable[..., Any]] = None,
    sanitize_filename: Optional[Callable[[str], str]] = None,
    max_rom_size: int = 8 * 1024 * 1024,
    allowed_rom_extensions: Optional[list] = None,
    configure_emulator_launch_ui: Optional[Callable[..., Any]] = None,
    sync_loaded_rom_state: Optional[Callable[..., Any]] = None,
    ensure_emulation_loop_running: Optional[Callable[[], Any]] = None,
) -> None:
    """Register ROM upload/load + game-state query endpoints.

    The upload-specific dependencies are optional — when missing, the upload
    endpoints return 503 so the surface stays importable in isolation. Game
    state / party / inventory / memory endpoints work with only
    ``game_state_getter`` + ``emulators_getter``.
    """

    def _emulators():
        return emulators_getter() or {}

    def _ext_or_default() -> list:
        return allowed_rom_extensions or [".gb", ".gbc", ".gba", ".rom"]

    # ------------------------------------------------------------------
    # Game state queries (read-only)
    # ------------------------------------------------------------------

    @app.route("/api/game/state", methods=["GET"])
    def api_game_state():
        return jsonify(game_state_getter()), 200

    @app.route("/api/party", methods=["GET"])
    def api_party():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        empty = {
            "party_count": 0,
            "party": [],
            "timestamp": datetime.now().isoformat(),
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty), 200
        try:
            emulator = emulators[active]
            party = []
            if hasattr(emulator, "get_party_info"):
                try:
                    raw_party = emulator.get_party_info() or []
                except Exception:
                    raw_party = []
                for idx, mon in enumerate(raw_party, 1):
                    mon = mon or {}
                    party.append(
                        {
                            "slot": mon.get("slot", idx),
                            "species_id": mon.get("species_id"),
                            "species_name": mon.get("species_name"),
                            "level": mon.get("level"),
                            "hp": mon.get("hp"),
                            "max_hp": mon.get("max_hp"),
                            "status": mon.get("status"),
                            "status_text": mon.get("status_text"),
                            "type1": mon.get("type1"),
                            "type2": mon.get("type2"),
                            "moves": mon.get("moves") or [],
                            "ot_id": mon.get("ot_id"),
                            "hp_percent": mon.get("hp_percent"),
                        }
                    )
            return jsonify(
                {
                    "party_count": len(party),
                    "party": party,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            return jsonify({**empty, "error": str(e)}), 200

    @app.route("/api/inventory", methods=["GET"])
    def api_inventory():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        empty = {
            "money": 0,
            "money_formatted": "¥0",
            "item_count": 0,
            "items": [],
            "timestamp": datetime.now().isoformat(),
        }
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty), 200
        try:
            emulator = emulators[active]
            items = []
            money = 0
            if hasattr(emulator, "get_inventory_info"):
                try:
                    inv = emulator.get_inventory_info() or {}
                except Exception:
                    inv = {}
                money = inv.get("money", 0) or 0
                for idx, item in enumerate(inv.get("items") or [], 1):
                    item = item or {}
                    items.append(
                        {
                            "slot": item.get("slot", idx),
                            "id": item.get("id", 0),
                            "name": item.get("name", "Unknown"),
                            "quantity": item.get("quantity", 0),
                        }
                    )
            return jsonify(
                {
                    "money": money,
                    "money_formatted": f"¥{money:,}",
                    "item_count": len(items),
                    "items": items,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            return jsonify({**empty, "error": str(e)}), 200

    @app.route("/api/memory/watch", methods=["GET"])
    def api_memory_watch():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(
                {"addresses": [], "values": [], "timestamp": datetime.now().isoformat()}
            ), 200
        try:
            emulator = emulators[active]
            payload = {
                "addresses": [],
                "values": [],
                "timestamp": datetime.now().isoformat(),
                "frame_count": emulator.get_frame_count()
                if hasattr(emulator, "get_frame_count")
                else 0,
            }
            return jsonify(payload), 200
        except Exception as e:
            return jsonify(
                {
                    "addresses": [],
                    "values": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            ), 200

    # ------------------------------------------------------------------
    # ROM upload / load
    # ------------------------------------------------------------------

    def _upload_requires_deps():
        missing = []
        if validate_file_upload is None:
            missing.append("validate_file_upload")
        if validate_string_input is None:
            missing.append("validate_string_input")
        if sanitize_filename is None:
            missing.append("sanitize_filename")
        if sync_loaded_rom_state is None:
            missing.append("sync_loaded_rom_state")
        if ensure_emulation_loop_running is None:
            missing.append("ensure_emulation_loop_running")
        return missing

    def _upload_rom_impl():
        missing = _upload_requires_deps()
        if missing:
            return (
                jsonify(
                    {
                        "error": "ROM upload dependencies not wired",
                        "missing": missing,
                    }
                ),
                503,
            )

        logger.info("=== ROM UPLOAD REQUEST RECEIVED ===")
        try:
            file = None
            if "rom_file" in request.files:
                file = request.files["rom_file"]
            elif "rom" in request.files:
                file = request.files["rom"]
            else:
                return jsonify({"error": "No ROM file provided"}), 400
            emulator_type = request.form.get("emulator_type", "gb")
            if emulator_type == "gb":
                emulator_type = "pyboy"
            launch_ui = request.form.get("launch_ui", "false")

            logger.info(f"File received: {file.filename}")
            logger.info(f"Emulator type: {emulator_type}")
            logger.info(f"Launch UI: {launch_ui}")
            logger.info(f"Available emulators: {list(_emulators().keys())}")

            if file.filename == "" or file.filename is None:
                return jsonify({"error": "No filename provided"}), 400

            if hasattr(file, "content_length") and file.content_length > max_rom_size:
                return (
                    jsonify(
                        {
                            "error": f"File size exceeds maximum allowed size of {max_rom_size // (1024*1024)}MB"
                        }
                    ),
                    400,
                )

            try:
                file_info = validate_file_upload(
                    file,
                    "ROM file",
                    allowed_extensions=_ext_or_default(),
                    max_size=max_rom_size,
                )
                logger.info(f"File validation passed: {file_info}")
            except ValueError as e:
                logger.error(f"File validation failed: {e}")
                return jsonify({"error": str(e)}), 400

            try:
                emulator_type = validate_string_input(
                    emulator_type,
                    "emulator_type",
                    min_length=2,
                    max_length=20,
                    allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789-",
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid emulator type: {str(e)}"}), 400

            try:
                launch_ui = validate_string_input(
                    launch_ui,
                    "launch_ui",
                    min_length=2,
                    max_length=10,
                    pattern=r"^(true|false)$",
                )
                launch_ui = launch_ui.lower() == "true"
            except ValueError as e:
                return jsonify({"error": f"Invalid launch_ui parameter: {str(e)}"}), 400

            safe_filename = sanitize_filename(file_info["filename"])

            # Use secure temporary file creation. Close the handle, then
            # save into it — saving into a still-open NamedTemporaryFile can
            # result in empty files on some setups.
            fd, temp_rom_path = tempfile.mkstemp(suffix=file_info["extension"])
            os.close(fd)
            file.stream.seek(0)
            with open(temp_rom_path, "wb") as rom_out:
                shutil.copyfileobj(file.stream, rom_out)
            file.stream.seek(0)
            os.chmod(temp_rom_path, 0o600)

            logger.info(f"ROM saved to temporary path: {temp_rom_path}")

            emulator_type_mapping = {
                "gb": "pyboy",
                "gba": "pygba",
                "pyboy": "pyboy",
                "pygba": "pygba",
            }

            emulators = _emulators()
            mapped_emulator_type = emulator_type_mapping.get(emulator_type.lower())
            if not mapped_emulator_type or mapped_emulator_type not in emulators:
                logger.error(f"Invalid emulator type: {emulator_type}")
                return (
                    jsonify(
                        {
                            "error": f"Invalid emulator type. Available: {list(emulator_type_mapping.keys())}"
                        }
                    ),
                    400,
                )

            emulator_type = mapped_emulator_type
            emulator_instance = emulators[emulator_type]
            if configure_emulator_launch_ui is not None:
                configure_emulator_launch_ui(emulator_instance, launch_ui)

            logger.info(f"Loading ROM into {emulator_type} emulator...")

            try:
                temp_file_size = os.path.getsize(temp_rom_path)
                if temp_file_size < 0x150:
                    logger.error(f"Uploaded ROM temp file is too small: {temp_file_size} bytes")
                    return (
                        jsonify(
                            {"error": f"File too small to be a valid ROM ({temp_file_size} bytes)"}
                        ),
                        400,
                    )

                with open(temp_rom_path, "rb") as rom_check:
                    file_header = rom_check.read(512)

                if len(file_header) < 0x150:
                    logger.error(f"Uploaded ROM header read was too short: {len(file_header)} bytes")
                    return (
                        jsonify(
                            {"error": f"Could not read enough ROM header bytes ({len(file_header)})"}
                        ),
                        400,
                    )

                nintendo_logo = [
                    0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B,
                    0x03, 0x73, 0x00, 0x83, 0x00, 0x0C, 0x00, 0x0D,
                    0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
                    0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99,
                    0xBB, 0xBB, 0x67, 0x63, 0x6E, 0x0E, 0xEC, 0xCC,
                    0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E,
                ]
                rom_logo = list(file_header[0x104:0x134])
                logo_matches = sum(
                    1
                    for i in range(len(nintendo_logo))
                    if i < len(rom_logo) and rom_logo[i] == nintendo_logo[i]
                )
                if logo_matches < 20:
                    logger.warning(
                        f"ROM logo validation weak ({logo_matches}/{len(nintendo_logo)} matches), but continuing"
                    )
            except Exception as validation_error:
                logger.warning(f"ROM header validation failed: {validation_error}")

            logger.info(f"Loading ROM into {emulator_type} emulator...")
            try:
                success = emulator_instance.load_rom(temp_rom_path)
                if success:
                    emulator = emulator_instance
                    test_success = False
                    test_error = None

                    if hasattr(emulator, "pyboy") and emulator.pyboy:
                        try:
                            for i in range(120):
                                render_this_frame = (i == 119)
                                result = emulator.pyboy.tick(1, render_this_frame)
                                if result is False and i > 0:
                                    logger.warning(f"Warm-up frame {i} returned False")
                            test_success = True
                            logger.info("Emulator warm-up frames completed successfully")
                        except Exception as tick_error:
                            test_error = str(tick_error)
                            logger.error(f"Emulator warm-up frames failed: {tick_error}")
                    else:
                        test_error = "Emulator has no pyboy attribute"
                        logger.warning("Emulator has no pyboy attribute")

                    if not test_success:
                        logger.warning("Emulator functionality test failed, but continuing")

                    ui_status = (
                        emulator.get_ui_status()
                        if hasattr(emulator, "get_ui_status")
                        else {"running": False}
                    )
                    logger.info(f"UI status: {ui_status}")
                    rom_name = file.filename or safe_filename
                    sync_loaded_rom_state(emulator_type, temp_rom_path, rom_name=rom_name)
                    ensure_emulation_loop_running()

                    response_data = {
                        "message": "ROM loaded successfully",
                        "rom_name": rom_name,
                        "original_filename": file.filename,
                        "emulator_type": emulator_type,
                        "rom_size": os.path.getsize(temp_rom_path),
                        "ui_launched": ui_status.get("running", False),
                        "ui_status": ui_status,
                        "test_success": test_success,
                        "test_error": test_error,
                        "frame_count": emulator.get_frame_count()
                        if hasattr(emulator, "get_frame_count")
                        else 0,
                        "temp_path": temp_rom_path,
                    }
                    auto_launch_enabled = ui_status.get("auto_launch_enabled", True)
                    ui_process_running = ui_status.get("running", False)
                    if not ui_process_running and auto_launch_enabled and launch_ui:
                        logger.warning("UI process failed to launch automatically")
                        response_data["ui_help"] = {
                            "message": "Automatic UI launch failed. You can:",
                            "actions": [
                                "Try launching UI manually using the UI control panel",
                                "Check if PyBoy is properly installed: pip install pyboy",
                                "Verify SDL2 libraries are available on your system",
                                "Check the emulator logs for more details",
                            ],
                        }
                    return jsonify(response_data), 200

                logger.error(f"Failed to load ROM: {temp_rom_path}")
                if os.path.exists(temp_rom_path):
                    os.unlink(temp_rom_path)
                return (
                    jsonify(
                        {
                            "error": "Failed to load ROM into emulator",
                            "details": "The emulator could not load the ROM file. This could be due to: - Corrupted ROM file - Incompatible ROM format - Emulator initialization failure",
                            "emulator_type": emulator_type,
                        }
                    ),
                    500,
                )
            except Exception as load_error:
                logger.error(f"Exception during ROM loading: {load_error}")
                if "temp_rom_path" in locals() and os.path.exists(temp_rom_path):
                    try:
                        os.unlink(temp_rom_path)
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean up temp file: {cleanup_error}")
                return (
                    jsonify(
                        {
                            "error": "Exception during ROM loading",
                            "details": str(load_error),
                            "emulator_type": emulator_type,
                        }
                    ),
                    500,
                )
        except Exception as e:
            logger.error(f"Error uploading ROM: {e}", exc_info=True)
            if "temp_rom_path" in locals() and os.path.exists(temp_rom_path):
                try:
                    os.unlink(temp_rom_path)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temp file: {cleanup_error}")
            return (
                jsonify({"error": "Internal server error", "details": str(e)}),
                500,
            )

    @app.route("/api/upload-rom", methods=["POST"])
    def upload_rom():
        return _upload_rom_impl()

    @app.route("/api/load_rom", methods=["POST"])
    def load_rom():
        """Legacy alias — forwards multipart to upload, accepts JSON path too."""
        logger.info("=== LEGACY LOAD_ROM REQUEST RECEIVED ===")
        try:
            if "rom" in request.files or "rom_file" in request.files:
                return _upload_rom_impl()

            data = request.get_json(silent=True) or {}
            if data:
                rom_path = data.get("path") or data.get("rom_path")
                emulator_type = data.get("emulator_type", "gb")
                launch_ui = data.get("launch_ui", False)
                if emulator_type == "gb":
                    emulator_type = "pyboy"
                if not rom_path:
                    return jsonify({"error": "No ROM path provided"}), 400
                emulators = _emulators()
                if emulator_type not in emulators:
                    return (
                        jsonify({"error": f"Emulator {emulator_type} not found"}),
                        404,
                    )
                emulator = emulators[emulator_type]
                if hasattr(emulator, "set_auto_launch_ui"):
                    emulator.set_auto_launch_ui(bool(launch_ui))
                result = emulator.load_rom(rom_path)
                if result:
                    sync_loaded_rom_state(
                        emulator_type,
                        rom_path,
                        os.path.basename(rom_path),
                    ) if sync_loaded_rom_state else None
                    return (
                        jsonify(
                            {
                                "status": "success",
                                "rom_loaded": True,
                                "rom_name": os.path.basename(rom_path),
                                "emulator": emulator_type,
                            }
                        ),
                        200,
                    )
                return jsonify({"error": "Failed to load ROM"}), 500

            return jsonify({"error": "No ROM file provided"}), 400
        except Exception as e:
            logger.error(f"Error in legacy load_rom endpoint: {e}", exc_info=True)
            return (
                jsonify({"error": "Internal server error", "details": str(e)}),
                500,
            )

    @app.route("/api/rom/load", methods=["POST"])
    def api_rom_load_alias():
        return load_rom()
