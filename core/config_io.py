# config_io.py - The Master-Class Configuration Manager
#
# This version has been completely re-engineered for:
# - A robust versioning and automatic migration system.
# - A modular, easy-to-extend schema.
# - Detailed, actionable error reporting for failed validations.
# - Support for loading configurations from multiple locations.
# - In-memory caching for performance.

import json
import os
import copy
from cerberus import Validator
from typing import Dict, Any, Optional
import shutil
import logging
from collections.abc import MutableMapping
from functools import lru_cache
import sys

# --- New: Centralized Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- New: Configuration Versioning ---
CONFIG_VERSION = 2

# --- New: Modular Schema Definitions ---
# This is a key change for future-proofing the config
CLICK_SCHEMA = {
    "offset_x": {"type": "integer", "default": 0, "required": True},
    "offset_y": {"type": "integer", "default": 0, "required": True},
    "double_click": {"type": "boolean", "default": False, "required": True},
    "scale_x": {"type": "float", "default": 1.0, "min": 0.8, "max": 1.2, "required": True},
    "scale_y": {"type": "float", "default": 1.0, "min": 0.8, "max": 1.2, "required": True}
}
WINDOW_SCHEMA = {
    "title": {"type": "string", "default": "", "required": True},
    "hwnd": {"type": "integer", "default": 0, "required": True}
}
RUNTIME_SCHEMA = {
    "lock_cursor_during_moves": {"type": "boolean", "default": True, "required": True},
    "background_sims": {"type": "integer", "default": 7, "min": 1, "max": 16, "required": True},
    "fps": {"type": "integer", "default": 10, "min": 1, "max": 60, "required": True},
    "auto_play": {"type": "boolean", "default": True, "required": True}
}
ROI_SCHEMA = {
    "board": {"type": "dict", "default": {"x": 0, "y": 0, "w": 0, "h": 0}, "schema": {"x": {"type": "integer"}, "y": {"type": "integer"}, "w": {"type": "integer"}, "h": {"type": "integer"}}, "required": True},
    "hud": {"type": "dict", "default": {"x": 0, "y": 0, "w": 0, "h": 0}, "schema": {"x": {"type": "integer"}, "y": {"type": "integer"}, "w": {"type": "integer"}, "h": {"type": "integer"}}, "required": True},
    "status": {"type": "dict", "default": {"x": 0, "y": 0, "w": 0, "h": 0}, "schema": {"x": {"type": "integer"}, "y": {"type": "integer"}, "w": {"type": "integer"}, "h": {"type": "integer"}}, "required": True},
    "ship_right": {"type": "dict", "default": {"x": 0, "y": 0, "w": 0, "h": 0}, "schema": {"x": {"type": "integer"}, "y": {"type": "integer"}, "w": {"type": "integer"}, "h": {"type": "integer"}}, "required": True},
    "ship_lower": {"type": "dict", "default": {"x": 0, "y": 0, "w": 0, "h": 0}, "schema": {"x": {"type": "integer"}, "y": {"type": "integer"}, "w": {"type": "integer"}, "h": {"type": "integer"}}, "required": True}
}
PATHS_SCHEMA = {
    "output_root": {"type": "string", "default": "dataset", "required": True},
    "models_dir": {"type": "string", "default": "models", "required": True}
}

# --- New: Composing the Final Schema ---
CONFIG_SCHEMA = {
    "version": {"type": "integer", "default": CONFIG_VERSION, "required": True},
    "click": {"type": "dict", "schema": CLICK_SCHEMA, "required": True},
    "window": {"type": "dict", "schema": WINDOW_SCHEMA, "required": True},
    "runtime": {"type": "dict", "schema": RUNTIME_SCHEMA, "required": True},
    "roi": {"type": "dict", "schema": ROI_SCHEMA, "required": True},
    "paths": {"type": "dict", "schema": PATHS_SCHEMA, "required": True}
}

# --- New: Default Configuration based on the new schema ---
DEFAULT = {
    "version": CONFIG_VERSION,
    "click": {"offset_x": 0, "offset_y": 0, "double_click": False, "scale_x": 1.0, "scale_y": 1.0},
    "window": {"title": "", "hwnd": 0},
    "runtime": {"lock_cursor_during_moves": True, "background_sims": 7, "fps": 10, "auto_play": True},
    "roi": {"board": {"x": 0, "y": 0, "w": 0, "h": 0}, "hud": {"x": 0, "y": 0, "w": 0, "h": 0},
            "status": {"x": 0, "y": 0, "w": 0, "h": 0}, "ship_right": {"x": 0, "y": 0, "w": 0, "h": 0},
            "ship_lower": {"x": 0, "y": 0, "w": 0, "h": 0}},
    "paths": {"output_root": "dataset", "models_dir": "models"}
}

class ConfigSection:
    """A recursive object-like wrapper for configuration dictionaries."""
    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                self.__dict__[key] = ConfigSection(value)
            else:
                self.__dict__[key] = value

    def to_dict(self) -> Dict:
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output

class ConfigManager:
    """
    Manages the bot's configuration with validation, versioning, and atomic saving.
    """
    def __init__(self, path: Optional[str] = None):
        self._path = path if path else os.path.join(os.getcwd(), "config.json")
        self._validator = Validator(CONFIG_SCHEMA)

    def _migrate_config(self, cfg: Dict, from_version: int) -> Dict:
        """
        Migrates a configuration dictionary from an old version to the current one.
        This is where all migration logic would live.
        """
        if from_version < 2:
            if "runtime" in cfg and "beam_width" in cfg["runtime"]:
                del cfg["runtime"]["beam_width"]
            if "runtime" in cfg and "lookahead" in cfg["runtime"]:
                del cfg["runtime"]["lookahead"]
            logger.info(f"Migrated configuration from version {from_version} to {CONFIG_VERSION}.")
            cfg["version"] = CONFIG_VERSION
        return cfg

    @lru_cache(maxsize=1)
    def load_config(self) -> ConfigSection:
        """
        Loads and validates the configuration from the file.
        Recovers gracefully from a corrupted file and migrates old versions.
        This function is now cached for improved performance.
        """
        path_found = False
        config_data = {}
        
        # New: Search multiple paths for the config file
        search_paths = [self._path, os.path.join(os.path.expanduser("~"), "BilgeBot", "config.json")]
        
        for p in search_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    self._path = p
                    path_found = True
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding config file at {p}: {e}")
                    continue
        
        if not path_found:
            logger.warning("Configuration file not found. Generating a default.")
            return ConfigSection(DEFAULT)

        version = config_data.get("version", 1)
        if version < CONFIG_VERSION:
            logger.info(f"Old config file version detected ({version}). Attempting migration...")
            config_data = self._migrate_config(config_data, version)
        
        # Validate and normalize the config using Cerberus
        if not self._validator.validate(config_data, CONFIG_SCHEMA):
            logger.error("Configuration validation failed. Errors:")
            for field, errors in self._validator.errors.items():
                logger.error(f"  - Field '{field}': {errors}")
            
            normalized_cfg = self._validator.normalized(config_data)
            if normalized_cfg is None:
                logger.critical("Validation failed. Reverting to default configuration.")
                return ConfigSection(DEFAULT)
            
            logger.warning("Using a normalized version of the config. Saving to disk...")
            self.save_config(ConfigSection(normalized_cfg))
            return ConfigSection(normalized_cfg)

        return ConfigSection(config_data)

    def save_config(self, cfg: ConfigSection):
        """
        Saves the configuration to a file using an atomic write operation.
        """
        temp_path = self._path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cfg.to_dict(), f, indent=2)
            shutil.move(temp_path, self._path)
            logger.info(f"Configuration saved successfully to {self._path}")
            # Clear the cache after saving to ensure the next load gets the new data
            self.load_config.cache_clear()
        except Exception as e:
            logger.error(f"Error saving configuration to {self._path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

# We expose a central manager for other modules to use
_manager = ConfigManager()

def load_config() -> ConfigSection:
    return _manager.load_config()

def save_config(cfg: ConfigSection):
    _manager.save_config(cfg)

if __name__ == "__main__":
    # Example usage
    config = load_config()
    print("Loaded Configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    # Example of saving a modified config
    config.runtime.fps = 60
    config.click.scale_x = 1.1
    save_config(config)
    print("Saved a modified configuration.")

    # A simple test to simulate a bad file
    bad_config_path = "bad_config.json"
    with open(bad_config_path, "w") as f:
        f.write('{"runtime": {"fps": "not-a-number"}}')
    
    # We create a new manager instance to avoid the cache from the previous call
    _manager_bad = ConfigManager(bad_config_path)
    try:
        bad_config = _manager_bad.load_config()
        print("Loaded a bad configuration, recovered with defaults:", bad_config.to_dict())
    except Exception as e:
        print(f"Failed to load bad config: {e}")
    
    os.remove(bad_config_path)