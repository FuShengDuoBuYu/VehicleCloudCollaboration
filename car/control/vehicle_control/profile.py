"""Load vehicle profiles and merge their safe runtime overrides."""

from copy import deepcopy
from pathlib import Path

import yaml


PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "raspbot_pi5"
SUPPORTED_BACKENDS = ("raspbot", "rosmaster")


def deep_merge(base, override):
    """Recursively merge mappings; scalars and lists replace base values."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_profile_path(selector=None, relative_to=None):
    selector = str(selector or DEFAULT_PROFILE).strip()
    if not selector:
        selector = DEFAULT_PROFILE
    candidate = Path(selector).expanduser()
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_absolute():
        if not candidate.is_absolute():
            candidate = Path(relative_to or Path.cwd()) / candidate
    else:
        candidate = PROFILE_DIR / f"{selector}.yaml"
    return candidate.resolve()


def load_vehicle_profile(selector=None, relative_to=None):
    path = resolve_profile_path(selector, relative_to=relative_to)
    if not path.is_file():
        available = ", ".join(
            sorted(item.stem for item in PROFILE_DIR.glob("*.yaml"))
        )
        raise FileNotFoundError(
            f"vehicle profile does not exist: {path}; available: {available}"
        )
    profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if profile.get("version") != 1:
        raise ValueError(f"vehicle profile version must be 1: {path}")
    profile_id = str(profile.get("id", "")).strip()
    backend = str(profile.get("backend", "")).strip()
    if not profile_id:
        raise ValueError(f"vehicle profile id is required: {path}")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported vehicle backend {backend!r}; expected {SUPPORTED_BACKENDS}"
        )
    profile["profile"] = profile_id
    profile["profile_path"] = str(path)
    return path, profile


def load_runtime_config(path, vehicle_selector=None):
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("version") != 1:
        raise ValueError("onboard runtime config version must be 1")
    configured = (config.get("vehicle") or {}).get("profile")
    profile_path, profile = load_vehicle_profile(
        vehicle_selector or configured or DEFAULT_PROFILE,
        relative_to=(Path.cwd() if vehicle_selector else config_path.parent),
    )
    resolved = deep_merge(config, profile.get("runtime_overrides", {}))
    public_profile = {
        key: deepcopy(value)
        for key, value in profile.items()
        if key != "runtime_overrides"
    }
    runtime_vehicle_overrides = {
        key: deepcopy(value)
        for key, value in (config.get("vehicle") or {}).items()
        if key != "profile"
    }
    resolved["vehicle"] = deep_merge(public_profile, runtime_vehicle_overrides)
    resolved["vehicle"]["profile"] = profile["profile"]
    resolved["vehicle"]["profile_path"] = str(profile_path)
    return config_path, resolved
