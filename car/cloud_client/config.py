"""Environment-backed configuration without import-time mutation or secret repr."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shlex
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .providers import get_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLOUD_API_BASE_URL = get_provider("qwen-realtime").default_url
DEFAULT_CLOUD_MODEL = get_provider("qwen-realtime").default_model


def load_settings(env_file=None):
    configured = env_file if env_file is not None else os.environ.get("CAR_CLOUD_ENV_FILE")
    path = Path(configured).expanduser() if configured else REPO_ROOT / ".env"
    settings = {}
    if configured and not path.is_file():
        raise ValueError("configured cloud env file does not exist")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if not sep or not (key.startswith("CAR_CLOUD_") or key == "DASHSCOPE_API_KEY"):
                continue
            try:
                words = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("invalid quoted cloud environment value") from None
            if len(words) > 1:
                raise ValueError("cloud environment values with spaces must be quoted")
            settings[key] = words[0] if words else ""
    settings.update(os.environ)
    return settings


def completion_endpoint(url):
    parts = urlsplit(url.strip())
    local = parts.hostname in {"127.0.0.1", "localhost", "::1"}
    if (not parts.hostname or (parts.scheme != "https" and not (parts.scheme == "http" and local))
            or parts.username is not None or parts.password is not None or parts.query or parts.fragment
            or "{" in url or "}" in url or any(character.isspace() for character in url)):
        raise ValueError("cloud endpoint must be HTTPS without credentials, query, or placeholders (HTTP is only allowed on loopback)")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        pass
    elif path:
        path += "/chat/completions"
    else:
        path = "/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def realtime_endpoint(config):
    url=config.realtime_ws_url
    if not url:
        if not config.workspace_id: raise ValueError('set CAR_CLOUD_WORKSPACE_ID or CAR_CLOUD_REALTIME_WS_URL')
        url=f'wss://{config.workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime'
    parts=urlsplit(url);local=parts.hostname in {'127.0.0.1','localhost','::1'}
    query=parse_qsl(parts.query,keep_blank_values=True)
    if (not parts.hostname or (parts.scheme!='wss' and not (local and parts.scheme=='ws'))
            or parts.username is not None or parts.password is not None or parts.fragment
            or any(k!='model' for k,v in query) or len(query)>1
            or any(c.isspace() for c in url) or '{' in url or '}' in url):
        raise ValueError('Realtime URL must be WSS without credentials or query other than model (WS only on loopback)')
    return urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode({'model':config.model}),''))


@dataclass(frozen=True)
class CloudConfig:
    provider: str = "qwen-realtime"
    url: str = DEFAULT_CLOUD_API_BASE_URL
    model: str = DEFAULT_CLOUD_MODEL
    api_key: str = field(default="", repr=False)
    timeout: float = 30.0
    max_tokens: int = 1024
    reasoning_effort: str = "none"
    response_format: str = "json_schema"
    image_limit_mb: float = 6.0
    contract: str = 'road-observation-fast-v1'
    workspace_id: str = ''
    realtime_ws_url: str = ''

    def __post_init__(self):
        get_provider(self.provider)
        from .contracts import get_contract
        get_contract(self.contract)
        completion_endpoint(self.url)
        if not self.model.strip():
            raise ValueError("cloud model is required")
        if self.provider=='qwen-realtime' and not self.model.endswith('-realtime'):
            raise ValueError('qwen-realtime requires a realtime model; use qwen for HTTP models')
        if self.workspace_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',self.workspace_id):
            raise ValueError('invalid Realtime workspace ID')
        if self.realtime_ws_url: realtime_endpoint(self)
        for name, value in (("timeout", self.timeout), ("image_limit_mb", self.image_limit_mb)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.reasoning_effort not in {"none", "low", "medium", "xhigh"}:
            raise ValueError("invalid Qwen reasoning_effort")
        if self.response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be json_schema or json_object")

    @classmethod
    def from_env(cls, env_file=None, **overrides):
        env = load_settings(env_file)
        provider = overrides.get("provider") or env.get("CAR_CLOUD_PROVIDER") or "qwen-realtime"
        adapter = get_provider(provider)
        values = {
            "provider": provider,
            "url": env.get("CAR_CLOUD_API_BASE_URL") or adapter.default_url,
            "model": env.get("CAR_CLOUD_MODEL") or adapter.default_model,
            "api_key": env.get("CAR_CLOUD_API_KEY") or env.get(adapter.key_env, ""),
            "timeout": env.get("CAR_CLOUD_TIMEOUT") or "30",
            "max_tokens": env.get("CAR_CLOUD_MAX_COMPLETION_TOKENS") or "1024",
            "reasoning_effort": env.get("CAR_CLOUD_REASONING_EFFORT") or "none",
            "response_format": env.get("CAR_CLOUD_RESPONSE_FORMAT") or "json_schema",
            "image_limit_mb": env.get("CAR_CLOUD_IMAGE_LIMIT_MB") or "6",
            'contract': env.get('CAR_CLOUD_CONTRACT') or adapter.default_contract,
            'workspace_id': env.get('CAR_CLOUD_WORKSPACE_ID') or '',
            'realtime_ws_url': env.get('CAR_CLOUD_REALTIME_WS_URL') or env.get('REALTIME_WS_URL') or '',
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        for name, converter in (("timeout", float), ("image_limit_mb", float), ("max_tokens", int)):
            try:
                if name == "max_tokens" and isinstance(values[name], (bool, float)):
                    raise ValueError()
                values[name] = converter(values[name])
            except (ValueError, TypeError, OverflowError):
                raise ValueError("invalid numeric cloud setting: " + name) from None
        return cls(**values)
