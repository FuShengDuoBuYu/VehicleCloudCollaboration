import base64
import json
import mimetypes
import os
import re
import shlex
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]

CLOUD_ENV_FILE = "CAR_CLOUD_ENV_FILE"
CLOUD_API_BASE_URL_ENV = "CAR_CLOUD_API_BASE_URL"
CLOUD_MODEL_ENV = "CAR_CLOUD_MODEL"
CLOUD_TIMEOUT_ENV = "CAR_CLOUD_TIMEOUT"
CLOUD_NUM_CTX_ENV = "CAR_CLOUD_NUM_CTX"
CLOUD_MAX_TOKENS_ENV = "CAR_CLOUD_MAX_COMPLETION_TOKENS"
CLOUD_TEMPERATURE_ENV = "CAR_CLOUD_TEMPERATURE"
CLOUD_TOP_P_ENV = "CAR_CLOUD_TOP_P"
CLOUD_SEED_ENV = "CAR_CLOUD_SEED"
CLOUD_THINK_ENV = "CAR_CLOUD_THINK"
CLOUD_IMAGE_LIMIT_MB_ENV = "CAR_CLOUD_IMAGE_LIMIT_MB"
CLOUD_USER_AGENT_ENV = "CAR_CLOUD_USER_AGENT"


def _load_env_file() -> None:
    configured = os.environ.get(CLOUD_ENV_FILE, "").strip()
    candidates = [Path(configured)] if configured else [REPO_ROOT / ".env", REPO_ROOT / ".env_example"]
    env_file = next((path for path in candidates if path.exists()), None)
    if env_file is None:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_without_dotenv(env_file)
        return

    load_dotenv(env_file, override=False)


def _load_env_file_without_dotenv(env_file: Path) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = _strip_env_value(value.strip())


def _strip_env_value(value: str) -> str:
    try:
        parsed = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.strip("\"'")
    return parsed[0] if parsed else ""


_load_env_file()

DEFAULT_CLOUD_API_BASE_URL = os.environ.get(CLOUD_API_BASE_URL_ENV, "").strip()
DEFAULT_CLOUD_MODEL = os.environ.get(CLOUD_MODEL_ENV, "qwen3.5:9b").strip() or "qwen3.5:9b"
DEFAULT_CLOUD_TIMEOUT = float(os.environ.get(CLOUD_TIMEOUT_ENV, "30") or 30)
DEFAULT_CLOUD_NUM_CTX = int(os.environ.get(CLOUD_NUM_CTX_ENV, "93696") or 93696)
DEFAULT_CLOUD_MAX_COMPLETION_TOKENS = int(os.environ.get(CLOUD_MAX_TOKENS_ENV, "256") or 256)
DEFAULT_CLOUD_TEMPERATURE = float(os.environ.get(CLOUD_TEMPERATURE_ENV, "0") or 0)
DEFAULT_CLOUD_TOP_P = float(os.environ.get(CLOUD_TOP_P_ENV, "0.95") or 0.95)
DEFAULT_CLOUD_SEED = int(os.environ.get(CLOUD_SEED_ENV, "42") or 42)
DEFAULT_CLOUD_THINK = os.environ.get(CLOUD_THINK_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_CLOUD_IMAGE_LIMIT_BYTES = int(float(os.environ.get(CLOUD_IMAGE_LIMIT_MB_ENV, "20") or 20) * 1024 * 1024)
DEFAULT_CLOUD_USER_AGENT = os.environ.get(CLOUD_USER_AGENT_ENV, "VehicleCloudCollaboration/1.0").strip() or "VehicleCloudCollaboration/1.0"

COMMAND_TO_ACTION = {
    "left": "lane-left",
    "right": "lane-right",
}

DECISION_SYSTEM_PROMPT = """
你是车云协同自动驾驶的云端决策服务。
你会收到车端长尾检测结果和当前摄像头画面。
请基于画面中的道路、障碍物、车道和交通风险，选择一个安全的车辆动作。
必须只返回 JSON，不要输出 Markdown、解释或多余文本。
command 只能是 left 或 right。
action 必须分别对应 lane-left 或 lane-right。
""".strip()

DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "enum": ["left", "right"]},
        "action": {"type": "string", "enum": ["lane-left", "lane-right"]},
        "reason": {"type": "string"},
    },
    "required": ["command", "action", "reason"],
    "additionalProperties": False,
}


@dataclass
class CloudDecision:
    command: str
    action: str
    reason: str
    latency_ms: float
    raw_response: Dict[str, Any]
    timings_ms: Optional[Dict[str, float]] = None


class CloudClient:
    def __init__(
        self,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        model: Optional[str] = None,
        force_left: bool = False,
        num_ctx: int = DEFAULT_CLOUD_NUM_CTX,
        max_completion_tokens: int = DEFAULT_CLOUD_MAX_COMPLETION_TOKENS,
        temperature: float = DEFAULT_CLOUD_TEMPERATURE,
        top_p: float = DEFAULT_CLOUD_TOP_P,
        seed: int = DEFAULT_CLOUD_SEED,
        think: bool = DEFAULT_CLOUD_THINK,
        image_limit_bytes: int = DEFAULT_CLOUD_IMAGE_LIMIT_BYTES,
        user_agent: Optional[str] = None,
    ):
        _load_env_file()
        self.base_url = (url or os.environ.get(CLOUD_API_BASE_URL_ENV) or DEFAULT_CLOUD_API_BASE_URL).strip()
        self.url = self._chat_completions_url(self.base_url)
        self.timeout = timeout if timeout is not None else float(os.environ.get(CLOUD_TIMEOUT_ENV) or DEFAULT_CLOUD_TIMEOUT)
        self.model = (model or os.environ.get(CLOUD_MODEL_ENV) or DEFAULT_CLOUD_MODEL).strip() or DEFAULT_CLOUD_MODEL
        self.force_left = force_left
        self.num_ctx = num_ctx
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.think = think
        self.image_limit_bytes = image_limit_bytes
        self.user_agent = (user_agent or os.environ.get(CLOUD_USER_AGENT_ENV) or DEFAULT_CLOUD_USER_AGENT).strip()
        self.last_request_payload = None

        if not self.base_url:
            raise ValueError(
                f"cloud API base URL is not configured; set {CLOUD_API_BASE_URL_ENV} in .env "
                "or pass url=..."
            )

    def build_prompt(self, image_path: str, detection_result: Dict[str, Any]) -> str:
        score = detection_result.get("score", 0.0)
        threshold = detection_result.get("threshold", 0.5)
        individual_scores = detection_result.get("individual_scores", {})
        detector = detection_result.get("detector") or detection_result.get("detector_names")
        return f"""
当前车辆已经触发长尾检测，请结合图片和检测信息返回一个车辆动作决策。

可选 command：
- left：左变道避让
- right：右变道避让

检测信息：
- image_path: {image_path}
- long_tail_score: {float(score):.3f}
- threshold: {float(threshold):.3f}
- detector: {detector}
- individual_scores: {json.dumps(individual_scores, ensure_ascii=False)}

请严格只返回如下 JSON 结构：
{{"command":"left|right","action":"lane-left|lane-right","reason":"简短中文原因"}}
""".strip()

    def build_payload(self, image_path: str, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.build_prompt(image_path, detection_result)},
                        self._build_image_content_part(image_path),
                    ],
                },
            ],
            "stream": False,
            "stream_options": {"include_usage": False},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": ["END"],
            "max_completion_tokens": self.max_completion_tokens,
            "max_tokens": self.max_completion_tokens,
            "n": 1,
            "response_format": {"type": "text"},
            "num_ctx": self.num_ctx,
            "think": self.think,
        }

    def request_decision(self, image_path: str, detection_result: Dict[str, Any]) -> CloudDecision:
        total_start = time.monotonic()
        payload_start = time.monotonic()
        payload = self.build_payload(image_path, detection_result)
        self.last_request_payload = payload
        payload_latency_ms = (time.monotonic() - payload_start) * 1000
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
                "ngrok-skip-browser-warning": "true",
            },
            method="POST",
        )

        http_start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"cloud API request failed ({exc.code}): {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cloud API request failed: {exc}") from exc

        http_latency_ms = (time.monotonic() - http_start) * 1000
        parse_start = time.monotonic()
        decision = self.parse_response(body, latency_ms=0.0)
        parse_latency_ms = (time.monotonic() - parse_start) * 1000
        decision.latency_ms = (time.monotonic() - total_start) * 1000
        decision.timings_ms = {
            "payload_build": round(payload_latency_ms, 3),
            "http": round(http_latency_ms, 3),
            "parse": round(parse_latency_ms, 3),
            "total": round(decision.latency_ms, 3),
        }
        return decision

    def parse_response(self, body: str, latency_ms: float = 0.0) -> CloudDecision:
        raw_response = self._load_json(body)
        return self._decision_from_response(raw_response, latency_ms)

    def _decision_from_response(self, raw_response: Dict[str, Any], latency_ms: float) -> CloudDecision:
        parsed = self._extract_payload(raw_response)

        command = self._normalize_command(parsed.get("command"))
        action = self._normalize_action(parsed.get("action")) or COMMAND_TO_ACTION.get(command)
        reason = parsed.get("reason") or parsed.get("raw_text") or "云端未提供原因"

        if self.force_left:
            command = "left"
            action = "lane-left"

        if command not in COMMAND_TO_ACTION:
            raise ValueError(f"cloud response did not contain a supported lane-change command: {parsed}")
        expected_action = COMMAND_TO_ACTION[command]
        if action != expected_action:
            action = expected_action

        return CloudDecision(
            command=command,
            action=action,
            reason=str(reason),
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

    def _build_image_content_part(self, image_path: str) -> Dict[str, Any]:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"image not found for cloud request: {image_path}")
        size = path.stat().st_size
        if size > self.image_limit_bytes:
            limit_mb = self.image_limit_bytes / (1024 * 1024)
            raise ValueError(f"image exceeds cloud limit: {size} bytes > {limit_mb:.1f}MB")

        mime_type, _encoding = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }

    def _load_json(self, body: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"raw_text": body}
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    def _extract_payload(self, response: Dict[str, Any]) -> Dict[str, Any]:
        direct_command = self._normalize_command(response.get("command"))
        if direct_command:
            return {
                "command": direct_command,
                "action": response.get("action"),
                "reason": response.get("reason") or response.get("raw_output"),
            }

        text = self._extract_text(response)
        if text:
            parsed_text = self._parse_text_payload(text)
            if parsed_text:
                return parsed_text

        return {"raw_text": "云端响应中没有可解析的左/右变道动作"}

    def _extract_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            for item in value:
                text = self._extract_text(item)
                if text:
                    return text
        if isinstance(value, dict):
            for key in ("response", "message", "content", "text", "answer", "raw_output", "raw_text"):
                text = self._extract_text(value.get(key))
                if text:
                    return text
            if "choices" in value:
                text = self._extract_text(value["choices"])
                if text:
                    return text
            if "data" in value:
                text = self._extract_text(value["data"])
                if text:
                    return text
        return None

    def _parse_text_payload(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        parsed = self._try_parse_json_object(cleaned)
        if parsed:
            return parsed

        command = self._normalize_command(cleaned)
        if command:
            return {
                "command": command,
                "action": COMMAND_TO_ACTION.get(command),
                "reason": cleaned,
            }

        return {"raw_text": cleaned}

    def _try_parse_json_object(self, text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_command(self, value: Any) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip().lower()
        if text in COMMAND_TO_ACTION:
            return text
        aliases = {
            "left": ("left", "lane-left", "左", "左变道", "左边道", "向左"),
            "right": ("right", "lane-right", "右", "右变道", "右边道", "向右"),
        }
        for command, words in aliases.items():
            if any(word in text for word in words):
                return command
        for command in COMMAND_TO_ACTION:
            if re.search(rf"\b{command}\b", text):
                return command
        return None

    def _normalize_action(self, value: Any) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip().lower()
        if text in set(COMMAND_TO_ACTION.values()):
            return text
        command = self._normalize_command(text)
        return COMMAND_TO_ACTION.get(command)

    def _chat_completions_url(self, url: str) -> str:
        normalized = url.rstrip("/")
        if normalized.endswith("/v1/chat/completions"):
            return normalized
        return f"{normalized}/v1/chat/completions"
