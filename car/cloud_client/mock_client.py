import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_MOCK_CHAT_URL = "http://lz1pe24291437.vicp.fun/gzt_code_server/llm_interface/chat"

COMMAND_TO_ACTION = {
    "left": "lane-left",
    "right": "lane-right",
    "straight": "forward",
    "stop": "stop",
}

LEFT_LANE_SYSTEM_PROMPT = """
你是车云协同自动驾驶中期联调用的 mock 决策服务。
当前是固定回归测试场景，不需要根据图像真实判断。
你必须只返回 JSON，不要输出 Markdown、解释或多余文本。
""".strip()

LEFT_LANE_EXPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "enum": ["left"]},
        "action": {"type": "string", "enum": ["lane-left"]},
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


class CloudMockClient:
    def __init__(self, url: str = DEFAULT_MOCK_CHAT_URL, timeout: float = 30.0, force_left: bool = True):
        self.url = url
        self.timeout = timeout
        self.force_left = force_left

    def build_prompt(self, image_path: str, detection_result: Dict[str, Any]) -> str:
        score = detection_result.get("score", 0.0)
        threshold = detection_result.get("threshold", 0.5)
        return f"""
当前车辆已经触发长尾检测，需要云端 mock 返回一个控制决策。

中期联调要求：
- 不根据图片内容做真实推理
- 不返回 right、straight 或 stop
- 始终返回向左改道

检测信息：
- image_path: {image_path}
- long_tail_score: {score:.3f}
- threshold: {threshold:.3f}

请严格只返回如下 JSON：
{{"command":"left","action":"lane-left","reason":"云端 mock 固定返回左变道"}}
""".strip()

    def build_payload(self, image_path: str, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": self.build_prompt(image_path, detection_result),
            "system": LEFT_LANE_SYSTEM_PROMPT,
            "expect_format": {
                "command": "left",
                "action": "lane-left",
                "reason": "云端 mock 固定返回左变道",
            },
            "expect_schema": LEFT_LANE_EXPECT_SCHEMA,
            "options": {"temperature": 0},
            "think": False,
            "raw": False,
            "json": True,
        }

    def request_decision(self, image_path: str, detection_result: Dict[str, Any]) -> CloudDecision:
        payload = self.build_payload(image_path, detection_result)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cloud mock request failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        return self.parse_response(body, latency_ms)

    def parse_response(self, body: str, latency_ms: float = 0.0) -> CloudDecision:
        raw_response = self._load_json(body)
        return self._decision_from_response(raw_response, latency_ms)

    def _decision_from_response(self, raw_response: Dict[str, Any], latency_ms: float) -> CloudDecision:
        parsed = self._extract_payload(raw_response)

        command = self._normalize_command(parsed.get("command"))
        action = parsed.get("action") or COMMAND_TO_ACTION.get(command)
        reason = parsed.get("reason") or "云端 mock 固定返回左变道"

        if self.force_left:
            command = "left"
            action = "lane-left"

        return CloudDecision(
            command=command,
            action=action,
            reason=str(reason),
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

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

        return {"command": "left", "action": "lane-left", "reason": "云端 mock 固定返回左变道"}

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
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        command = self._normalize_command(cleaned)
        if command:
            return {
                "command": command,
                "action": COMMAND_TO_ACTION.get(command),
                "reason": cleaned,
            }

        return {}

    def _normalize_command(self, value: Any) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip().lower()
        if text in COMMAND_TO_ACTION:
            return text
        for command in ("left", "right", "straight", "stop"):
            if re.search(rf"\b{command}\b", text):
                return command
        return None
