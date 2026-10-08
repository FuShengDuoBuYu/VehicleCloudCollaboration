"""Offline contract tests: no API billing, camera, or actuator imports."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CAR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAR_DIR))

from cloud_client import CloudClient


def scene():
    return {
        "scene_summary": "前方通道被纸箱挡住",
        "road_state": "blocked",
        "objects": [{"category": "纸箱", "position": "center",
                     "blocks_corridor": True, "evidence": "覆盖当前走廊"}],
        "signs": [], "risk_level": "high", "recommendation": "stop",
        "route_hint": "none", "uncertainties": [], "reason": "当前走廊无法通行",
    }


def completion(result=None, finish="stop"):
    return json.dumps({
        "id": "chat-test", "model": "qwen3.8-max-0902",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": json.dumps(result or scene())}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 150, "total_tokens": 1050},
    })


class SceneClientContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "frame.jpg"
        # The client transports bytes; image decoding is the provider's job.
        self.image.write_bytes(b"\xff\xd8\xfftest-image")
        self.env_file = Path(self.temp.name) / "absent.env"
        self.env_file.write_text("", encoding="utf-8")

    def client(self, **kwargs):
        settings={'provider':'qwen','model':'qwen3.8-max','contract':'road-scene-v1'};settings.update(kwargs)
        return CloudClient(env_file=self.env_file, **settings)

    def test_qwen_payload_uses_vision_schema_and_no_legacy_parameters(self):
        with patch.dict(os.environ, {}, clear=True):
            client = self.client()
        payload = client.build_payload([self.image], {"vehicle_stopped": True})
        self.assertEqual(payload["model"], "qwen3.8-max")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertIn("stop", payload["response_format"]["json_schema"]["schema"]["properties"]["recommendation"]["enum"])
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        for name in ("num_ctx", "think", "stream_options", "max_completion_tokens", "stop"):
            self.assertNotIn(name, payload)
        self.assertNotIn("command 只能是 left 或 right", payload["messages"][0]["content"])

    def test_endpoint_supports_base_and_full_qwen_or_other_provider(self):
        cases = [
            ("https://dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            ("https://example.com/v1/chat/completions", "https://example.com/v1/chat/completions"),
            ("https://ark.cn-beijing.volces.com/api/v3", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"),
            ("https://example.com", "https://example.com/v1/chat/completions"),
        ]
        for base, endpoint in cases:
            with self.subTest(base=base):
                self.assertEqual(self.client(url=base).url, endpoint)

    def test_generic_provider_does_not_receive_qwen_reasoning_parameter(self):
        client = self.client(provider="openai-compatible", url="https://example.com/v1", model="other-vision")
        payload = client.build_payload(self.image, {})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["model"], "other-vision")

    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            client = self.client(api_key="")
        with patch("urllib.request.OpenerDirector.open") as network:
            with self.assertRaisesRegex(ValueError, "API key"):
                client.request_scene(self.image, {})
            network.assert_not_called()

    def test_placeholder_key_is_rejected_before_network(self):
        for key in ("YOUR_API_KEY", "<YOUR_API_KEY>", "REPLACE_WITH_YOUR_API_KEY"):
            with self.subTest(key=key):
                with patch("urllib.request.OpenerDirector.open") as network:
                    with self.assertRaisesRegex(ValueError, "API key"):
                        self.client(api_key=key).request_scene(self.image, {})
                    network.assert_not_called()

    def test_accepts_complete_valid_scene_and_records_usage(self):
        result = self.client().parse_response(completion())
        self.assertEqual(result.scene["recommendation"], "stop")
        self.assertEqual(result.response_model, "qwen3.8-max-0902")
        self.assertEqual(result.usage["prompt_tokens"], 900)
        self.assertFalse(hasattr(result, "action"))

    def test_rejects_partial_refusal_and_legacy_response(self):
        cases = [completion(finish="length"), "not json", completion({"command": "left", "action": "lane-left", "reason": "x"}),
                 json.dumps({"choices": [{"finish_reason": "stop", "message": {"refusal": "refused", "content": None}}]})]
        for body in cases:
            with self.subTest(body=body[:70]):
                with self.assertRaises(ValueError):
                    self.client().parse_response(body)

    def test_rejects_missing_extra_and_wrong_typed_fields(self):
        bad = []
        value = scene(); del value["road_state"]; bad.append(value)
        value = scene(); value["pwm"] = 80; bad.append(value)
        value = scene(); value["objects"][0]["blocks_corridor"] = "false"; bad.append(value)
        value = scene(); value["uncertainties"] = "none"; bad.append(value)
        value = scene(); value["recommendation"] = "left"; bad.append(value)
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.client().parse_response(completion(value))

    def test_rejects_contradictory_resume_advice(self):
        value = scene(); value["recommendation"] = "resume_candidate"
        with self.assertRaisesRegex(ValueError, "resume"):
            self.client().parse_response(completion(value))

    def test_rejects_duplicate_json_fields(self):
        envelope = json.loads(completion())
        envelope["choices"][0]["message"]["content"] = json.dumps(scene()).replace(
            '"recommendation": "stop"', '"recommendation": "resume_candidate", "recommendation": "stop"')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.client().parse_response(json.dumps(envelope))

    def test_rejects_overflow_json_number_in_usage(self):
        body = completion().replace('"prompt_tokens": 900', '"prompt_tokens": 1e999')
        with self.assertRaisesRegex(ValueError, "finite"):
            self.client().parse_response(body)

    def test_legacy_client_is_explicit_and_remains_compatible(self):
        from cloud_client import LegacyCloudClient
        client = LegacyCloudClient(url="inprocess://test")
        old = {"command": "left", "action": "lane-left", "reason": "legacy fixture"}
        decision = client.parse_response(completion(old))
        self.assertEqual(decision.action, "lane-left")
        self.assertFalse(hasattr(self.client(), "request_decision"))

    def test_multiframe_payload_keeps_order_and_context_and_aggregate_limit(self):
        second = self.image.with_name("second.png"); second.write_bytes(b"second-frame")
        payload = self.client().build_payload([self.image, second], {"frame_timestamps": [1, 2]})
        parts = payload["messages"][1]["content"]
        self.assertEqual(len(parts), 3)
        self.assertIn('"frame_timestamps": [1, 2]', parts[0]["text"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(parts[2]["image_url"]["url"].startswith("data:image/png;base64,"))
        with self.assertRaises(ValueError):
            self.client(image_limit_mb=0.00001).build_payload([self.image, second], {})

    def test_env_file_preserves_explicit_environment_and_key_is_not_in_repr(self):
        file = Path(self.temp.name) / "cloud.env"
        file.write_text('CAR_CLOUD_PROVIDER="qwen"\nCAR_CLOUD_API_KEY="file-test-key"\nCAR_CLOUD_MODEL="file-model"\n', encoding="utf-8")
        with patch.dict(os.environ, {"CAR_CLOUD_MODEL": "env-model"}, clear=True):
            client = CloudClient(env_file=file)
        self.assertEqual(client.model, "env-model")
        self.assertNotIn("file-test-key", repr(client.config))

    def test_explicit_numeric_override_wins_over_invalid_environment(self):
        with patch.dict(os.environ, {"CAR_CLOUD_TIMEOUT": "invalid"}, clear=True):
            client = self.client(timeout=1.5)
        self.assertEqual(client.config.timeout, 1.5)

    def test_unknown_road_cannot_produce_route_candidate(self):
        value = scene()
        value.update(road_state="unknown", risk_level="low", recommendation="route_candidate", route_hint="left")
        with self.assertRaisesRegex(ValueError, "route"):
            self.client().parse_response(completion(value))

    def test_rejects_unsafe_endpoint_and_unknown_provider(self):
        for url in ("http://example.com/v1", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "https://{WorkspaceId}.example.com/v1"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    self.client(url=url)
        with self.assertRaises(ValueError):
            self.client(provider="typo")


if __name__ == "__main__":
    unittest.main()
