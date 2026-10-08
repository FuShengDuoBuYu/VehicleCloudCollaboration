"""Small provider adapters; the road semantics contract stays provider independent."""
class QwenProvider:
    default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen3.8-omni-flash"
    default_contract = "road-scene-v1"
    key_env = "DASHSCOPE_API_KEY"

    @staticmethod
    def configure_payload(payload, config):
        payload["reasoning_effort"] = config.reasoning_effort
        if 'omni' in config.model:
            payload['modalities'] = ['text']


class QwenRealtimeProvider:
    default_url = QwenProvider.default_url
    default_model = 'qwen3.8-omni-flash-realtime'
    default_contract = 'road-observation-fast-v1'
    key_env = 'DASHSCOPE_API_KEY'


class OpenAICompatibleProvider:
    default_url = ""
    default_model = ""
    key_env = "CAR_CLOUD_API_KEY"
    default_contract = "road-scene-v1"

    @staticmethod
    def configure_payload(payload, config):
        # Do not forward Qwen-specific generation controls to other services.
        pass


PROVIDERS = {"qwen-realtime": QwenRealtimeProvider, "qwen": QwenProvider, "openai-compatible": OpenAICompatibleProvider}


def get_provider(name):
    if name not in PROVIDERS:
        raise ValueError("unsupported cloud provider; use qwen-realtime, qwen or openai-compatible")
    return PROVIDERS[name]
