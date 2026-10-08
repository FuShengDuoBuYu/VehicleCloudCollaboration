from .client import CloudAPIError, CloudClient, CloudSceneResult
from .config import CloudConfig, DEFAULT_CLOUD_API_BASE_URL, DEFAULT_CLOUD_MODEL
from .frames import ImageFrame
from .worker import LatestSceneWorker, SceneOutcome

__all__ = [
    "CloudClient",
    "CloudSceneResult",
    "CloudConfig",
    "CloudAPIError",
    'ImageFrame','LatestSceneWorker','SceneOutcome',
    "LegacyCloudClient",
    "CloudDecision",
    "DEFAULT_CLOUD_API_BASE_URL",
    "DEFAULT_CLOUD_MODEL",
]


def __getattr__(name):
    # Importing the modern client must not load legacy .env_example defaults.
    if name in {"LegacyCloudClient", "CloudDecision"}:
        from .mock_client import CloudClient as LegacyCloudClient, CloudDecision
        return {"LegacyCloudClient": LegacyCloudClient, "CloudDecision": CloudDecision}[name]
    raise AttributeError(name)
