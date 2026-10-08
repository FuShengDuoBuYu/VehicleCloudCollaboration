"""Immutable encoded frame snapshots for asynchronous submission."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageFrame:
    data: bytes = field(repr=False)
    name: str = 'frame.jpg'

    def __post_init__(self):
        if not isinstance(self.data,bytes) or not self.data: raise ValueError('ImageFrame requires nonempty immutable bytes')
        if not isinstance(self.name,str) or not self.name: raise ValueError('ImageFrame requires a name')
