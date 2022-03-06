"""Dataset implementation to load/save data from/to a video file."""

__all__ = ["VideoDataSet"]

from contextlib import suppress

with suppress(ImportError):
    from .video_dataset import VideoDataSet
