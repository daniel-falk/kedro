from pathlib import PurePosixPath
from typing import Any, Dict

from kedro.io.core import (
    AbstractDataSet,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import cv2
import PIL.Image
import numpy as np


# TODO: Implement save


class SlicedVideo:
    def __init__(self, video, slice_indexes):
        self.video = video
        self.indexes = range(*slice_indexes.indices(len(video)))

    def __getitem__(self, index) -> PIL.Image:
        if isinstance(index, slice):
            return SlicedVideo(self, index)
        return self.video[self.indexes[index]]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getattr__(self, item):
        return getattr(self.video, item)


class Video:
    def __init__(self, filepath: str) -> None:
        self._cap = cv2.VideoCapture(filepath)
        self._n_frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.index = 0 # Next available frame

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            return SlicedVideo(self, index)

        if index < 0:
            index += len(self)

        if index != self.index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.index = index + 1  # Next frame to decode after this
        ret, frame_bgr = self._cap.read()
        height, width = frame_bgr.shape[:2]
        return PIL.Image.frombuffer("RGB", (width, height), frame_bgr, "raw", "BGR", 0, 0)

    def __len__(self) -> int:
        # OpenCV's frame count might be an approximation depending on what
        # headers are available in the video file
        return int(round(self._n_frames))


class VideoDataSet(AbstractDataSet):

    """``VideoDataSet`` loads / save video data from a given filepath as sequence of PIL.Image using OpenCV.

    Example:
    ::

        >>> video VideoDataSet(filepath='/img/file/path.mp4').load()
        >>> frame = video[0]
        >>> import numpy as np
        >>> np.sum(np.array(frame))
    """

    def __init__(self, filepath: str):
        """Creates a new instance of VideoDataSet to load / save video data for given filepath.

        Args:
            filepath: The location of the video file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> Video:
        """Loads data from the video file.

        Returns:
            Data from the video file as a Video object
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            return Video(load_path)

    def _save(self, data: PIL.Image) -> None:
        """Saves image data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            raise NotImplementedError("Saving video is not yet supported")

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
