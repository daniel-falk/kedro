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
        self.index = 0  # Next available frame

    @property
    def fourcc(self):
        fourcc = self._cap.get(cv2.CAP_PROP_FOURCC)
        return int(fourcc).to_bytes(4, "little").decode("ascii")

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def size(self):
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            return SlicedVideo(self, index)

        if index < 0:
            index += len(self)

        if index != self.index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.index = index + 1  # Next frame to decode after this
        ret, frame_bgr = self._cap.read()
        if not ret:
            raise IndexError()
        height, width = frame_bgr.shape[:2]
        return PIL.Image.frombuffer(
            "RGB", (width, height), frame_bgr, "raw", "BGR", 0, 0
        )

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

    def _save(self, video: Video) -> None:
        """Saves image data to the specified filepath.

        TODO: This can only save video from video file, i.e. one cant modify the video.
        Create subclasses of Video so that videos can be created from an iterable of frames or
        from a generator yielding frames.
        """
        save_path = get_filepath_str(self._filepath, self._protocol)
        # TODO: This assumes that the output file has the same fourcc code as the input file,
        # this might not be the case since we can use one VideoDataSet to read e.g. a mp4 file with H264 video
        # and then save it to another VideoDataSet which should use an .avi file with MJPEG
        # TODO: There is no way to use the OpenVN VideoWrite to write to an open file, so it does not
        # work together with fsspec. Investigate this further...
        writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*video.fourcc), video.fps, video.size
        )
        try:
            for frame in video:
                writer.write(  # PIL images are RGB, opencv expects BGR
                    np.array(frame)[:, :, ::-1]
                )
        finally:
            writer.release()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
