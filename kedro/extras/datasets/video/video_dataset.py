from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Generator, Union

from kedro.io.core import (
    AbstractDataSet,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import cv2
import PIL.Image
import numpy as np
import more_itertools


class SlicedVideo:
    def __init__(self, video, slice_indexes):
        self.video = video
        self.indexes = range(*slice_indexes.indices(len(video)))

    def __getitem__(self, index: Union[int, slice]) -> PIL.Image.Image:
        if isinstance(index, slice):
            return SlicedVideo(self, index)
        return self.video[self.indexes[index]]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getattr__(self, item):
        return getattr(self.video, item)


class AbstractVideo:
    def __init__(self):
        raise NotImplementedError("AbstractVideo should be extended")

    @property
    def fourcc(self):
        raise NotImplementedError()

    @property
    def fps(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        return int(round(self._n_frames))


class FileVideo(AbstractVideo):
    """A video object read from a file"""

    def __init__(self, filepath: str) -> None:
        self._cap = cv2.VideoCapture(filepath)
        # OpenCV's frame count might be an approximation depending on what
        # headers are available in the video file
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

    def __getitem__(self, index: Union[int, slice]):
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
        return PIL.Image.frombuffer(  # Convert to PIL image with RGB instead of BGR
            "RGB", (width, height), frame_bgr, "raw", "BGR", 0, 0
        )


class IterableVideo(AbstractVideo):
    """A video object read from an iterable of frames"""

    def __init__(
        self, frames: Iterable[PIL.Image.Image], fps: float, fourcc: str = "avc1"
    ) -> None:
        self._n_frames = len(frames)
        self._frames = frames
        self._fourcc = fourcc
        self._size = frames[0].size
        self._fps = fps
        self.index = 0  # Next available frame

    @property
    def fourcc(self):
        return self._fourcc

    @property
    def fps(self):
        return self._fps

    @property
    def size(self):
        return self._size

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return SlicedVideo(self, index)
        return self._frames[index]


class GeneratorVideo(AbstractVideo):
    """A video object with frames yielded by a generator"""

    def __init__(
        self,
        frames: Generator[PIL.Image.Image, None, None],
        length,
        fps: float,
        fourcc: str = "avc1",
    ) -> None:
        self._n_frames = length
        self._gen = more_itertools.peekable(frames)
        self._fourcc = fourcc
        self._size = self._gen.peek().size
        self._fps = fps

    @property
    def fourcc(self):
        return self._fourcc

    @property
    def fps(self):
        return self._fps

    @property
    def size(self):
        return self._size

    def __next__(self):
        return next(self._gen)

    def __iter__(self):
        return self


class VideoDataSet(AbstractDataSet):

    """``VideoDataSet`` loads / save video data from a given filepath as sequence of PIL.Image.Image using OpenCV.

    Example:
    ::

        >>> video VideoDataSet(filepath='/img/file/path.mp4').load()
        >>> frame = video[0]
        >>> import numpy as np
        >>> np.sum(np.asarray(frame))
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

    def _load(self) -> AbstractVideo:
        """Loads data from the video file.

        Returns:
            Data from the video file as a AbstractVideo object
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="r") as f:
            return FileVideo(load_path)

    def _save(self, video: AbstractVideo) -> None:
        """Saves image data to the specified filepath.
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
                    np.asarray(frame)[:, :, ::-1]
                )
        finally:
            writer.release()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)
