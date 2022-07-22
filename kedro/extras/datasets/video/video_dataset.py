"""``VideoDataSet`` loads/saves video data from an underlying
filesystem (e.g.: local, S3, GCS). It uses OpenCV VideoCapture to read
and decode videos and OpenCV VideoWriter to encode and write video.
"""
import os
import tempfile
from collections import abc
from pathlib import PurePosixPath
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union

import cv2
import fsspec
import more_itertools
import numpy as np
import PIL.Image

from kedro.io.core import AbstractDataSet, get_protocol_and_path


class SlicedVideo:
    """A representation of slices of other video types"""

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


class AbstractVideo(abc.Sequence):
    """Base class for the underlying video data"""

    _n_frames = 0
    _index = 0  # Next available frame

    @property
    def fourcc(self) -> str:
        """Get the codec fourcc specification"""
        raise NotImplementedError()

    @property
    def fps(self) -> float:
        """Get the video frame rate"""
        raise NotImplementedError()

    @property
    def size(self) -> Tuple[int, int]:
        """Get the resolution of the video"""
        raise NotImplementedError()

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, index: Union[int, slice]):
        """Get a frame from the video"""
        raise NotImplementedError()


class FileVideo(AbstractVideo):
    """A video object read from a file"""

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._cap = cv2.VideoCapture(filepath)
        self._n_frames = self._get_length()

    @property
    def fourcc(self) -> str:
        fourcc = self._cap.get(cv2.CAP_PROP_FOURCC)
        return int(fourcc).to_bytes(4, "little").decode("ascii")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def size(self) -> Tuple[int, int]:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return SlicedVideo(self, index)

        if index < 0:
            index += len(self)
        if index >= len(self):
            raise IndexError()

        if index != self._index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        self._index = index + 1  # Next frame to decode after this
        ret, frame_bgr = self._cap.read()
        if not ret:
            raise IndexError()

        height, width = frame_bgr.shape[:2]
        return PIL.Image.frombuffer(  # Convert to PIL image with RGB instead of BGR
            "RGB", (width, height), frame_bgr, "raw", "BGR", 0, 0
        )

    def _get_length(self) -> int:
        # OpenCV's frame count might be an approximation depending on what
        # headers are available in the video file
        length = int(round(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if length >= 0:
            return length

        # On some video files it fails, we can count the frames but it will be
        # very slow... It also seems like that opencv has issues to seek in those
        # videos...
        raise ValueError("Failed to load video since number of frames can't be infered")


class SequenceVideo(AbstractVideo):
    """A video object read from an indexable sequence of frames"""

    def __init__(
        self, frames: Sequence[PIL.Image.Image], fps: float, fourcc: str = "mp4v"
    ) -> None:
        self._n_frames = len(frames)
        self._frames = frames
        self._fourcc = fourcc
        self._size = frames[0].size
        self._fps = fps

    @property
    def fourcc(self) -> str:
        return self._fourcc

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
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
        fourcc: str = "mp4v",
    ) -> None:
        self._n_frames = length
        self._gen = more_itertools.peekable(frames)
        self._fourcc = fourcc
        self._size = self._gen.peek().size
        self._fps = fps

    @property
    def fourcc(self) -> str:
        return self._fourcc

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def __getitem__(self, index: Union[int, slice]):
        raise NotImplementedError("Underlying video is a generator")

    def __next__(self):
        return next(self._gen)

    def __iter__(self):
        return self


class VideoDataSet(AbstractDataSet):
    """``VideoDataSet`` loads / save video data from a given filepath as sequence
    of PIL.Image.Image using OpenCV.

    Example:
    ::

        >>> video VideoDataSet(filepath='/img/file/path.mp4').load()
        >>> frame = video[0]
        >>> import numpy as np
        >>> np.sum(np.asarray(frame))
    """

    def __init__(self, filepath: str, fourcc: Optional[str] = "mp4v"):
        """Creates a new instance of VideoDataSet to load / save video data for given filepath.

        Args:
            filepath: The location of the video file to load / save data.
            fourcc: The codec to use when writing video, note that depending on how opencv is
                    installed there might be more or less codecs avaiable. If set to None, the
                    fourcc from the video object will be used.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fourcc = fourcc

    def _load(self) -> AbstractVideo:
        """Loads data from the video file.

        Returns:
            Data from the video file as a AbstractVideo object
        """
        with fsspec.open(
            f"filecache::{self._protocol}://{self._filepath}", mode="rb"
        ) as fs_file:
            return FileVideo(fs_file.name)

    def _save(self, data: AbstractVideo) -> None:
        """Saves video data to the specified filepath."""
        if self._protocol == "file":
            # Write directly to the local file destination
            self._write_to_filepath(data, str(self._filepath))
        else:
            # VideoWriter cant write to an open file object, instead write to a
            # local tmpfile and then copy that to the destination with fsspec
            tmp_file =  tempfile.NamedTemporaryFile(
                suffix=self._filepath.suffix, mode="w+b"
            ) 
            self._write_to_filepath(data, tmp_file.name)
            with fsspec.open(
                f"{self._protocol}://{self._filepath}", "wb"
            ) as f_target:
                f_target.write(tmp_file.read())
            os.unlink(tmp_file.name)  # Clean up after fsspec upload

    def _write_to_filepath(self, video: AbstractVideo, filepath: str) -> None:
        # TODO: This uses the codec specified in the VideoDataSet if it is not None, this is due
        # to compatibility issues since e.g. h264 coded is licensed and is thus not included in
        # opencv if installed from a binary distribution. Since a h264 video can be read, but not
        # written, it would be error prone to use the videos fourcc code. Further, an issue is
        # that the video object does not know what container format will be used since that is
        # selected by the suffix in the file name of the VideoDataSet. Some combinations of codec
        # and container format might not work or will have bad support.
        fourcc = self._fourcc or video.fourcc

        writer = cv2.VideoWriter(
            filepath, cv2.VideoWriter_fourcc(*fourcc), video.fps, video.size
        )
        if not writer.isOpened():
            raise ValueError(
                "Failed to open video writer with params: "
                + f"fourcc={fourcc} fps={video.fps} size={video.size[0]}x{video.size[1]} "
                + f"path={filepath}"
            )
        try:
            for frame in iter(video):
                writer.write(  # PIL images are RGB, opencv expects BGR
                    np.asarray(frame)[:, :, ::-1]
                )
        finally:
            writer.release()

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, protocol=self._protocol)

    def _exists(self) -> bool:
        fsystem = fsspec.filesystem(self._protocol)
        return fsystem.exists(self._filepath)
