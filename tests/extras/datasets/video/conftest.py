from pathlib import Path

import pytest
from PIL import Image
from utils import TEST_HEIGHT, TEST_WIDTH

from kedro.extras.datasets.video.video_dataset import FileVideo, SequenceVideo


@pytest.fixture
def color_video():
    red = Image.new("RGB", (TEST_WIDTH, TEST_HEIGHT), (255, 0, 0))
    green = Image.new("RGB", (TEST_WIDTH, TEST_HEIGHT), (0, 255, 0))
    blue = Image.new("RGB", (TEST_WIDTH, TEST_HEIGHT), (0, 0, 255))
    yellow = Image.new("RGB", (TEST_WIDTH, TEST_HEIGHT), (255, 255, 0))
    purple = Image.new("RGB", (TEST_WIDTH, TEST_HEIGHT), (255, 0, 255))
    return SequenceVideo([red, green, blue, yellow, purple], fps=1)


@pytest.fixture
def filepath_mp4():
    return str(Path(__file__).parent / "data/video.mp4")


@pytest.fixture
def filepath_mjpeg():
    return str(Path(__file__).parent / "data/video.mjpeg")


@pytest.fixture
def mp4_object(filepath_mp4):
    return FileVideo(filepath_mp4)


@pytest.fixture
def mjpeg_object(filepath_mjpeg):
    return FileVideo(filepath_mjpeg)
