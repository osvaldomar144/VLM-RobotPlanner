"""
Shared pytest fixtures.
"""

import pytest
from PIL import Image


@pytest.fixture
def synthetic_scene_image(tmp_path):
    """
    A minimal 224x224 RGB image that simulates a scene photo.
    Generated programmatically — no external file dependency.
    Enough to test PIL loading and message construction without a real camera.
    """
    img = Image.new("RGB", (224, 224), color=(120, 80, 60))
    path = tmp_path / "scene.jpg"
    img.save(path)
    return path


@pytest.fixture
def synthetic_pil_image():
    """A PIL Image directly (no file I/O)."""
    return Image.new("RGB", (224, 224), color=(200, 150, 100))
