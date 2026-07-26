"""
Tests for capture/change_detection.py

Covers:
- compute_phash returns a non-empty hex string
- is_novel_frame returns True when hashes differ significantly
- is_novel_frame returns False when hashes are identical
- is_novel_frame returns True when prior hash is None (cold start)
- is_novel_frame returns True on malformed hash strings (safe default)
- Boundary: difference exactly equal to threshold is NOT novel
"""
import pytest
from PIL import Image

from capture.change_detection import compute_phash, is_novel_frame


def _solid_image(color: tuple[int, int, int], size: tuple[int, int] = (100, 100)) -> Image.Image:
    img = Image.new("RGB", size, color=color)
    return img


class TestComputePhash:
    def test_returns_non_empty_string(self):
        img = _solid_image((128, 128, 128))
        result = compute_phash(img)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_identical_images_produce_identical_hash(self):
        img1 = _solid_image((200, 100, 50))
        img2 = _solid_image((200, 100, 50))
        assert compute_phash(img1) == compute_phash(img2)

    def test_very_different_images_produce_different_hashes(self):
        white = _solid_image((255, 255, 255))
        black = _solid_image((0, 0, 0))
        assert compute_phash(white) != compute_phash(black)


class TestIsNovelFrame:
    def test_novel_when_prior_is_none(self):
        img = _solid_image((100, 100, 100))
        phash = compute_phash(img)
        assert is_novel_frame(phash, None, threshold=5) is True

    def test_novel_when_prior_is_empty_string(self):
        img = _solid_image((100, 100, 100))
        phash = compute_phash(img)
        assert is_novel_frame(phash, "", threshold=5) is True

    def test_not_novel_for_identical_frames(self):
        img = _solid_image((100, 100, 100))
        phash = compute_phash(img)
        assert is_novel_frame(phash, phash, threshold=5) is False

    def test_novel_for_very_different_frames(self):
        # Solid images have nearly identical pHashes (uniform DCT coefficients).
        # Use a high-frequency checkerboard to guarantee a large Hamming distance.
        checker = Image.new("RGB", (64, 64))
        pixels = checker.load()
        for y in range(64):
            for x in range(64):
                pixels[x, y] = (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)
        solid = _solid_image((128, 128, 128))
        h_checker = compute_phash(checker)
        h_solid = compute_phash(solid)
        assert is_novel_frame(h_checker, h_solid, threshold=5) is True

    def test_malformed_hash_defaults_to_novel(self):
        """Malformed hashes should be treated as novel (safe default)."""
        assert is_novel_frame("not_a_valid_hash", "also_invalid", threshold=5) is True

    def test_boundary_equal_to_threshold_is_not_novel(self):
        """
        The contract: difference > threshold → novel.
        A difference exactly equal to threshold is NOT considered novel.
        This test documents the boundary explicitly to prevent off-by-one regressions.
        """
        img = _solid_image((128, 128, 128))
        phash = compute_phash(img)
        # Same frame → difference is 0, which is <= any positive threshold
        assert is_novel_frame(phash, phash, threshold=0) is False
