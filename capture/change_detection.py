import imagehash
from PIL import Image

def compute_phash(image: Image.Image) -> str:
    """
    Computes a perceptual hash (pHash) of an image.
    This is useful for detecting near-identical frames even if slight compression artifacts occur.
    """
    return str(imagehash.phash(image))

def is_novel_frame(phash_new: str, phash_old: str, threshold: int = 5) -> bool:
    """
    Compares two pHash strings. Returns True if the difference exceeds the threshold,
    indicating the frame is sufficiently different (novel).
    """
    if not phash_new or not phash_old:
        return True
    
    try:
        hash_new = imagehash.hex_to_hash(phash_new)
        hash_old = imagehash.hex_to_hash(phash_old)
        return (hash_new - hash_old) > threshold
    except ValueError:
        # If there's a parsing issue with the hash string, default to treating it as novel
        return True
