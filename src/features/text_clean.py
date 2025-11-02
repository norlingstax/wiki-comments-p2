from __future__ import annotations
import re
from typing import Iterable, List, Optional

__all__ = ["minimal_clean", "bulk_clean"]

_WS_RE = re.compile(r"\s+", re.UNICODE)

def minimal_clean(text: Optional[str], lowercase: bool = True, strip: bool = True) -> str:
    """
    Minimal cleaning:
      - optional lowercasing
      - collapse whitespace
      - strip leading/trailing spaces
    Avoid aggressive steps (stemming, lemmatisation, punctuation removal),
    since they hurt transformers and embedding models.
    """
    if text is None:
        return ""
    t = text
    if lowercase:
        t = t.lower()
    t = _WS_RE.sub(" ", t)
    if strip:
        t = t.strip()
    return t

def bulk_clean(texts: Iterable[str], **kwargs) -> List[str]:
    """Vectorised convenience wrapper for lists/iterables."""
    return [minimal_clean(t, **kwargs) for t in texts]
