from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict

def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()

def manifest_signature(manif_dir: str | Path) -> Dict[str, str]:
    """
    Returns a stable signature dict for {train,val,test} manifests.
    If any file content changes, the signature changes -> cache invalidates.
    """
    manif_dir = Path(manif_dir)
    out = {}
    for split in ("train", "val", "test"):
        p = manif_dir / f"{split}.jsonl"
        if not p.exists():
            out[split] = "missing"
            continue
        md5 = _file_md5(p)
        size = p.stat().st_size
        try:
            with p.open("r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
        except Exception:
            n_lines = -1
        out[split] = f"{md5}:{size}:{n_lines}"
    # single combined key
    combo = hashlib.md5("".join(out[k] for k in ("train","val","test")).encode()).hexdigest()[:10]
    out["combo"] = combo
    return out
