from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np

from src.core.cache import cache_dir, save_json
from src.features.text_clean import bulk_clean
from src.core.data_sig import manifest_signature

__all__ = [
    "KeyedVectorsLike",
    "load_vectors",
    "average_embed",
    "build_or_load_wordvec_features",
]

@dataclass
class KeyedVectorsLike:
    vectors: Dict[str, np.ndarray]
    dim: int

    def __contains__(self, token: str) -> bool:
        return token in self.vectors

    def get(self, token: str) -> np.ndarray | None:
        return self.vectors.get(token, None)

def _file_sig(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]

def load_vectors(path: str | Path, limit: int | None = None) -> KeyedVectorsLike:
    """
    Loads text-based word vectors (.txt/.vec) or word2vec binary files.
    Prefers gensim if available; falls back to a simple parser.
    """
    path = Path(path)
    try:
        from gensim.models import KeyedVectors
        binary = path.suffix in {".bin"}  # word2vec binary
        kv = KeyedVectors.load_word2vec_format(str(path), binary=binary, limit=limit, no_header=False)
        dim = kv.vector_size
        vectors = {w: kv.get_vector(w) for w in kv.key_to_index.keys()}
        return KeyedVectorsLike(vectors=vectors, dim=dim)
    except Exception:
        # Fallback: simple text parser for .txt/.vec with header optional
        vectors: Dict[str, np.ndarray] = {}
        dim = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            # Header detection: "n d"
            parts = first.strip().split()
            header_is_count = len(parts) == 2 and all(p.isdigit() for p in parts)
            if not header_is_count:
                # first line is a vector
                line = first
                word, *vals = line.rstrip("\n").split(" ")
                vec = np.asarray([float(x) for x in vals], dtype=np.float32)
                dim = vec.shape[0]
                vectors[word] = vec
            for line in f:
                if not line.strip():
                    continue
                word, *vals = line.rstrip("\n").split(" ")
                if limit is not None and len(vectors) >= limit:
                    break
                vec = np.asarray([float(x) for x in vals], dtype=np.float32)
                if dim is None:
                    dim = vec.shape[0]
                vectors[word] = vec
        if dim is None:
            raise ValueError(f"Could not infer vector dim from {path}")
        return KeyedVectorsLike(vectors=vectors, dim=dim)

def average_embed(
    texts: Iterable[str],
    kv: KeyedVectorsLike,
    lowercase: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Convert texts to vectors by averaging token embeddings.
    OOV tokens are skipped; if all tokens OOV, returns zero vector.
    """
    texts = bulk_clean(texts, lowercase=lowercase)
    out = np.zeros((len(texts), kv.dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = t.split()
        if not toks:
            continue
        vecs = [kv.get(tok) for tok in toks if tok in kv]
        if vecs:
            v = np.vstack(vecs).mean(axis=0)
            out[i] = v
        # else keep zeros
    # normalise to unit length for LR
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.divide(out, np.maximum(norms, epsilon), where=norms > 0)
    return out

def build_or_load_wordvec_features(
    manifests_dir: str | Path,
    processed_dir: str | Path,
    vectors_path: str | Path,
    lowercase: bool = True,
    cache: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns cached (X, y) arrays for train/val/test built from averaged word vectors.
    Cache key includes the vector file signature and lowercase flag.
    """
    from src.core.data_io import read_manifest, iter_text_label

    sig = manifest_signature(manifests_dir)["combo"]
    params = dict(
        vec_sig=_file_sig(vectors_path),
        lowercase=lowercase,
        data=sig,
        ver=1,
    )
    cdir = cache_dir(processed_dir, "wordvec_avg", params)
    meta_path = Path(cdir) / "meta.json"

    if cache and (Path(cdir) / "train_X.npy").exists():
        return {s: (np.load(Path(cdir) / f"{s}_X.npy"), np.load(Path(cdir) / f"{s}_y.npy")) for s in ["train", "val", "test"]}

    kv = load_vectors(vectors_path)

    def encode_split(split: str):
        items = read_manifest(Path(manifests_dir) / f"{split}.jsonl")
        texts, ys = zip(*list(iter_text_label(items)))
        X = average_embed(texts, kv=kv, lowercase=lowercase)
        y = np.asarray(ys, dtype=np.int64)
        Path(cdir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cdir) / f"{split}_X.npy", X)
        np.save(Path(cdir) / f"{split}_y.npy", y)

    for s in ["train", "val", "test"]:
        encode_split(s)

    save_json({"params": params, "dim": kv.dim, "counts": {s: int(np.load(Path(cdir)/f'{s}_X.npy').shape[0]) for s in ["train","val","test"]}}, meta_path)
    
    return {s: (np.load(Path(cdir) / f"{s}_X.npy"), np.load(Path(cdir) / f"{s}_y.npy")) for s in ["train", "val", "test"]}
