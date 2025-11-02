from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from src.core.cache import cache_dir, save_json
from src.features.text_clean import bulk_clean
from src.core.data_sig import manifest_signature

__all__ = ["build_or_load_sentence_embeddings"]

def build_or_load_sentence_embeddings(
    manifests_dir: str | Path,
    processed_dir: str | Path,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 256,
    batch_size: int = 64,
    normalize: bool = True,
    lowercase: bool = True,
    cache: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Encode texts with a sentence-transformers model and cache (X,y) per split.
    The cache key includes model_id, max_length, and lowercase flag.
    """
    from sentence_transformers import SentenceTransformer
    import torch
    from src.core.data_io import read_manifest, iter_text_label

    sig = manifest_signature(manifests_dir)["combo"]
    params = dict(model=model_id, maxlen=max_length, lowercase=lowercase, data=sig, ver=1)
    cdir = cache_dir(processed_dir, "sent_embed", params)
    meta_path = Path(cdir) / "meta.json"

    if cache and (Path(cdir) / "train_X.npy").exists():
        return {s: (np.load(Path(cdir)/f"{s}_X.npy"), np.load(Path(cdir)/f"{s}_y.npy")) for s in ["train","val","test"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)

    def encode_texts(texts):
        texts = bulk_clean(texts, lowercase=lowercase)
        X = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return X.astype(np.float32, copy=False)

    def encode_split(split: str):
        items = read_manifest(Path(manifests_dir) / f"{split}.jsonl")
        texts, ys = zip(*list(iter_text_label(items)))
        X = encode_texts(list(texts))
        y = np.asarray(ys, dtype=np.int64)
        Path(cdir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cdir) / f"{split}_X.npy", X)
        np.save(Path(cdir) / f"{split}_y.npy", y)

    for s in ["train", "val", "test"]:
        encode_split(s)

    save_json(
        {"params": params, "dim": int(np.load(Path(cdir)/"train_X.npy").shape[1])},
        meta_path,
    )
    return {s: (np.load(Path(cdir)/f"{s}_X.npy"), np.load(Path(cdir)/f"{s}_y.npy")) for s in ["train","val","test"]}
