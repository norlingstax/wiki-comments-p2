from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DatasetSplits:
    train_path: Path
    val_path: Path
    test_path: Path

def _manifest_paths(manif_dir: Path) -> DatasetSplits:
    """Return paths for train/val/test manifests."""
    return DatasetSplits(
        train_path=manif_dir / "train.jsonl",
        val_path=manif_dir / "val.jsonl",
        test_path=manif_dir / "test.jsonl",
    )

def load_raw_csv(path: str | Path, text_col: str, label_col: str, drop_na_text: bool = True) -> pd.DataFrame:
    """Load raw CSV and return DataFrame with text and label columns."""
    df = pd.read_csv(path)
    if drop_na_text:
        df = df[df[text_col].notna()].copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]

def make_or_load_splits(
    raw_csv: str | Path,
    manif_dir: str | Path,
    test_size: float,
    val_size: float,
    stratify: bool = True,
    random_state: int = 1337,
) -> DatasetSplits:
    """Make or load train/val/test splits."""
    manif_dir = Path(manif_dir)
    manif_dir.mkdir(parents=True, exist_ok=True)
    paths = _manifest_paths(manif_dir)

    if paths.train_path.exists() and paths.val_path.exists() and paths.test_path.exists():
        return paths
    
    # Load raw data
    raw_csv = Path(raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    df = pd.read_csv(raw_csv)
    
    # First split: train+val vs test
    y = df["label"] if stratify else None
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    # Second split: train vs val
    val_size_abs = math.ceil(len(df_trainval) * val_size)
    y_tv = df_trainval["label"] if stratify else None
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_abs,
        random_state=random_state,
        stratify=y_tv if stratify else None,
    )
    # Write manifests
    def _to_jsonl(d: pd.DataFrame, p: Path):
        with p.open("w", encoding="utf-8") as f:
            for r in d.itertuples(index=False):
                f.write(json.dumps({"text": r.text, "label": int(r.label)}, ensure_ascii=False) + "\n")

    _to_jsonl(df_train, paths.train_path)
    _to_jsonl(df_val, paths.val_path)
    _to_jsonl(df_test, paths.test_path)
    return paths

def read_manifest(path: str | Path) -> List[dict]:
    """Read JSONL manifest and return list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def iter_text_label(items: Iterable[dict]) -> Iterable[Tuple[str, int]]:
    """Iterate over text-label pairs."""
    for it in items:
        yield it["text"], int(it["label"])
