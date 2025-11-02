from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Dict, Any

def key_from_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.md5(s).hexdigest()[:10]

def cache_dir(root: str | Path, name: str, params: Dict[str, Any]) -> Path:
    human = "__".join(f"{k}-{v}" for k, v in params.items() if isinstance(v, (int, float, str, bool)))
    return Path(root) / name / f"{human}__{key_from_dict(params)}"

def save_json(obj: Dict, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
