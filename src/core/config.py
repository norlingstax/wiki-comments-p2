from __future__ import annotations
import os
import copy
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

def _load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow+recursive merge (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

@dataclass
class Config:
    raw: Dict[str, Any]
    path: Path

    def get(self, *keys, default=None):
        cur = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def seed(self) -> int:
        return int(self.get("runtime", "seed", default=1337))

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

def load_config(cfg_path: str | os.PathLike) -> Config:
    """
    Supports 'inherit' key in child YAML (relative to repo root).
    Example:
      inherit: configs/default.yaml
    """
    cfg_path = Path(cfg_path)
    child = _load_yaml(cfg_path)
    inherit = child.get("inherit")
    if inherit:
        base = _load_yaml(inherit)
        merged = _merge_dicts(base, child)
    else:
        merged = child
    return Config(merged, cfg_path.resolve())
