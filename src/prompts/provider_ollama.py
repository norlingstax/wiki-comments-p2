from __future__ import annotations
import json
from typing import List, Optional
import requests

from src.prompts.parsing_utils import parse_label_score_json, label_to_prob

class OllamaProvider:
    """
    Provider for a locally-running Ollama model.
    Requires: `ollama serve` and a pulled model, e.g.: `ollama pull phi3:mini`
    """
    def __init__(
        self,
        model_id: str = "phi3:mini",
        host: str = "http://localhost:11434",
        timeout_s: int = 120,
    ):
        self.name = "ollama"
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def predict_batch(self, texts: List[str], template: Optional[str] = None) -> List[float]:
        """Call Ollama with a JSON-format prompt and return P(toxic) per text."""
        prompt_template = template or (
            "Classify the comment as 'toxic' or 'non-toxic'. "
            "Return ONLY JSON: {\"label\":\"toxic|non-toxic\",\"score\":0.xx}\n"
            "Comment: {text}\n"
            "Output: "
        )
        url = f"{self.host}/api/generate"
        headers = {"Content-Type": "application/json"}
        out: List[float] = []

        def _render(tpl: str, text: str) -> str:
            escaped = tpl.replace('{', '{{').replace('}', '}}')
            escaped = escaped.replace('{{text}}', '{text}')
            return escaped.format(text=text)

        for t in texts:
            prompt = _render(prompt_template, t)
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
                "format": "json", 
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            txt = data.get("response", "")

            # parse {"label": "...", "score": ...}
            lbl, scr = parse_label_score_json(txt)

            # fallback: if score missing, map label->prob; if both missing, 0.5
            p = scr if scr is not None else label_to_prob(lbl, default=0.5)
            out.append(float(p))
            
        return out
