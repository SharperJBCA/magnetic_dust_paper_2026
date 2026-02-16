from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}
