import json
from pathlib import Path
from typing import Any


def save_json(data: Any, filepath: str) -> None:
    """
    Save Python data as pretty JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load JSON data from disk.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)