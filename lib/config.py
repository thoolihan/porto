import json
from pathlib import Path

def get_config():
    cfg = Path("config.json")
    return json.loads(cfg.read_text())