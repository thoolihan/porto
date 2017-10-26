import json
from pathlib import Path

def get_config():
    cfg = Path("./config.json")
    with cfg.open(mode = 'rt') as fh:
        return json.loads(fh.read())