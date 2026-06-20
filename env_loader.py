import os
from typing import Dict


# Values inserted into os.environ by load_dotenv(). This lets settings.py
# distinguish user-level .env defaults from real process environment overrides.
DOTENV_VALUES: Dict[str, str] = {}


def load_dotenv(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            DOTENV_VALUES[key] = value
