# accounts/model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass(slots=True, frozen=True)
class Account:
    path: str
    username: str
    password: str
    email: str
    user_agent: str
    state: str = "active"

    @property
    def cookies_path(self) -> str:
        return str(Path(self.path).with_name("cookies.json"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))  # compact
