import sys
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = "config.yaml"

def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "preprocess" not in cfg:
        sys.exit("Config must contain a top‑level 'preprocess' section")
    return cfg

def _read_data(path_pattern: str) -> pd.DataFrame:
    paths = list(Path().glob(path_pattern)) if any(c in path_pattern for c in "*?[") else [Path(path_pattern)]
    if not paths:
        sys.exit(f"No files match pattern: {path_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    # frames = []
    # for p in paths:
    #     try:
    #         df = pd.read_csv(p)
    #     except pd.errors.ParserError as e:
    #         print(
    #             f"Warning: ParserError reading '{p}', retrying with python engine and skipping bad lines: {e}",
    #             file=sys.stderr,
    #         )
    #         df = pd.read_csv(p, engine='python', on_bad_lines='skip')
    #     frames.append(df)
    return pd.concat(frames, ignore_index=True)