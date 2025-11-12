"""运行配置 diff 工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_run_config(run_dir: Path) -> Tuple[Dict[str, Any], str | None]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    config = payload.get("config", {})
    fingerprint = payload.get("_meta", {}).get("fingerprint")
    return config, fingerprint


def diff_configs(
    config_a: Any,
    config_b: Any,
    prefix: str = "",
) -> List[str]:
    diffs: List[str] = []

    if isinstance(config_a, dict) and isinstance(config_b, dict):
        keys = sorted(set(config_a.keys()) | set(config_b.keys()))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in config_a:
                diffs.append(f"{path}: <missing> -> {format_value(config_b[key])}")
            elif key not in config_b:
                diffs.append(f"{path}: {format_value(config_a[key])} -> <missing>")
            else:
                diffs.extend(diff_configs(config_a[key], config_b[key], path))
        return diffs

    if isinstance(config_a, list) and isinstance(config_b, list):
        length = max(len(config_a), len(config_b))
        for idx in range(length):
            path = f"{prefix}[{idx}]"
            if idx >= len(config_a):
                diffs.append(f"{path}: <missing> -> {format_value(config_b[idx])}")
            elif idx >= len(config_b):
                diffs.append(f"{path}: {format_value(config_a[idx])} -> <missing>")
            else:
                diffs.extend(diff_configs(config_a[idx], config_b[idx], path))
        return diffs

    if config_a != config_b:
        diffs.append(f"{prefix}: {format_value(config_a)} -> {format_value(config_b)}")

    return diffs


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

