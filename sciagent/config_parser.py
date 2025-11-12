"""训练命令配置解析与规范化工具。"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

Primitive = Dict[str, Any]


def split_command(command: str) -> List[str]:
    """按照当前平台规则解析命令字符串为 tokens。"""
    posix = os.name != "nt"
    try:
        return shlex.split(command, posix=posix)
    except ValueError:
        # 回退到较宽松的拆分策略
        return command.strip().split()


def parse_train_command(command: str) -> Tuple[List[str], Dict[str, Any]]:
    """解析训练命令，返回 tokens 与粗略的配置字典。"""
    tokens = split_command(command)
    config = parse_cli_tokens(tokens[1:] if tokens else [])
    return tokens, config


def parse_cli_tokens(tokens: Iterable[str]) -> Dict[str, Any]:
    """将形如 --key value / key=value 的参数列表解析为嵌套字典。"""
    result: Dict[str, Any] = {}
    pending_key: str | None = None

    for token in tokens:
        if pending_key is not None:
            set_deep_key(result, pending_key, coerce_value(token))
            pending_key = None
            continue

        if token.startswith("--"):
            key, value = split_key_value(token[2:])
            if value is None:
                # 下一个 token 作为值；若没有则视为 True
                pending_key = key
            else:
                set_deep_key(result, key, coerce_value(value))
        elif token.startswith("-") and len(token) > 1:
            key, value = split_key_value(token[1:])
            if value is None:
                pending_key = key
            else:
                set_deep_key(result, key, coerce_value(value))
        else:
            # 裸值不会直接写入配置
            continue

    if pending_key:
        set_deep_key(result, pending_key, True)

    return result


def split_key_value(part: str) -> Tuple[str, str | None]:
    """解析 key=value 形式，返回键与可能的值。"""
    if "=" in part:
        key, value = part.split("=", 1)
        return key.strip(), value.strip()
    return part.strip(), None


def set_deep_key(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """支持 a.b.c=1 的嵌套赋值。"""
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        return

    node = target
    for name in parts[:-1]:
        if name not in node or not isinstance(node[name], dict):
            node[name] = {}
        node = node[name]
    node[parts[-1]] = value


def coerce_value(value: str) -> Any:
    """将字符串值转换为 int/float/bool/None 等原始类型。"""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    try:
        if value.startswith("0") and value != "0":
            raise ValueError  # 保留形如 001 的字符串
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def merge_external_configs(cli_config: Dict[str, Any], config_files: Iterable[Path]) -> Dict[str, Any]:
    """将 CLI 配置与 YAML/JSON 配置文件合并（后者优先级更高）。"""
    merged = dict(cli_config)

    for config_path in config_files:
        if not config_path.exists():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                if config_path.suffix.lower() in {".yaml", ".yml"}:
                    loaded = yaml.safe_load(fp)  # type: ignore[assignment]
                else:
                    loaded = json.load(fp)
            if isinstance(loaded, dict):
                deep_update(merged, loaded)
        except Exception:
            continue

    return merged


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """递归合并字典。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value


# --------------------------------------------------------------------------- #
# 配置规范化与指纹
# --------------------------------------------------------------------------- #
SUSPECT_PATH_KEYS = {"path", "dir", "directory", "root", "checkpoint", "ckpt"}


def normalize_config_tree(payload: Any) -> Any:
    """对配置树进行稳定化处理，便于 fingerprint 与 diff。

    规则：
    - dict：按 key 排序生成新 dict。
    - list/tuple：逐项规范化。
    - float：统一为 10 位小数后去除尾随 0。
    - 字符串：若为绝对路径则只保留最后一段；去除多余空白。
    - 其他原始类型保持不变。
    """

    if isinstance(payload, dict):
        normalized: Dict[str, Any] = {}
        for key in sorted(payload.keys()):
            value = payload[key]
            normalized[key] = normalize_config_tree(_sanitize_value(key, value))
        return normalized

    if isinstance(payload, (list, tuple)):
        return [normalize_config_tree(_sanitize_value(None, item)) for item in payload]

    if isinstance(payload, float):
        return _normalize_float(payload)

    return payload


def _sanitize_value(key: str | None, value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if (key and _is_path_key(key)) or _looks_like_path(stripped):
            try:
                return Path(stripped).name
            except Exception:
                return Path(stripped).name if "/" in stripped or "\\" in stripped else stripped
        return stripped

    if isinstance(value, float):
        return _normalize_float(value)

    if isinstance(value, dict):
        return normalize_config_tree(value)

    if key and isinstance(key, str) and "seed" in key.lower():
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return value

    return value


def _normalize_float(value: float) -> float | str:
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    # 若格式化后为空字符串，说明是 0
    if not formatted:
        formatted = "0"
    try:
        return float(formatted)
    except ValueError:
        return formatted


def _looks_like_path(text: str) -> bool:
    if not text:
        return False
    path = Path(text)
    if path.is_absolute():
        return True
    # 包含分隔符也视为路径
    return "/" in text or "\\" in text


def _is_path_key(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in SUSPECT_PATH_KEYS)


