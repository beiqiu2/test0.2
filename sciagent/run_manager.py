"""运行管家：负责创建运行目录、执行命令、采集日志。"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from rich.console import Console
from rich.table import Table

from .config_parser import merge_external_configs, normalize_config_tree, parse_train_command

DEFAULT_TAIL_LINES = 200


class RunManager:
    """封装一次训练运行的生命周期。"""

    def __init__(
        self,
        command: str,
        workdir: Path,
        name: str,
        *,
        tail_lines: int = DEFAULT_TAIL_LINES,
        encoding: str = "utf-8",
        extra_config_files: Optional[Iterable[Path]] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.command = command
        self.workdir = workdir.resolve()
        self.name = name or "run"
        self.tail_lines = max(1, tail_lines)
        self.encoding = encoding
        self.extra_config_files = list(extra_config_files or [])
        self.console = console or Console()

        self.run_dir = self._prepare_run_directory()
        self.stdout_log_path = self.run_dir / "stdout.log"
        self.tail_log_path = self.run_dir / "logs_tail.txt"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.command_path = self.run_dir / "cmd.txt"
        self.env_path = self.run_dir / "env.json"
        self.fingerprint_path = self.run_dir / "fingerprint.txt"

    # --------------------------------------------------------------------- #
    # public api
    # --------------------------------------------------------------------- #
    def execute(self) -> int:
        """执行训练命令并同步产出运行文件。"""
        tokens, cli_config = parse_train_command(self.command)
        merged_config = merge_external_configs(cli_config, self._detect_config_files(cli_config))
        normalized_config = normalize_config_tree(merged_config)

        fingerprint = self._calculate_fingerprint(normalized_config)

        self._write_text(self.command_path, self.command + os.linesep)
        self._write_json(self.config_path, self._wrap_config(tokens, normalized_config, fingerprint))
        self._write_json(self.env_path, collect_environment_snapshot())
        self._write_text(self.fingerprint_path, fingerprint + os.linesep)

        tail_buffer: deque[str] = deque(maxlen=self.tail_lines)
        metrics_fp = self.metrics_path.open("a", encoding="utf-8")
        stdout_fp = self.stdout_log_path.open("a", encoding=self.encoding, buffering=1)

        table = self._build_run_header(tokens)
        self.console.print(table)
        self.console.rule()

        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding=self.encoding,
        )

        return_code = 0

        try:
            assert process.stdout is not None
            for line in process.stdout:
                stdout_fp.write(line)
                tail_buffer.append(line)
                self._write_tail(tail_buffer)
                self.console.print(line.rstrip())
                self._try_append_metrics(metrics_fp, line)
            return_code = process.wait()
        except KeyboardInterrupt:
            self.console.print("[bold yellow]捕获到 KeyboardInterrupt，正在请求训练进程停止...[/]")
            self._terminate_process(process)
            return_code = 130
        finally:
            stdout_fp.close()
            metrics_fp.close()
            if process.poll() is None:
                process.kill()

        status_message = "[bold green]运行完成[/]" if return_code == 0 else f"[bold red]运行结束，退出码 {return_code}[/]"
        self.console.rule(status_message)

        return return_code

    # ------------------------------------------------------------------ utils
    def _prepare_run_directory(self) -> Path:
        runs_root = (self.workdir / "runs").resolve()
        runs_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = slugify(self.name)
        run_dir = runs_root / f"{timestamp}_{base_name}"

        suffix = 1
        while run_dir.exists():
            run_dir = runs_root / f"{timestamp}_{base_name}-{suffix}"
            suffix += 1

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _wrap_config(self, tokens: Iterable[str], normalized_config: Dict[str, Any], fingerprint: str) -> Dict[str, Any]:
        return {
            "_meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "working_directory": os.getcwd(),
                "fingerprint": fingerprint,
            },
            "command": " ".join(tokens),
            "program": tokens[0] if tokens else None,
            "arguments": list(tokens[1:]),
            "config": normalized_config,
        }

    def _detect_config_files(self, cli_config: Dict[str, Any]) -> Iterable[Path]:
        """扫描 CLI 参数中常见的配置文件字段。"""
        candidates = []
        for key in ("config", "cfg", "yaml", "yml"):
            value = cli_config.get(key)
            if isinstance(value, str):
                candidates.append(Path(value))
        for path in self.extra_config_files:
            candidates.append(path)
        return candidates

    def _try_append_metrics(self, metrics_fp, line: str) -> None:
        """尝试识别 JSON 行并写入指标流。"""
        stripped = line.strip()
        if not stripped:
            return
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                return
            metrics_fp.write(stripped + "\n")

    def _write_tail(self, tail_buffer: deque[str]) -> None:
        with self.tail_log_path.open("w", encoding="utf-8") as fp:
            fp.writelines(tail_buffer)

    def _write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def _build_run_header(self, tokens: Iterable[str]) -> Table:
        table = Table(title="SciAgent Run Summary", show_header=False, box=None)
        table.add_column("field", style="cyan", no_wrap=True)
        table.add_column("value", style="white")
        table.add_row("Run Dir", str(self.run_dir))
        table.add_row("Command", self.command)
        table.add_row("Tokens", " ".join(tokens))
        return table

    def _terminate_process(self, process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def _calculate_fingerprint(self, config: Dict[str, Any]) -> str:
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def slugify(raw: str) -> str:
    allowed = []
    for ch in raw:
        if ch.isalnum():
            allowed.append(ch.lower())
        elif ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    return slug or "run"


def collect_environment_snapshot() -> Dict[str, Any]:
    """收集最小必要的环境信息。"""
    snapshot: Dict[str, Any] = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "os": os.name,
            "cwd": os.getcwd(),
        },
    }
    git_info = _try_collect_git_info()
    if git_info:
        snapshot["git"] = git_info
    return snapshot


def _try_collect_git_info() -> Dict[str, Any] | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
    return {"commit": commit, "branch": branch}


