"""SciAgent CLI 入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .diff_utils import diff_configs, load_run_config
from .run_manager import RunManager

app = typer.Typer(add_completion=False, help="RunGuardian / SciPilot 训练守护 CLI")


@app.command()
def run(
    cmd: str = typer.Option(..., "--cmd", help="要执行的训练命令，使用引号包裹整条命令"),
    workdir: Path = typer.Option(Path("./exp"), "--workdir", help="输出目录（将创建 runs/<id>）"),
    name: str = typer.Option("run", "--name", help="自定义运行名称"),
    tail_lines: int = typer.Option(200, "--tail-lines", help="logs_tail.txt 中保留的最新行数"),
    encoding: str = typer.Option("utf-8", "--encoding", help="训练进程 stdout 编码"),
    config: Optional[list[Path]] = typer.Option(
        None,
        "--config-file",
        "-c",
        help="附加的配置文件路径（可多次指定，优先级高于 CLI 参数）",
    ),
) -> None:
    """包裹执行训练命令，产出标准化运行目录。"""
    console = Console()
    manager = RunManager(
        cmd,
        workdir,
        name,
        tail_lines=tail_lines,
        encoding=encoding,
        extra_config_files=config,
        console=console,
    )
    exit_code = manager.execute()
    raise typer.Exit(code=exit_code)


@app.command()
def diff(
    run_a: Path = typer.Argument(..., help="运行目录 A（包含 config.json）"),
    run_b: Path = typer.Argument(..., help="运行目录 B"),
) -> None:
    """对比两次运行的配置差异。"""
    console = Console()
    try:
        config_a, fingerprint_a = load_run_config(run_a)
        config_b, fingerprint_b = load_run_config(run_b)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc))

    changes = diff_configs(config_a, config_b)

    console.rule("[bold cyan]Config Diff[/]")
    console.print(f"[bold]A:[/] {run_a}")
    console.print(f"[bold]B:[/] {run_b}")
    console.print(f"[bold]Fingerprint A:[/] {fingerprint_a or 'N/A'}")
    console.print(f"[bold]Fingerprint B:[/] {fingerprint_b or 'N/A'}")
    console.print()

    if not changes:
        console.print("[green]配置无差异[/]")
    else:
        for line in changes:
            console.print(f"- {line}")


if __name__ == "__main__":
    app()


