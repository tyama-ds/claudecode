"""Command-line interface for multi-agent discussion tool."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table

from .config import create_config, Config, AgentRole
from .main import MultiAgentDiscussion, run_discussion
from .conversation import DiscussionSession, Message


console = Console()


def print_message(message: Message):
    """Print a message to the console with formatting."""
    style_map = {
        "opening": "bold cyan",
        "closing": "bold cyan",
        "moderation": "yellow",
        "contribution": "green",
        "evaluation": "magenta",
    }
    style = style_map.get(message.message_type.value, "white")

    console.print(f"\n[{style}][{message.agent_name}][/{style}]")
    console.print(message.content)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Multi-Agent Discussion Tool - AI-powered group discussions."""
    pass


@cli.command()
@click.argument("topic")
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider to use"
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model name (uses provider default if not specified)"
)
@click.option(
    "--max-rounds", "-r",
    default=3,
    type=int,
    help="Maximum number of discussion rounds"
)
@click.option(
    "--personas", "-P",
    default=None,
    help="JSON string or file path with participant personas"
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output file for transcript (markdown format)"
)
@click.option(
    "--no-evaluation",
    is_flag=True,
    default=False,
    help="Skip the evaluation phase"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress real-time output"
)
def discuss(
    topic: str,
    provider: str,
    model: Optional[str],
    max_rounds: int,
    personas: Optional[str],
    output: Optional[str],
    no_evaluation: bool,
    quiet: bool,
):
    """
    Start a multi-agent discussion on the given TOPIC.

    Example:
        mad discuss "AIの倫理的な問題について"
        mad discuss "リモートワークの是非" --max-rounds 5
        mad discuss "教育改革" --personas '[{"name": "教師", "persona": "現場の教師として"}]'
    """
    console.print(Panel(f"[bold]議論トピック:[/bold] {topic}", title="Multi-Agent Discussion"))

    # Parse personas
    participant_personas = None
    if personas:
        try:
            if Path(personas).exists():
                with open(personas, "r", encoding="utf-8") as f:
                    participant_personas = json.load(f)
            else:
                participant_personas = json.loads(personas)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            console.print(f"[red]Error parsing personas: {e}[/red]")
            sys.exit(1)

    # Create config
    config = create_config(
        topic=topic,
        provider=provider,
        model=model,
        participant_personas=participant_personas,
        max_rounds=max_rounds,
    )
    config.discussion.enable_evaluation = not no_evaluation

    # Show participants
    console.print("\n[bold]参加者:[/bold]")
    for agent in config.agents:
        role_icon = {
            AgentRole.MODERATOR: "🎙️",
            AgentRole.PARTICIPANT: "💬",
            AgentRole.EVALUATOR: "📊",
        }.get(agent.role, "❓")
        console.print(f"  {role_icon} {agent.name} ({agent.role.value})")

    console.print()

    # Run discussion
    try:
        discussion = MultiAgentDiscussion(config)

        def progress_callback(status: str, progress: float):
            if not quiet:
                console.print(f"[dim]{status} ({progress*100:.0f}%)[/dim]", end="\r")

        def message_callback(message: Message):
            if not quiet:
                print_message(message)

        result = discussion.run(
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        # Print summary
        console.print("\n" + "=" * 50)
        console.print(Panel("[bold green]議論完了[/bold green]"))

        table = Table(title="議論サマリー")
        table.add_column("項目", style="cyan")
        table.add_column("値", style="green")

        table.add_row("セッションID", result["session_id"])
        table.add_row("ラウンド数", str(result["rounds"]))
        table.add_row("メッセージ数", str(result["message_count"]))

        if result["session_path"]:
            table.add_row("セッションファイル", result["session_path"])

        console.print(table)

        # Print evaluation if available
        if result["evaluation"]:
            console.print("\n[bold]評価結果:[/bold]")
            eval_data = result["evaluation"]
            console.print(f"[cyan]サマリー:[/cyan] {eval_data.get('summary', 'N/A')}")
            console.print(f"[cyan]品質スコア:[/cyan] {eval_data.get('quality_score', 0):.2f}")

            if eval_data.get("key_points"):
                console.print("[cyan]主要論点:[/cyan]")
                for point in eval_data["key_points"]:
                    console.print(f"  • {point}")

        # Save transcript
        if output:
            output_path = Path(output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["transcript"])
            console.print(f"\n[green]トランスクリプトを保存しました: {output_path}[/green]")

    except ValueError as e:
        console.print(f"[red]設定エラー: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]エラーが発生しました: {e}[/red]")
        raise


@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "json", "text"]),
    default="markdown",
    help="Output format"
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output file path"
)
def export(session_file: str, format: str, output: Optional[str]):
    """
    Export a discussion session to various formats.

    Example:
        mad export session_abc123.json -f markdown -o discussion.md
    """
    session = DiscussionSession.load(Path(session_file))

    if format == "markdown":
        content = session.generate_transcript()
    elif format == "json":
        content = json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
    else:  # text
        lines = []
        for msg in session.all_messages:
            lines.append(f"[{msg.agent_name}]: {msg.content}\n")
        content = "\n".join(lines)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]エクスポートしました: {output}[/green]")
    else:
        console.print(content)


@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
def info(session_file: str):
    """
    Show information about a discussion session.

    Example:
        mad info session_abc123.json
    """
    session = DiscussionSession.load(Path(session_file))

    table = Table(title=f"セッション情報: {session.session_id}")
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")

    table.add_row("トピック", session.topic)
    table.add_row("状態", session.state.value)
    table.add_row("ラウンド数", str(len(session.rounds)))
    table.add_row("メッセージ数", str(session.message_count))
    table.add_row("参加者", ", ".join(session.participants))
    table.add_row("作成日時", session.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("更新日時", session.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

    console.print(table)

    # Show evaluation if available
    if "evaluation" in session.metadata:
        console.print("\n[bold]評価結果:[/bold]")
        eval_data = session.metadata["evaluation"]
        console.print(f"品質スコア: {eval_data.get('quality_score', 'N/A')}")


@cli.command()
@click.option(
    "--session-dir", "-d",
    default="./discussion_sessions",
    type=click.Path(),
    help="Directory containing session files"
)
def list_sessions(session_dir: str):
    """
    List all discussion sessions in a directory.

    Example:
        mad list-sessions -d ./my_sessions
    """
    dir_path = Path(session_dir)
    if not dir_path.exists():
        console.print(f"[yellow]ディレクトリが存在しません: {session_dir}[/yellow]")
        return

    session_files = list(dir_path.glob("session_*.json"))

    if not session_files:
        console.print("[yellow]セッションが見つかりません[/yellow]")
        return

    table = Table(title="議論セッション一覧")
    table.add_column("ID", style="cyan")
    table.add_column("トピック", style="green")
    table.add_column("状態", style="yellow")
    table.add_column("ラウンド", style="blue")
    table.add_column("作成日時", style="dim")

    for session_file in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            session = DiscussionSession.load(session_file)
            table.add_row(
                session.session_id,
                session.topic[:30] + "..." if len(session.topic) > 30 else session.topic,
                session.state.value,
                str(len(session.rounds)),
                session.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        except Exception as e:
            table.add_row(
                session_file.stem,
                f"[red]Error: {e}[/red]",
                "-",
                "-",
                "-",
            )

    console.print(table)


@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
def replay(session_file: str):
    """
    Replay a discussion session in the terminal.

    Example:
        mad replay session_abc123.json
    """
    session = DiscussionSession.load(Path(session_file))

    console.print(Panel(f"[bold]トピック:[/bold] {session.topic}", title="議論リプレイ"))
    console.print(f"[dim]セッションID: {session.session_id}[/dim]\n")

    for round_obj in session.rounds:
        console.print(f"\n[bold cyan]--- ラウンド {round_obj.round_number} ---[/bold cyan]\n")

        for message in round_obj.all_messages:
            print_message(message)

        if round_obj.summary:
            console.print(f"\n[dim italic]ラウンドまとめ: {round_obj.summary}[/dim italic]")

        # Pause between rounds
        if click.confirm("\n続けますか?", default=True):
            continue
        else:
            break

    console.print("\n[green]リプレイ完了[/green]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
