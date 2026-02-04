"""
Command Line Interface for Deep Research Tool.
"""

import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

from .config import Config, LLMProvider, SearchMethod, ReportFormat, create_config
from .main import DeepResearchTool


console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   Deep Research Tool                          ║
║           Automated Research with AI Assistance               ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold blue")


def print_config_summary(config: Config):
    """Print configuration summary."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("LLM Provider", config.api.provider.value)
    table.add_row("Model", config.api.get_active_model())
    table.add_row("Search Method", config.search.method.value)
    table.add_row("Research Iterations", str(config.research.min_iterations))
    table.add_row("Output Format", config.report.format.value)
    table.add_row("Output Directory", str(config.report.output_dir))

    # Show output length targets if set
    if config.report.target_pages:
        table.add_row("Target Pages", str(config.report.target_pages))
    if config.report.target_characters:
        table.add_row("Target Characters", f"{config.report.target_characters:,}")

    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Deep Research Tool - Automated research with AI assistance."""
    pass


@cli.command()
@click.argument("query")
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider to use"
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model name (e.g., gpt-4o-mini, claude-3-5-sonnet)"
)
@click.option(
    "--search", "-s",
    type=click.Choice(["duckduckgo", "selenium"]),
    default="duckduckgo",
    help="Web search method"
)
@click.option(
    "--iterations", "-i",
    type=int,
    default=3,
    help="Number of research iterations per section"
)
@click.option(
    "--output-format", "-f",
    type=click.Choice(["markdown", "docx", "pdf", "html"]),
    default="markdown",
    help="Output report format"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="./output",
    help="Output directory"
)
@click.option(
    "--requirements", "-r",
    default="",
    help="Additional research requirements"
)
@click.option(
    "--documents", "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="Additional documents to include (PDF, DOCX, etc.)"
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Enable/disable verification"
)
@click.option(
    "--target-pages",
    type=int,
    default=None,
    help="Target page count for output (approximate, e.g., 10 for ~10 pages)"
)
@click.option(
    "--target-characters",
    type=int,
    default=None,
    help="Target character count for output"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key"
)
@click.option(
    "--anthropic-key",
    envvar="ANTHROPIC_API_KEY",
    help="Anthropic API key"
)
def research(
    query: str,
    provider: str,
    model: Optional[str],
    search: str,
    iterations: int,
    output_format: str,
    output_dir: str,
    requirements: str,
    documents: tuple,
    verify: bool,
    target_pages: Optional[int],
    target_characters: Optional[int],
    verbose: bool,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
):
    """
    Conduct research on a given query.

    QUERY is the research topic or question to investigate.

    Example:
        deep-research research "AI trends in healthcare 2024"
    """
    print_banner()

    # Create configuration
    config = create_config(
        provider=provider,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        model=model,
        search_method=search,
        research_iterations=iterations,
        output_format=output_format,
        output_dir=output_dir,
        additional_documents=list(documents) if documents else None,
        enable_verification=verify,
        verbose=verbose,
        target_pages=target_pages,
        target_characters=target_characters,
    )

    # Validate config
    errors = config.validate()
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  - {error}", style="red")
        sys.exit(1)

    print_config_summary(config)

    # Confirm before starting
    if not click.confirm("Start research with these settings?"):
        console.print("Cancelled.", style="yellow")
        return

    try:
        # Initialize tool
        tool = DeepResearchTool(config)

        # Progress callback
        def progress_callback(message: str, percentage: float):
            if percentage >= 0:
                console.print(f"[{percentage:5.1f}%] {message}")
            else:
                console.print(f"[ERROR] {message}", style="red")

        # Conduct research
        console.print("\n[bold green]Starting research...[/bold green]\n")

        result = tool.run(
            query=query,
            requirements=requirements,
            progress_callback=progress_callback,
        )

        # Print results summary
        console.print("\n")
        console.print(Panel(
            f"[bold green]Research completed![/bold green]\n\n"
            f"Session ID: {result['session_id']}\n"
            f"Report: {result['report_path']}\n"
            f"Evidence: {result.get('evidence_json', 'N/A')}\n"
            f"Verification: {result.get('verification_html', 'N/A')}",
            title="Results",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option(
    "--output-format", "-f",
    type=click.Choice(["markdown", "docx", "pdf", "html"]),
    default=None,
    help="Output report format (default: use original)"
)
@click.option(
    "--target-pages",
    type=int,
    default=None,
    help="Target page count for output (approximate)"
)
@click.option(
    "--target-characters",
    type=int,
    default=None,
    help="Target character count for output"
)
def report(
    session_path: str,
    output_format: Optional[str],
    target_pages: Optional[int],
    target_characters: Optional[int],
):
    """
    Generate a report from a saved research session.

    SESSION_PATH is the path to the session JSON file.

    Examples:
        deep-research report session.json --output-format pdf
        deep-research report session.json --target-pages 10
        deep-research report session.json --target-characters 25000
    """
    print_banner()

    try:
        from .research.researcher import ResearchSession
        from .evidence.locker import EvidenceLocker
        from .report.generator import ReportGenerator, ReportFormat

        # Load session
        session = ResearchSession.load(Path(session_path))
        console.print(f"Loaded session: {session.session_id}")

        # Try to load evidence
        session_dir = Path(session_path).parent
        evidence_path = session_dir / "evidence" / f"evidence_{session.session_id}.json"

        if evidence_path.exists():
            evidence_locker = EvidenceLocker.load_from_json(evidence_path)
        else:
            evidence_locker = EvidenceLocker(research_id=session.session_id)
            console.print("[yellow]Warning: Evidence file not found[/yellow]")

        # Determine format
        if output_format:
            fmt = ReportFormat(output_format)
        else:
            fmt = ReportFormat.MARKDOWN

        # Generate report
        generator = ReportGenerator(output_dir=session_dir / "reports")

        # Show current length info before adjustment
        length_info = generator.get_length_info(session, fmt)
        console.print(f"Current content: {length_info.total_characters:,} characters (~{length_info.estimated_pages:.1f} pages)")

        if target_pages or target_characters:
            target_desc = f"{target_pages} pages" if target_pages else f"{target_characters:,} characters"
            console.print(f"Target: {target_desc}")

        report_path = generator.generate_report(
            session=session,
            evidence_locker=evidence_locker,
            format=fmt,
            target_pages=target_pages,
            target_characters=target_characters,
        )

        console.print(f"\n[green]Report generated: {report_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("report_path", type=click.Path(exists=True))
@click.option(
    "--evidence", "-e",
    type=click.Path(exists=True),
    help="Path to evidence JSON file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output path for verification report"
)
@click.option(
    "--strictness", "-s",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Verification strictness level"
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider for verification"
)
def verify(
    report_path: str,
    evidence: Optional[str],
    output: Optional[str],
    strictness: str,
    provider: str,
):
    """
    Verify a research report for potential hallucinations.

    REPORT_PATH is the path to the report file to verify.
    """
    print_banner()

    try:
        from .api import get_client
        from .evidence.locker import EvidenceLocker
        from .verification.verifier import Verifier

        # Read report content
        report_path = Path(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        console.print(f"Verifying: {report_path.name}")
        console.print(f"Content length: {len(content)} characters")

        # Load or create evidence locker
        if evidence:
            evidence_locker = EvidenceLocker.load_from_json(Path(evidence))
        else:
            evidence_locker = EvidenceLocker()

        # Initialize verifier
        llm_client = get_client(provider)
        verifier = Verifier(llm_client)

        console.print("\n[bold]Running verification...[/bold]\n")

        # Run verification
        result = verifier.verify_content(
            content=content,
            evidence_locker=evidence_locker,
            document_title=report_path.stem,
            strictness=strictness,
        )

        # Print summary
        table = Table(title="Verification Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Claims", str(result.total_claims))
        table.add_row("Reliability Score", f"{result.overall_reliability_score:.1%}")
        table.add_row("High Confidence", str(result.high_confidence_count))
        table.add_row("Medium Confidence", str(result.medium_confidence_count))
        table.add_row("Low Confidence", str(result.low_confidence_count))
        table.add_row("Unsupported", str(result.unsupported_count))
        table.add_row("Hallucination Risks", str(result.hallucination_risk_count))

        console.print(table)

        # Generate HTML report
        if output:
            output_path = Path(output)
        else:
            output_path = report_path.parent / f"{report_path.stem}_verification.html"

        verifier.generate_verification_report_html(result, output_path)
        console.print(f"\n[green]Verification report: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command("add-figures")
@click.argument("session_path", type=click.Path(exists=True))
@click.option(
    "--report", "-r",
    type=click.Path(exists=True),
    help="Path to report file (auto-detected if not specified)"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="Output directory for figures"
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include images from web sources"
)
@click.option(
    "--include-tables/--no-tables",
    default=True,
    help="Include extracted tables"
)
@click.option(
    "--include-charts/--no-charts",
    default=True,
    help="Include generated charts"
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default=None,
    help="LLM provider for content analysis (optional)"
)
@click.option(
    "--max-images",
    type=int,
    default=2,
    help="Maximum images per section"
)
def add_figures(
    session_path: str,
    report: Optional[str],
    output_dir: Optional[str],
    include_images: bool,
    include_tables: bool,
    include_charts: bool,
    provider: Optional[str],
    max_images: int,
):
    """
    Add figures and tables to an existing report.

    SESSION_PATH is the path to the session JSON file.

    This command extracts figures from referenced web sources and
    creates tables/charts from numerical data in the content.

    Examples:
        deep-research add-figures session.json
        deep-research add-figures session.json --include-charts
        deep-research add-figures session.json --provider openai
    """
    print_banner()

    try:
        from .research.researcher import ResearchSession
        from .evidence.locker import EvidenceLocker
        from .report.figure_table_generator import (
            FigureTableGenerator,
            add_figures_to_report,
        )

        # Load session
        session_path = Path(session_path)
        session = ResearchSession.load(session_path)
        console.print(f"Loaded session: {session.session_id}")

        # Find report file if not specified
        session_dir = session_path.parent
        if report:
            report_path = Path(report)
        else:
            # Look for report in common locations
            possible_reports = [
                session_dir / "reports" / f"research_report_{session.session_id}.md",
                session_dir / f"research_report_{session.session_id}.md",
                session_dir / "reports" / f"research_report_{session.session_id}.html",
            ]
            report_path = None
            for p in possible_reports:
                if p.exists():
                    report_path = p
                    break

            if not report_path:
                console.print("[red]Error: Could not find report file. Please specify with --report[/red]")
                sys.exit(1)

        console.print(f"Report file: {report_path}")

        # Load evidence
        evidence_path = session_dir / "evidence" / f"evidence_{session.session_id}.json"
        if evidence_path.exists():
            evidence_locker = EvidenceLocker.load_from_json(evidence_path)
            console.print(f"Loaded evidence: {len(evidence_locker.get_all_evidence())} items")
        else:
            evidence_locker = EvidenceLocker(research_id=session.session_id)
            console.print("[yellow]Warning: Evidence file not found[/yellow]")

        # Create LLM client if provider specified
        llm_client = None
        if provider:
            from .api import get_client
            llm_client = get_client(provider)
            console.print(f"Using {provider} for content analysis")

        # Set output directory
        if output_dir:
            figures_dir = Path(output_dir)
        else:
            figures_dir = report_path.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[bold]Generating figures and tables...[/bold]")
        console.print(f"  - Images: {'Yes' if include_images else 'No'}")
        console.print(f"  - Tables: {'Yes' if include_tables else 'No'}")
        console.print(f"  - Charts: {'Yes' if include_charts else 'No'}")
        console.print(f"  - Max images per section: {max_images}")
        console.print()

        # Create generator
        generator = FigureTableGenerator(
            llm_client=llm_client,
            output_dir=figures_dir,
            language="ja",
            max_images_per_section=max_images,
        )

        # Generate figures and tables
        collection = generator.generate_figures_and_tables(
            session=session,
            evidence_locker=evidence_locker,
            include_images=include_images,
            include_tables=include_tables,
            include_charts=include_charts,
        )

        # Show summary
        table = Table(title="Generated Content")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Figures (Images)", str(len(collection.figures)))
        table.add_row("Tables", str(len(collection.tables)))
        table.add_row("Charts", str(len(collection.charts)))

        console.print(table)

        # Add to report if markdown
        if report_path.suffix == '.md':
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            updated_content = generator.add_figures_to_markdown(content, collection)

            output_path = report_path.parent / f"{report_path.stem}_with_figures.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            console.print(f"\n[green]Updated report: {output_path}[/green]")
        else:
            console.print(f"\n[yellow]Note: Automatic insertion only supported for Markdown.[/yellow]")
            console.print(f"Figures saved to: {figures_dir}")

        # Export collection metadata
        collection_path = figures_dir / "figures_tables.json"
        generator.export_collection(collection, collection_path)
        console.print(f"Collection metadata: {collection_path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
def info():
    """Display information about the tool and configuration."""
    print_banner()

    # Check dependencies
    console.print("[bold]Checking dependencies...[/bold]\n")

    deps = [
        ("openai", "OpenAI API"),
        ("anthropic", "Anthropic API"),
        ("duckduckgo_search", "DuckDuckGo Search"),
        ("selenium", "Selenium"),
        ("docx", "python-docx"),
        ("pypdf", "PyPDF"),
        ("fitz", "PyMuPDF"),
        ("reportlab", "ReportLab"),
        ("matplotlib", "Matplotlib (Charts)"),
        ("bs4", "BeautifulSoup"),
        ("rich", "Rich"),
        ("click", "Click"),
    ]

    table = Table(title="Dependencies")
    table.add_column("Package", style="cyan")
    table.add_column("Description")
    table.add_column("Status", style="green")

    for module, desc in deps:
        try:
            __import__(module)
            status = "[green]Installed[/green]"
        except ImportError:
            status = "[red]Not installed[/red]"

        table.add_row(module, desc, status)

    console.print(table)

    # Check API keys
    console.print("\n[bold]API Keys:[/bold]")
    import os
    if os.getenv("OPENAI_API_KEY"):
        console.print("  OPENAI_API_KEY: [green]Set[/green]")
    else:
        console.print("  OPENAI_API_KEY: [yellow]Not set[/yellow]")

    if os.getenv("ANTHROPIC_API_KEY"):
        console.print("  ANTHROPIC_API_KEY: [green]Set[/green]")
    else:
        console.print("  ANTHROPIC_API_KEY: [yellow]Not set[/yellow]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
