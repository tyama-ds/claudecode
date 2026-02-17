"""
Command Line Interface for Patent Research Tool.
"""

import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table

from .config import create_patent_config
from .main import PatentResearchTool

console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                Patent Research Tool                          ║
║        AI-Powered Patent Analysis with Multi-Layer Search    ║
╚══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold blue")


def print_config_summary(config):
    """Print configuration summary."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("LLM Provider", config.api.provider.value)
    table.add_row("Model", config.api.get_active_model())
    table.add_row("Language", config.language)
    table.add_row("Output Directory", str(config.report.output_dir))

    # Patent search sources
    sources = []
    if config.patent_search.enable_google_patents:
        sources.append("Google Patents")
    if config.patent_search.enable_jplatpat:
        sources.append("J-PlatPat")
    if config.patent_search.enable_espacenet:
        sources.append("Espacenet")
    table.add_row("Patent DBs", ", ".join(sources))

    # Auxiliary search
    aux = []
    if config.auxiliary.enable_academic_search:
        aux.append("Academic")
    if config.auxiliary.enable_examination_search:
        aux.append("Examination")
    if config.auxiliary.enable_business_search:
        aux.append("Business")
    table.add_row("Auxiliary Search", ", ".join(aux))

    table.add_row("Jurisdictions", ", ".join(config.patent_search.patent_jurisdictions))

    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Patent Research Tool - AI-powered patent analysis with multi-layer search."""
    pass


@cli.command()
@click.argument("query")
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider"
)
@click.option("--model", "-m", default=None, help="Model name")
@click.option(
    "--output-format", "-f",
    type=click.Choice(["markdown", "docx", "pdf", "html"]),
    default="markdown",
    help="Report format"
)
@click.option("--output-dir", "-o", default="./output/patent_research", help="Output directory")
@click.option("--language", "-l", default="ja", help="Output language")
@click.option(
    "--patents",
    multiple=True,
    help="Specific patent numbers to analyze (can be specified multiple times)"
)
@click.option(
    "--ipc",
    multiple=True,
    help="IPC codes to focus on (can be specified multiple times)"
)
@click.option(
    "--jurisdictions", "-j",
    default="JP,US,EP",
    help="Comma-separated jurisdictions to search"
)
@click.option("--requirements", "-r", default="", help="Research requirements")
@click.option("--google-patents/--no-google-patents", default=True, help="Enable Google Patents")
@click.option("--jplatpat/--no-jplatpat", default=True, help="Enable J-PlatPat")
@click.option("--espacenet/--no-espacenet", default=True, help="Enable Espacenet")
@click.option("--academic/--no-academic", default=True, help="Enable academic paper search")
@click.option("--examination/--no-examination", default=True, help="Enable examination doc search")
@click.option("--business/--no-business", default=True, help="Enable business evidence search")
@click.option("--claim-chart/--no-claim-chart", default=True, help="Generate claim chart")
@click.option("--landscape/--no-landscape", default=True, help="Generate technology landscape")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def research(
    query, provider, model, output_format, output_dir, language,
    patents, ipc, jurisdictions, requirements,
    google_patents, jplatpat, espacenet,
    academic, examination, business,
    claim_chart, landscape, verbose,
):
    """Conduct patent research on a given topic."""
    print_banner()

    # Parse jurisdictions
    jurisdiction_list = [j.strip() for j in jurisdictions.split(",") if j.strip()]

    config = create_patent_config(
        provider=provider,
        model=model,
        language=language,
        output_format=output_format,
        output_dir=output_dir,
        verbose=verbose,
        enable_google_patents=google_patents,
        enable_jplatpat=jplatpat,
        enable_espacenet=espacenet,
        patent_jurisdictions=jurisdiction_list,
        ipc_codes=list(ipc),
        enable_academic_search=academic,
        enable_examination_search=examination,
        enable_business_search=business,
        generate_claim_chart=claim_chart,
        generate_landscape=landscape,
    )

    print_config_summary(config)

    # Progress callback
    def progress_callback(message: str, percentage: float):
        if percentage >= 0:
            console.print(f"[{percentage:.0f}%] {message}", style="dim")
        else:
            console.print(f"[!] {message}", style="red")

    try:
        tool = PatentResearchTool(config=config)
        result = tool.run(
            query=query,
            requirements=requirements,
            target_patents=list(patents) if patents else None,
            ipc_focus=list(ipc) if ipc else None,
            progress_callback=progress_callback,
        )

        # Print results
        console.print()
        results_table = Table(title="Research Results", show_header=False)
        results_table.add_column("Item", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Session ID", result["session_id"])
        results_table.add_row("Patents Found", str(result["patents_found"]))
        results_table.add_row("Academic Papers", str(result["academic_papers"]))
        results_table.add_row("Examination Docs", str(result["examination_documents"]))
        results_table.add_row("Business Evidence", str(result["business_evidence"]))
        results_table.add_row("Claim Charts", str(result["claim_charts"]))
        results_table.add_row("Report", result["report_path"])

        console.print(results_table)

    except Exception as e:
        console.print(f"Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
