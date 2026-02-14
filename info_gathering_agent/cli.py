"""
Command-line interface for Information Gathering Agent.
"""

import sys
import json
import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Information Gathering Agent - Automated information collection."""
    pass


@cli.command()
@click.argument("query")
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic", "local"]), default="openai",
              help="LLM provider")
@click.option("--model", "-m", default=None, help="Model name")
@click.option("--search", "-s", type=click.Choice(["duckduckgo", "selenium"]), default="duckduckgo",
              help="Search method")
@click.option("--iterations", "-i", type=int, default=3, help="Research iterations per section")
@click.option("--output-dir", "-o", type=click.Path(), default="./output", help="Output directory")
@click.option("--requirements", "-r", default="", help="Specific research requirements")
@click.option("--language", "-l", default="ja", help="Target language (ISO 639-1)")
@click.option("--extended/--no-extended", default=False, help="Enable extended mode (deep site crawling)")
@click.option("--verbose/--quiet", default=False, help="Verbose output")
@click.option("--output-format", type=click.Choice(["json", "csv", "both"]), default="both",
              help="Evidence export format")
@click.option("--crawl-mode", type=click.Choice(["standard", "fast_batch", "fast_parallel"]),
              default="standard", help="Crawl mode for performance")
@click.option("--content-filter", type=click.Choice(["strict", "moderate", "minimal", "none"]),
              default="moderate", help="Content filter strictness")
def gather(query, provider, model, search, iterations, output_dir, requirements,
           language, extended, verbose, output_format, crawl_mode, content_filter):
    """Gather information on a topic.

    QUERY is the research topic or question to investigate.

    Examples:

        info-gather gather "AI trends in healthcare 2024"

        info-gather gather "再生可能エネルギー市場分析" -p anthropic -i 5 -l ja

        info-gather gather "Machine learning applications" --extended --crawl-mode fast_batch
    """
    from .config import create_config
    from .main import InfoGatheringAgent
    from .api.base import get_token_stats

    try:
        config = create_config(
            provider=provider,
            model=model,
            search_method=search,
            research_iterations=iterations,
            output_dir=output_dir,
            verbose=verbose,
            extended_mode=extended,
            crawl_mode=crawl_mode,
            content_filter_mode=content_filter,
            export_evidence_json=output_format in ("json", "both"),
            export_evidence_csv=output_format in ("csv", "both"),
            language=language,
        )

        agent = InfoGatheringAgent(config)

        def progress_callback(message: str, percentage: float):
            if percentage >= 0:
                click.echo(f"[{percentage:5.1f}%] {message}")

        result = agent.gather(
            query=query,
            requirements=requirements,
            progress_callback=progress_callback if verbose else None,
        )

        # Display results summary
        click.echo("")
        click.echo("=" * 60)
        click.echo("Information Gathering Complete")
        click.echo("=" * 60)
        click.echo(f"Session ID:     {result.session_id}")
        click.echo(f"Query:          {result.query}")
        click.echo(f"Sections:       {len(result.section_summaries)}")
        click.echo(f"Total Sources:  {result.quality_statistics.get('total_sources', 0)}")
        click.echo("")

        # Executive summary
        if result.executive_summary:
            summary_text = result.executive_summary.get("executive_summary", "")
            if summary_text:
                click.echo("Executive Summary:")
                click.echo("-" * 40)
                click.echo(summary_text[:500])
                if len(summary_text) > 500:
                    click.echo("...")
                click.echo("")

            key_findings = result.executive_summary.get("key_findings", [])
            if key_findings:
                click.echo("Key Findings:")
                for i, finding in enumerate(key_findings[:5], 1):
                    click.echo(f"  {i}. {finding}")
                click.echo("")

        # Output files
        click.echo("Output Files:")
        if result.evidence_json_path:
            click.echo(f"  Evidence JSON: {result.evidence_json_path}")
        if result.evidence_csv_path:
            click.echo(f"  Evidence CSV:  {result.evidence_csv_path}")
        if result.session_path:
            click.echo(f"  Session:       {result.session_path}")
        click.echo("")

        # Token usage
        if result.token_usage:
            total = result.token_usage.get("total_tokens", 0)
            if total > 0:
                click.echo(f"Token Usage:    {total:,} tokens")
                click.echo("")

        click.echo("=" * 60)

    except ValueError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nGathering interrupted by user.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic", "local"]), default="openai")
@click.option("--model", "-m", default=None)
@click.option("--max-results", type=int, default=5)
def quick(query, provider, model, max_results):
    """Quick information lookup (no full research loop).

    QUERY is the topic to quickly look up.
    """
    from .config import create_config
    from .main import InfoGatheringAgent

    try:
        config = create_config(provider=provider, model=model)
        agent = InfoGatheringAgent(config)

        result = agent.quick_gather(query=query, max_results=max_results)

        click.echo(f"\nQuery: {result['query']}")
        click.echo(f"Sources: {result['results_count']}")
        click.echo("")

        for source in result["sources"]:
            click.echo(f"  - {source['title']}")
            click.echo(f"    {source['url']}")

        click.echo(f"\nSummary:\n{result['summary']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
