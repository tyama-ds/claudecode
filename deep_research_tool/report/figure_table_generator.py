"""
Figure and Table Generator - Add figures and tables to research reports.

This module provides functionality to:
- Extract relevant images from referenced web pages
- Identify and extract numerical data for tables
- Generate matplotlib charts from extracted data
- Add figures and tables to existing reports
"""

import io
import os
import re
import json
import base64
import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, urljoin

import requests


class FigureType(str, Enum):
    """Types of figures."""
    IMAGE = "image"          # Image from web source
    CHART = "chart"          # Generated matplotlib chart
    DIAGRAM = "diagram"      # Diagram or flowchart


class ChartType(str, Enum):
    """Types of charts that can be generated."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"


@dataclass
class Figure:
    """A figure (image or chart) for the report."""
    figure_id: str
    figure_type: FigureType
    title: str
    caption: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    section_id: str = ""
    image_path: Optional[Path] = None
    image_data: Optional[bytes] = None
    alt_text: str = ""
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "figure_id": self.figure_id,
            "figure_type": self.figure_type.value,
            "title": self.title,
            "caption": self.caption,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "section_id": self.section_id,
            "image_path": str(self.image_path) if self.image_path else None,
            "alt_text": self.alt_text,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TableData:
    """Data for a table."""
    table_id: str
    title: str
    caption: str
    headers: List[str]
    rows: List[List[Any]]
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    section_id: str = ""
    data_type: str = "general"  # general, time_series, comparison
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_id": self.table_id,
            "title": self.title,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "section_id": self.section_id,
            "data_type": self.data_type,
            "unit": self.unit,
        }


@dataclass
class FigureTableCollection:
    """Collection of figures and tables for a report."""
    figures: List[Figure] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    charts: List[Figure] = field(default_factory=list)  # Generated charts

    def get_figures_for_section(self, section_id: str) -> List[Figure]:
        """Get all figures for a section."""
        return [f for f in self.figures if f.section_id == section_id]

    def get_tables_for_section(self, section_id: str) -> List[TableData]:
        """Get all tables for a section."""
        return [t for t in self.tables if t.section_id == section_id]

    def get_charts_for_section(self, section_id: str) -> List[Figure]:
        """Get all charts for a section."""
        return [c for c in self.charts if c.section_id == section_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "figures": [f.to_dict() for f in self.figures],
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts],
        }


class FigureTableGenerator:
    """
    Generate figures and tables for research reports.

    Extracts images from referenced sources and creates tables/charts
    from numerical data found in the content.
    """

    # Common image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}

    # Patterns for numerical data extraction
    NUMBER_PATTERN = re.compile(r'[\d,]+\.?\d*')
    YEAR_PATTERN = re.compile(r'(19|20)\d{2}')
    PERCENTAGE_PATTERN = re.compile(r'(\d+\.?\d*)\s*[%％]')
    CURRENCY_PATTERN = re.compile(r'[\$€¥£]\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*(円|ドル|ユーロ|億|万)')

    def __init__(
        self,
        llm_client=None,
        output_dir: Path = None,
        language: str = "ja",
        max_images_per_section: int = 2,
        image_max_width: int = 800,
        download_timeout: int = 10,
    ):
        """
        Initialize FigureTableGenerator.

        Args:
            llm_client: LLM client for content analysis
            output_dir: Directory to save generated files
            language: Output language
            max_images_per_section: Maximum images per section
            image_max_width: Maximum image width in pixels
            download_timeout: Timeout for downloading images
        """
        self.llm = llm_client
        self.output_dir = output_dir or Path("./output/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.max_images_per_section = max_images_per_section
        self.image_max_width = image_max_width
        self.download_timeout = download_timeout

    def generate_figures_and_tables(
        self,
        session,
        evidence_locker,
        include_images: bool = True,
        include_tables: bool = True,
        include_charts: bool = True,
    ) -> FigureTableCollection:
        """
        Generate figures and tables for all sections.

        Args:
            session: Research session with content
            evidence_locker: Evidence locker with sources
            include_images: Include images from sources
            include_tables: Include extracted tables
            include_charts: Include generated charts

        Returns:
            FigureTableCollection with all figures and tables
        """
        collection = FigureTableCollection()

        # Get section information
        sections = self._get_sections(session)

        for section_id, section_data in sections.items():
            if section_id.startswith("_"):
                continue

            # Get evidence for this section
            section_evidence = evidence_locker.get_section_evidence(section_id)
            content = section_data.get("content", "")

            # Extract images from sources
            if include_images:
                images = self._extract_images_for_section(
                    section_id=section_id,
                    section_data=section_data,
                    evidence_list=section_evidence,
                )
                collection.figures.extend(images)

            # Extract numerical data and create tables
            if include_tables:
                tables = self._extract_tables_for_section(
                    section_id=section_id,
                    section_data=section_data,
                    content=content,
                    evidence_list=section_evidence,
                )
                collection.tables.extend(tables)

            # Generate charts from tables
            if include_charts and collection.tables:
                section_tables = [t for t in collection.tables if t.section_id == section_id]
                for table in section_tables:
                    chart = self._generate_chart_for_table(table)
                    if chart:
                        collection.charts.append(chart)

        return collection

    def _get_sections(self, session) -> Dict[str, Dict[str, Any]]:
        """Get section information from session."""
        return session.section_contents

    def _extract_images_for_section(
        self,
        section_id: str,
        section_data: Dict[str, Any],
        evidence_list: List,
    ) -> List[Figure]:
        """
        Extract relevant images for a section.

        Args:
            section_id: Section identifier
            section_data: Section content and metadata
            evidence_list: Evidence items for this section

        Returns:
            List of Figure objects
        """
        figures = []

        # First, check if images are already stored in section_data
        existing_images = section_data.get("images", [])
        for idx, img_data in enumerate(existing_images[:self.max_images_per_section]):
            src = img_data.get("src", "")
            if not src:
                continue

            # Download and save image
            image_path, image_data = self._download_image(src)

            if image_path or image_data:
                figure = Figure(
                    figure_id=f"fig_{section_id}_{idx+1}",
                    figure_type=FigureType.IMAGE,
                    title=img_data.get("suggested_caption", f"Figure {idx+1}"),
                    caption=img_data.get("suggested_caption", ""),
                    source_url=src,
                    source_title=img_data.get("page_title", ""),
                    section_id=section_id,
                    image_path=image_path,
                    image_data=image_data,
                    alt_text=img_data.get("alt", ""),
                )
                figures.append(figure)

        # If no images found, try to get from evidence URLs
        if not figures and evidence_list:
            for idx, evidence in enumerate(evidence_list[:self.max_images_per_section]):
                # Try to extract image from the evidence URL
                images_from_url = self._extract_images_from_url(
                    evidence.url,
                    evidence.title,
                    limit=1,
                )
                for img in images_from_url:
                    img.section_id = section_id
                    img.figure_id = f"fig_{section_id}_{len(figures)+1}"
                    figures.append(img)
                    if len(figures) >= self.max_images_per_section:
                        break
                if len(figures) >= self.max_images_per_section:
                    break

        return figures

    def _extract_images_from_url(
        self,
        url: str,
        page_title: str,
        limit: int = 3,
    ) -> List[Figure]:
        """
        Extract images from a web page URL.

        Args:
            url: Web page URL
            page_title: Page title for attribution
            limit: Maximum images to extract

        Returns:
            List of Figure objects
        """
        figures = []

        try:
            from bs4 import BeautifulSoup

            response = requests.get(
                url,
                timeout=self.download_timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find images
            img_tags = soup.find_all('img')

            for img in img_tags:
                if len(figures) >= limit:
                    break

                src = img.get('src', '') or img.get('data-src', '')
                if not src:
                    continue

                # Make absolute URL
                src = urljoin(url, src)

                # Skip small icons and tracking pixels
                width = img.get('width')
                height = img.get('height')
                if width and int(width) < 100:
                    continue
                if height and int(height) < 100:
                    continue

                # Skip common non-content images
                if any(skip in src.lower() for skip in ['icon', 'logo', 'avatar', 'button', 'banner', 'ad']):
                    continue

                # Download image
                image_path, image_data = self._download_image(src)

                if image_path or image_data:
                    alt_text = img.get('alt', '')
                    title = img.get('title', '') or alt_text

                    figure = Figure(
                        figure_id="",  # Will be set by caller
                        figure_type=FigureType.IMAGE,
                        title=title or f"Image from {page_title}",
                        caption=f"Source: {page_title}",
                        source_url=src,
                        source_title=page_title,
                        section_id="",  # Will be set by caller
                        image_path=image_path,
                        image_data=image_data,
                        alt_text=alt_text,
                    )
                    figures.append(figure)

        except Exception as e:
            print(f"Error extracting images from {url}: {e}")

        return figures

    def _download_image(
        self,
        url: str,
    ) -> Tuple[Optional[Path], Optional[bytes]]:
        """
        Download an image from URL.

        Args:
            url: Image URL

        Returns:
            Tuple of (file path, image data)
        """
        try:
            response = requests.get(
                url,
                timeout=self.download_timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '')
            if not any(t in content_type.lower() for t in ['image', 'jpeg', 'png', 'gif', 'webp']):
                return None, None

            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = self._get_extension_from_url(url) or '.png'
            filename = f"img_{url_hash}{ext}"
            filepath = self.output_dir / filename

            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)

            return filepath, response.content

        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return None, None

    def _get_extension_from_url(self, url: str) -> Optional[str]:
        """Get file extension from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in self.IMAGE_EXTENSIONS:
            if path.endswith(ext):
                return ext
        return None

    def _extract_tables_for_section(
        self,
        section_id: str,
        section_data: Dict[str, Any],
        content: str,
        evidence_list: List,
    ) -> List[TableData]:
        """
        Extract numerical data and create tables for a section.

        Args:
            section_id: Section identifier
            section_data: Section content and metadata
            content: Section text content
            evidence_list: Evidence items for this section

        Returns:
            List of TableData objects
        """
        tables = []

        # Use LLM to extract tabular data if available
        if self.llm:
            extracted_tables = self._extract_tables_with_llm(
                section_id=section_id,
                content=content,
                section_title=section_data.get("title", ""),
            )
            tables.extend(extracted_tables)
        else:
            # Fallback: Pattern-based extraction
            extracted_tables = self._extract_tables_by_pattern(
                section_id=section_id,
                content=content,
                section_title=section_data.get("title", ""),
            )
            tables.extend(extracted_tables)

        # Add source attribution
        if tables and evidence_list:
            primary_source = evidence_list[0] if evidence_list else None
            for table in tables:
                if primary_source and not table.source_url:
                    table.source_url = primary_source.url
                    table.source_title = primary_source.title

        return tables

    def _extract_tables_with_llm(
        self,
        section_id: str,
        content: str,
        section_title: str,
    ) -> List[TableData]:
        """
        Use LLM to extract tabular data from content.

        Args:
            section_id: Section identifier
            content: Text content to analyze
            section_title: Section title for context

        Returns:
            List of TableData objects
        """
        tables = []

        prompt = f"""Analyze the following text and extract any numerical data that could be presented in a table format.
Look for:
- Time series data (yearly, monthly data)
- Comparisons between items
- Statistics and metrics
- Rankings or percentages

Section Title: {section_title}

Content:
{content[:3000]}

Return a JSON array of tables. Each table should have:
- "title": Table title
- "headers": Column headers as array
- "rows": Data rows as 2D array
- "data_type": "time_series" or "comparison" or "statistics"
- "unit": Unit of measurement if applicable

Return empty array [] if no tabular data found.
Return ONLY valid JSON, no other text."""

        try:
            response = self.llm.generate(prompt)
            response_text = response.content

            # Extract JSON from response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                extracted = json.loads(json_str)

                for idx, table_data in enumerate(extracted):
                    if not table_data.get("headers") or not table_data.get("rows"):
                        continue

                    table = TableData(
                        table_id=f"table_{section_id}_{idx+1}",
                        title=table_data.get("title", f"Table {idx+1}"),
                        caption=table_data.get("title", ""),
                        headers=table_data["headers"],
                        rows=table_data["rows"],
                        section_id=section_id,
                        data_type=table_data.get("data_type", "general"),
                        unit=table_data.get("unit"),
                    )
                    tables.append(table)

        except (json.JSONDecodeError, Exception) as e:
            print(f"Error extracting tables with LLM: {e}")

        return tables

    def _extract_tables_by_pattern(
        self,
        section_id: str,
        content: str,
        section_title: str,
    ) -> List[TableData]:
        """
        Extract tabular data using pattern matching.

        Args:
            section_id: Section identifier
            content: Text content to analyze
            section_title: Section title for context

        Returns:
            List of TableData objects
        """
        tables = []

        # Look for year-based data patterns
        years = self.YEAR_PATTERN.findall(content)
        if years and len(set(years)) >= 2:
            # Try to extract time series data
            time_series = self._extract_time_series(content, years)
            if time_series:
                table = TableData(
                    table_id=f"table_{section_id}_ts",
                    title=f"{section_title} - Time Series Data",
                    caption="Extracted time series data",
                    headers=["Year"] + list(time_series.keys()),
                    rows=self._format_time_series_rows(time_series),
                    section_id=section_id,
                    data_type="time_series",
                )
                tables.append(table)

        # Look for percentage comparisons
        percentages = self.PERCENTAGE_PATTERN.findall(content)
        if percentages and len(percentages) >= 3:
            # This could be a comparison or distribution
            pass  # Would need more context to create meaningful table

        return tables

    def _extract_time_series(
        self,
        content: str,
        years: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract time series data from content.

        Args:
            content: Text content
            years: List of years found in content

        Returns:
            Dictionary mapping metric names to year->value mappings
        """
        # Simplified extraction - would need more sophisticated parsing
        time_series = {}

        # Look for patterns like "2020年: 100億円" or "in 2020: $100M"
        for year in set(years):
            # Find values near each year mention
            pattern = rf'{year}[年\s:：]?\s*([\d,]+\.?\d*)\s*(億|万|%|円|ドル)?'
            matches = re.findall(pattern, content)
            if matches:
                for value, unit in matches:
                    metric = unit or "Value"
                    if metric not in time_series:
                        time_series[metric] = {}
                    try:
                        time_series[metric][year] = float(value.replace(',', ''))
                    except ValueError:
                        pass

        return time_series

    def _format_time_series_rows(
        self,
        time_series: Dict[str, Dict[str, Any]],
    ) -> List[List[Any]]:
        """Format time series data into table rows."""
        if not time_series:
            return []

        # Get all years
        all_years = set()
        for metric_data in time_series.values():
            all_years.update(metric_data.keys())

        years = sorted(all_years)
        rows = []

        for year in years:
            row = [year]
            for metric in time_series.keys():
                value = time_series[metric].get(year, "")
                row.append(value)
            rows.append(row)

        return rows

    def _generate_chart_for_table(
        self,
        table: TableData,
    ) -> Optional[Figure]:
        """
        Generate a matplotlib chart for a table.

        Args:
            table: TableData to visualize

        Returns:
            Figure object with chart, or None
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            # Try to use Japanese font if available
            try:
                # Common Japanese fonts
                japanese_fonts = ['IPAGothic', 'IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic']
                available_fonts = [f.name for f in fm.fontManager.ttflist]
                for font in japanese_fonts:
                    if font in available_fonts:
                        plt.rcParams['font.family'] = font
                        break
            except Exception:
                pass

            # Determine chart type based on data type
            if table.data_type == "time_series":
                chart_type = ChartType.LINE
            elif table.data_type == "comparison":
                chart_type = ChartType.BAR
            else:
                chart_type = ChartType.BAR

            # Prepare data
            if len(table.headers) < 2 or len(table.rows) < 2:
                return None

            x_labels = [row[0] for row in table.rows]
            data_columns = []
            for col_idx in range(1, len(table.headers)):
                col_data = []
                for row in table.rows:
                    try:
                        if col_idx < len(row):
                            val = row[col_idx]
                            if isinstance(val, (int, float)):
                                col_data.append(val)
                            elif isinstance(val, str):
                                # Try to parse number
                                clean_val = val.replace(',', '').replace('億', '').replace('万', '')
                                col_data.append(float(clean_val))
                            else:
                                col_data.append(0)
                        else:
                            col_data.append(0)
                    except (ValueError, TypeError):
                        col_data.append(0)
                data_columns.append((table.headers[col_idx], col_data))

            if not data_columns or not any(any(v != 0 for v in col[1]) for col in data_columns):
                return None

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == ChartType.LINE:
                for label, data in data_columns:
                    ax.plot(x_labels, data, marker='o', label=label)
                ax.legend()
            elif chart_type == ChartType.BAR:
                import numpy as np
                x = np.arange(len(x_labels))
                width = 0.8 / len(data_columns)
                for i, (label, data) in enumerate(data_columns):
                    offset = (i - len(data_columns) / 2 + 0.5) * width
                    ax.bar(x + offset, data, width, label=label)
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
                ax.legend()

            ax.set_title(table.title)
            if table.unit:
                ax.set_ylabel(table.unit)

            plt.tight_layout()

            # Save chart
            chart_filename = f"chart_{table.table_id}.png"
            chart_path = self.output_dir / chart_filename
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Read image data
            with open(chart_path, 'rb') as f:
                image_data = f.read()

            chart_figure = Figure(
                figure_id=f"chart_{table.table_id}",
                figure_type=FigureType.CHART,
                title=f"Chart: {table.title}",
                caption=table.caption,
                source_url=table.source_url,
                source_title=table.source_title,
                section_id=table.section_id,
                image_path=chart_path,
                image_data=image_data,
                alt_text=f"Chart showing {table.title}",
            )

            return chart_figure

        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return None
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

    def add_figures_to_markdown(
        self,
        markdown_content: str,
        collection: FigureTableCollection,
    ) -> str:
        """
        Add figures and tables to markdown content.

        Args:
            markdown_content: Original markdown content
            collection: Collection of figures and tables

        Returns:
            Updated markdown content
        """
        # Find section markers and insert figures/tables after each section
        lines = markdown_content.split('\n')
        result_lines = []
        current_section = None

        for line in lines:
            result_lines.append(line)

            # Detect section headers
            if line.startswith('## ') or line.startswith('### '):
                # Extract section ID from header
                match = re.match(r'^##+ (\d+(?:\.\d+)?)\. ', line)
                if match:
                    current_section = match.group(1)

            # After section content, add figures and tables
            if current_section and (line.strip() == '' or line.startswith('---')):
                # Add figures for this section
                section_figures = collection.get_figures_for_section(current_section)
                for fig in section_figures:
                    result_lines.append('')
                    if fig.image_path:
                        result_lines.append(f'![{fig.title}]({fig.image_path})')
                    result_lines.append(f'*{fig.caption}*')
                    if fig.source_url:
                        result_lines.append(f'Source: [{fig.source_title or fig.source_url}]({fig.source_url})')
                    result_lines.append('')

                # Add tables for this section
                section_tables = collection.get_tables_for_section(current_section)
                for table in section_tables:
                    result_lines.append('')
                    result_lines.append(f'**{table.title}**')
                    result_lines.append('')

                    # Create markdown table
                    header_line = '| ' + ' | '.join(str(h) for h in table.headers) + ' |'
                    separator = '|' + '|'.join(['---'] * len(table.headers)) + '|'
                    result_lines.append(header_line)
                    result_lines.append(separator)

                    for row in table.rows:
                        row_line = '| ' + ' | '.join(str(cell) for cell in row) + ' |'
                        result_lines.append(row_line)

                    if table.source_url:
                        result_lines.append(f'*Source: [{table.source_title or "Link"}]({table.source_url})*')
                    result_lines.append('')

                # Add charts for this section
                section_charts = collection.get_charts_for_section(current_section)
                for chart in section_charts:
                    result_lines.append('')
                    if chart.image_path:
                        result_lines.append(f'![{chart.title}]({chart.image_path})')
                    result_lines.append(f'*{chart.caption}*')
                    result_lines.append('')

                current_section = None  # Reset to avoid duplicate insertions

        return '\n'.join(result_lines)

    def export_collection(
        self,
        collection: FigureTableCollection,
        filepath: Path,
    ) -> None:
        """
        Export figure/table collection to JSON.

        Args:
            collection: Collection to export
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(collection.to_dict(), f, ensure_ascii=False, indent=2)


def add_figures_to_report(
    report_path: Path,
    session,
    evidence_locker,
    llm_client=None,
    output_dir: Path = None,
    language: str = "ja",
) -> Path:
    """
    Add figures and tables to an existing report.

    Args:
        report_path: Path to the report file
        session: Research session
        evidence_locker: Evidence locker
        llm_client: Optional LLM client for analysis
        output_dir: Output directory for images
        language: Language for captions

    Returns:
        Path to the updated report
    """
    # Determine output directory
    if output_dir is None:
        output_dir = report_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = FigureTableGenerator(
        llm_client=llm_client,
        output_dir=output_dir,
        language=language,
    )

    # Generate figures and tables
    collection = generator.generate_figures_and_tables(
        session=session,
        evidence_locker=evidence_locker,
    )

    # Read existing report
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add figures and tables
    if report_path.suffix == '.md':
        updated_content = generator.add_figures_to_markdown(content, collection)

        # Write updated report
        updated_path = report_path.parent / f"{report_path.stem}_with_figures.md"
        with open(updated_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        # Export collection metadata
        collection_path = output_dir / "figures_tables.json"
        generator.export_collection(collection, collection_path)

        return updated_path
    else:
        # For other formats, just return the collection path
        collection_path = output_dir / "figures_tables.json"
        generator.export_collection(collection, collection_path)
        return collection_path
