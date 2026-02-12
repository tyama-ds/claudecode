"""
Manual Evidence Loader - Load evidence from CSV/XLSX files.

This module provides functionality to load pre-collected evidence
from spreadsheet files for manual search mode.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .locker import Evidence, EvidenceLocker, EvidenceType


# Optional Excel support
try:
    import openpyxl
    XLSX_SUPPORT = True
except ImportError:
    XLSX_SUPPORT = False


class ManualEvidenceLoader:
    """
    Load evidence from CSV or XLSX files.

    Supports flexible column mapping to accommodate different file formats.

    Expected columns (flexible, case-insensitive):
    - No. / ID / no / id: Evidence number/ID
    - Title / title / タイトル: Source title
    - Contents / Content / contents / content / 内容: Main content
    - URL / url / リンク: Source URL
    - Author / author / 著者: Author name
    - Date / date / 日付: Publication date
    - Others / other / その他 / Notes / notes: Additional notes
    - Section / section / セクション: Section reference

    Example CSV:
    No.,Title,Contents,URL,Author,Date,Others
    1,Source A,Content of source A...,https://example.com/a,Author Name,2024-01-01,Additional notes
    """

    # Column name mappings (lowercase)
    COLUMN_MAPPINGS = {
        "id": ["no.", "no", "id", "番号", "number"],
        "title": ["title", "タイトル", "題名", "name"],
        "content": ["contents", "content", "内容", "本文", "text", "body"],
        "url": ["url", "link", "リンク", "source", "ソース"],
        "author": ["author", "著者", "執筆者", "writer"],
        "date": ["date", "日付", "published_date", "公開日", "year"],
        "notes": ["others", "other", "その他", "notes", "note", "備考", "memo"],
        "section": ["section", "セクション", "章", "category", "カテゴリ"],
        "publisher": ["publisher", "出版社", "source_name", "媒体"],
        "evidence_type": ["type", "evidence_type", "種類", "タイプ"],
    }

    def __init__(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize ManualEvidenceLoader.

        Args:
            file_path: Path to CSV or XLSX file
            encoding: File encoding for CSV files
            column_mapping: Optional custom column name mapping
                          e.g., {"title": "Source Name", "content": "Description"}
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.custom_mapping = column_mapping or {}

        if not self.file_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {file_path}")

        # Determine file type
        suffix = self.file_path.suffix.lower()
        if suffix == ".xlsx":
            if not XLSX_SUPPORT:
                raise ImportError(
                    "openpyxl is required for XLSX files. "
                    "Install with: pip install openpyxl"
                )
            self.file_type = "xlsx"
        elif suffix == ".csv":
            self.file_type = "csv"
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx")

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name to lowercase and strip whitespace."""
        return name.lower().strip()

    def _map_column_to_field(self, column_name: str) -> Optional[str]:
        """Map a column name to an evidence field."""
        normalized = self._normalize_column_name(column_name)

        # Check custom mapping first
        for field, custom_col in self.custom_mapping.items():
            if self._normalize_column_name(custom_col) == normalized:
                return field

        # Check standard mappings
        for field, aliases in self.COLUMN_MAPPINGS.items():
            if normalized in aliases:
                return field

        return None

    def _read_csv(self) -> List[Dict[str, Any]]:
        """Read data from CSV file."""
        rows = []

        # Try different encodings if UTF-8 fails
        encodings_to_try = [self.encoding, "utf-8-sig", "cp932", "shift_jis", "latin1"]

        for enc in encodings_to_try:
            try:
                with open(self.file_path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise RuntimeError(f"Failed to read CSV file: {e}")

        if not rows:
            raise RuntimeError(
                f"Failed to read CSV file with any encoding: {encodings_to_try}"
            )

        return rows

    def _read_xlsx(self) -> List[Dict[str, Any]]:
        """Read data from XLSX file."""
        wb = openpyxl.load_workbook(self.file_path, read_only=True, data_only=True)
        ws = wb.active

        rows = []
        headers = []

        for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
            if row_idx == 0:
                # First row is headers
                headers = [str(cell) if cell else f"col_{i}" for i, cell in enumerate(row)]
            else:
                # Skip empty rows
                if all(cell is None or str(cell).strip() == "" for cell in row):
                    continue

                row_dict = {}
                for i, cell in enumerate(row):
                    if i < len(headers):
                        row_dict[headers[i]] = str(cell) if cell is not None else ""
                rows.append(row_dict)

        wb.close()
        return rows

    def _row_to_evidence(
        self,
        row: Dict[str, Any],
        row_number: int,
        column_map: Dict[str, str],
    ) -> Evidence:
        """Convert a row to an Evidence object."""

        def get_value(field: str) -> str:
            """Get value for a field from the row."""
            if field in column_map:
                col_name = column_map[field]
                return str(row.get(col_name, "")).strip()
            return ""

        # Get basic fields
        title = get_value("title") or f"Evidence #{row_number}"
        content = get_value("content")
        url = get_value("url")
        author = get_value("author")
        date_str = get_value("date")
        notes = get_value("notes")
        section = get_value("section")
        publisher = get_value("publisher")
        evidence_type_str = get_value("evidence_type")

        # Determine evidence type
        evidence_type = EvidenceType.USER_PROVIDED
        if evidence_type_str:
            type_lower = evidence_type_str.lower()
            if "pdf" in type_lower:
                evidence_type = EvidenceType.PDF_DOCUMENT
            elif "paper" in type_lower or "research" in type_lower:
                evidence_type = EvidenceType.RESEARCH_PAPER
            elif "news" in type_lower:
                evidence_type = EvidenceType.NEWS_ARTICLE
            elif "official" in type_lower or "gov" in type_lower:
                evidence_type = EvidenceType.OFFICIAL_DOCUMENT
            elif "web" in type_lower:
                evidence_type = EvidenceType.WEB_PAGE

        # Create evidence
        evidence = Evidence(
            evidence_type=evidence_type,
            url=url,
            title=title,
            author=author,
            publisher=publisher,
            published_date=date_str,
            content_excerpt=content,
            section_reference=section,
            metadata={
                "row_number": row_number,
                "notes": notes,
                "source_file": str(self.file_path),
            },
        )

        return evidence

    def load(self) -> List[Evidence]:
        """
        Load evidence from the file.

        Returns:
            List of Evidence objects
        """
        # Read raw data
        if self.file_type == "csv":
            rows = self._read_csv()
        else:
            rows = self._read_xlsx()

        if not rows:
            return []

        # Build column mapping from first row
        first_row = rows[0]
        column_map = {}  # field -> column_name

        for col_name in first_row.keys():
            field = self._map_column_to_field(col_name)
            if field:
                column_map[field] = col_name

        # Convert rows to evidence
        evidence_list = []
        for i, row in enumerate(rows, 1):
            # Skip rows with no meaningful content
            content = ""
            title = ""
            for field in ["content", "title"]:
                if field in column_map:
                    val = str(row.get(column_map[field], "")).strip()
                    if field == "content":
                        content = val
                    elif field == "title":
                        title = val

            if not content and not title:
                continue

            evidence = self._row_to_evidence(row, i, column_map)
            evidence_list.append(evidence)

        return evidence_list

    def load_to_locker(
        self,
        research_id: str = None,
        output_dir: Path = None,
    ) -> EvidenceLocker:
        """
        Load evidence directly into an EvidenceLocker.

        Args:
            research_id: Research session ID
            output_dir: Output directory for the locker

        Returns:
            EvidenceLocker populated with loaded evidence
        """
        locker = EvidenceLocker(
            research_id=research_id,
            output_dir=output_dir,
        )

        evidence_list = self.load()

        for evidence in evidence_list:
            # Add to locker (will handle deduplication)
            locker.add_evidence(
                url=evidence.url,
                title=evidence.title,
                content_excerpt=evidence.content_excerpt,
                evidence_type=evidence.evidence_type,
                section_reference=evidence.section_reference,
                author=evidence.author,
                publisher=evidence.publisher,
                published_date=evidence.published_date,
                metadata=evidence.metadata,
            )

        return locker

    def get_column_preview(self, max_rows: int = 5) -> Dict[str, Any]:
        """
        Get a preview of the file structure for validation.

        Args:
            max_rows: Maximum number of rows to preview

        Returns:
            Dictionary with column info and sample data
        """
        # Read raw data
        if self.file_type == "csv":
            rows = self._read_csv()
        else:
            rows = self._read_xlsx()

        if not rows:
            return {"columns": [], "mapped_fields": {}, "sample_rows": []}

        # Get columns
        columns = list(rows[0].keys())

        # Map columns to fields
        mapped_fields = {}
        for col in columns:
            field = self._map_column_to_field(col)
            if field:
                mapped_fields[col] = field

        # Get sample rows
        sample_rows = rows[:max_rows]

        return {
            "columns": columns,
            "mapped_fields": mapped_fields,
            "unmapped_columns": [c for c in columns if c not in mapped_fields],
            "total_rows": len(rows),
            "sample_rows": sample_rows,
        }


def load_evidence_file(
    file_path: str | Path,
    research_id: str = None,
    output_dir: Path = None,
    column_mapping: Optional[Dict[str, str]] = None,
    encoding: str = "utf-8",
) -> EvidenceLocker:
    """
    Convenience function to load evidence from a file into an EvidenceLocker.

    Args:
        file_path: Path to CSV or XLSX file
        research_id: Research session ID
        output_dir: Output directory for evidence exports
        column_mapping: Custom column name mapping
        encoding: File encoding for CSV files

    Returns:
        EvidenceLocker populated with evidence from the file

    Example:
        # Load from CSV
        locker = load_evidence_file("research_data.csv")

        # Load from XLSX with custom mapping
        locker = load_evidence_file(
            "data.xlsx",
            column_mapping={"title": "Source Name", "content": "Description"}
        )
    """
    loader = ManualEvidenceLoader(
        file_path=file_path,
        encoding=encoding,
        column_mapping=column_mapping,
    )

    return loader.load_to_locker(
        research_id=research_id,
        output_dir=output_dir,
    )
