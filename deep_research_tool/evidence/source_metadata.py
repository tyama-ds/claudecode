"""Conservative, shared provenance extraction for every ingestion path.

Publication, modification, effective and observation dates are different facts.
Only labelled dates or explicit metadata populate them; an arbitrary year in
the body is never treated as a publication date.
"""

import json
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse


def host_matches(host: str, domain: str) -> bool:
    host = host.lower().rstrip(".")
    domain = domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def source_type_for_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return "unknown"
    if not host:
        return "unknown"
    if re.search(r"\.(?:gov|gov\.[a-z]{2}|go\.[a-z]{2}|govt\.[a-z]{2}|gouv\.fr)$", host) or any(
        host_matches(host, d) for d in ("who.int", "un.org", "worldbank.org", "europa.eu")
    ):
        return "official"
    if re.search(r"\.(?:edu|edu\.[a-z]{2}|ac\.[a-z]{2})$", host) or any(
        host_matches(host, d) for d in ("arxiv.org", "pubmed.ncbi.nlm.nih.gov")
    ):
        return "academic"
    groups = {
        "news": ("reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "cnn.com",
                 "nytimes.com", "washingtonpost.com", "theguardian.com", "nhk.or.jp",
                 "asahi.com", "nikkei.com", "ft.com", "wsj.com", "economist.com"),
        "wiki": ("wikipedia.org", "wikimedia.org"),
        "social": ("twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com", "reddit.com"),
        "forum": ("quora.com", "stackoverflow.com"),
        "blog": ("medium.com", "substack.com", "wordpress.com", "blogspot.com"),
    }
    for kind, domains in groups.items():
        if any(host_matches(host, d) for d in domains):
            return kind
    if re.search(r"\.(?:com|co\.[a-z]{2})$", host):
        return "commercial"
    return "unknown"


def normalize_source_date(value, *, allow_future=False) -> str:
    """Validate a date, retaining year/month precision instead of inventing it."""
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    match = re.fullmatch(r"(\d{4})(?:[-/年](\d{1,2})(?:[-/月](\d{1,2})日?)?)?(?:[T ].*)?", raw)
    if match:
        year, month, day = match.groups()
        try:
            parsed = date(int(year), int(month or 1), int(day or 1))
        except ValueError:
            return ""
        if not allow_future and parsed > date.today():
            return ""
        return year + (f"-{int(month):02}" if month else "") + (f"-{int(day):02}" if day else "")
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            parsed = datetime.strptime(raw, fmt).date()
            return parsed.isoformat() if allow_future or parsed <= date.today() else ""
        except ValueError:
            pass
    try:
        parsed = parsedate_to_datetime(raw).date()
        return parsed.isoformat() if allow_future or parsed <= date.today() else ""
    except (ValueError, TypeError, OverflowError):
        return ""


_DATE_TEXT = r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|(?:Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)\s+\d{1,2},?\s+\d{4})"


def extract_source_metadata(url, text="", metadata=None, html="") -> dict:
    """Return normalized source fields and the origin of every accepted date."""
    supplied = dict(metadata) if isinstance(metadata, dict) else {}
    candidates = {str(k).lower(): (v, "metadata:" + str(k)) for k, v in supplied.items()}
    if isinstance(html, str) and html:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("meta"):
            key = tag.get("property") or tag.get("name") or tag.get("itemprop")
            if key and tag.get("content"):
                candidates.setdefault(key.lower(), (tag["content"], "html:" + key))
        # Only the document-level objects/graphs, never unrelated dates from
        # comments, related articles or nested citation objects.
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or script.get_text())
            except (ValueError, TypeError):
                continue
            objects = data if isinstance(data, list) else [data]
            for obj in list(objects):
                if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
                    objects.extend(obj["@graph"])
            document_types = {"Article", "NewsArticle", "ScholarlyArticle", "Report", "WebPage", "BlogPosting", "TechArticle", "Dataset"}
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                kinds = obj.get("@type", [])
                if isinstance(kinds, str):
                    kinds = [kinds]
                if not isinstance(kinds, list) or not document_types.intersection(kinds):
                    continue
                object_url = obj.get("url")
                if isinstance(object_url, str) and object_url.rstrip("/").split("#")[0] != str(url).rstrip("/").split("#")[0]:
                    continue
                for key in ("datePublished", "dateModified", "temporalCoverage", "publisher"):
                    if key in obj:
                        value = obj[key]
                        if key == "publisher" and isinstance(value, dict):
                            value = value.get("name", "")
                        candidates.setdefault(key.lower(), (value, "jsonld:" + key))
    result = {"published_date": "", "updated_at": "", "data_period": "", "effective_at": "",
              "publisher": "", "source_type": source_type_for_url(url), "date_provenance": {}}
    aliases = {
        "published_date": ("published_date", "date_published", "published_at", "datepublished", "article:published_time", "publication_date", "citation_publication_date", "dc.date.issued"),
        "updated_at": ("updated_at", "datemodified", "article:modified_time", "last-modified", "dc.date.modified"),
        "effective_at": ("effective_at", "effective_date"),
    }
    for field, keys in aliases.items():
        for key in keys:
            if key in candidates:
                value, origin = candidates[key]
                normalized = normalize_source_date(value, allow_future=field == "effective_at")
                if normalized:
                    result[field] = normalized
                    result["date_provenance"][field] = {"source": origin, "value": str(value)}
                    break
    labels = {
        "published_date": r"(?:公表日|公開日|発行日|発表日|掲載日|published(?:\s+on)?|publication\s+date)",
        "updated_at": r"(?:最終更新日|更新日|last\s+updated|updated(?:\s+on)?)",
        "effective_at": r"(?:施行日|適用開始日|effective(?:\s+date|\s+from)?)",
    }
    body = text if isinstance(text, str) else ""
    for field, label in labels.items():
        if result[field]:
            continue
        # Publication suffix supports Japanese notices such as 2026年9月14日公表.
        patterns = [label + r"\s*[:：]?\s*(" + _DATE_TEXT + r")"]
        if field == "published_date":
            patterns.append(r"(" + _DATE_TEXT + r")\s*(?:公表|公開|発行|発表)(?!予定)")
        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                value = normalize_source_date(match.group(1), allow_future=field == "effective_at")
                if value:
                    result[field] = value
                    result["date_provenance"][field] = {"source": "body:label", "value": match.group(0), "start": match.start(), "end": match.end()}
                    break
    for key in ("publisher", "og:site_name"):
        value = candidates.get(key, ("", ""))[0]
        if isinstance(value, str) and value.strip():
            result["publisher"] = value.strip()
            break
    period, origin = candidates.get("data_period", candidates.get("temporalcoverage", ("", "")))
    if isinstance(period, str) and period.strip():
        result["data_period"] = period.strip()
        result["date_provenance"]["data_period"] = {"source": origin, "value": period}
    result["is_primary_source"] = result["source_type"] == "official"
    previous_provenance = supplied.get("date_provenance")
    if isinstance(previous_provenance, dict):
        for field in ("published_date", "updated_at", "effective_at", "data_period"):
            if result[field] and isinstance(previous_provenance.get(field), dict):
                result["date_provenance"][field] = dict(previous_provenance[field])
    return result
