"""
Glossary Manager - Extract and manage terminology for consistent usage.

Version 2.0 feature for ensuring terminology consistency across the report.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class TermCandidate:
    """A candidate term extracted from content."""
    term: str
    frequency: int = 1
    contexts: List[str] = field(default_factory=list)
    is_acronym: bool = False
    expanded_form: str = ""


class GlossaryManager:
    """
    Extract, manage, and apply terminology for consistent report generation.

    Features:
    - Automatic term extraction from research plan and content
    - Acronym detection and expansion
    - Preferred form management
    - Term standardization in generated content
    """

    def __init__(self, llm_client=None, language: str = "ja"):
        """
        Initialize GlossaryManager.

        Args:
            llm_client: LLM client for intelligent term extraction
            language: Target language
        """
        self.llm = llm_client
        self.language = language
        self.terms: Dict[str, TermCandidate] = {}

        # Common acronym patterns
        self.acronym_patterns = {
            "ja": [
                # Pattern to match "ABC（何々）" or "何々（ABC）"
                r'([A-Z]{2,6})（([^）]+)）',
                r'([^（]+)（([A-Z]{2,6})）',
            ],
            "en": [
                r'([A-Z]{2,6})\s*\(([^)]+)\)',
                r'([^(]+)\s*\(([A-Z]{2,6})\)',
            ],
        }

    def extract_terms_from_plan(
        self,
        research_plan: Any,
        research_topic: str,
    ) -> Dict[str, Any]:
        """
        Extract key terms from the research plan.

        Args:
            research_plan: ResearchPlan object
            research_topic: Original research topic

        Returns:
            Dict of term -> definition/info
        """
        terms = {}

        # Extract from key_terms in the plan
        if hasattr(research_plan, 'key_terms'):
            for term in research_plan.key_terms:
                terms[term.lower()] = {
                    "term": term,
                    "definition": "",
                    "source": "research_plan",
                }

        # Extract from ToC titles
        if hasattr(research_plan, 'table_of_contents'):
            toc = research_plan.table_of_contents
            for item in toc.items if hasattr(toc, 'items') else []:
                # Extract potential terms from section titles
                extracted = self._extract_terms_from_text(item.title)
                for term in extracted:
                    if term.lower() not in terms:
                        terms[term.lower()] = {
                            "term": term,
                            "definition": "",
                            "source": f"toc_{item.section}",
                        }

        # Use LLM to identify and define key terms
        if self.llm:
            llm_terms = self._extract_terms_with_llm(research_topic, list(terms.keys()))
            terms.update(llm_terms)

        return terms

    def extract_terms_from_content(
        self,
        content: str,
        section: str = "",
    ) -> List[TermCandidate]:
        """
        Extract potential terms from chapter content.

        Args:
            content: Chapter content
            section: Section identifier

        Returns:
            List of TermCandidate objects
        """
        candidates = []

        # Extract acronyms with their expansions
        patterns = self.acronym_patterns.get(self.language, [])
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Determine which is the acronym
                if re.match(r'^[A-Z]{2,6}$', match[0]):
                    acronym, expansion = match[0], match[1]
                else:
                    expansion, acronym = match[0], match[1]

                candidate = TermCandidate(
                    term=acronym,
                    frequency=content.count(acronym),
                    contexts=[content[max(0, content.find(acronym) - 50):content.find(acronym) + 50]],
                    is_acronym=True,
                    expanded_form=expansion,
                )
                candidates.append(candidate)

        # Extract capitalized terms (likely proper nouns or technical terms)
        if self.language == "en":
            cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            cap_matches = re.findall(cap_pattern, content)
            for term in set(cap_matches):
                if len(term) > 3 and term not in ["The", "This", "That", "These", "Those"]:
                    candidate = TermCandidate(
                        term=term,
                        frequency=content.count(term),
                        contexts=[],
                    )
                    candidates.append(candidate)

        # Extract katakana terms (Japanese)
        if self.language == "ja":
            kata_pattern = r'[ァ-ヶー]{3,}'
            kata_matches = re.findall(kata_pattern, content)
            for term in set(kata_matches):
                candidate = TermCandidate(
                    term=term,
                    frequency=content.count(term),
                    contexts=[],
                )
                candidates.append(candidate)

        return candidates

    def _extract_terms_from_text(self, text: str) -> List[str]:
        """Extract potential terms from a text string."""
        terms = []

        # Extract English terms/acronyms
        english_terms = re.findall(r'\b[A-Z][A-Za-z0-9]*\b', text)
        terms.extend([t for t in english_terms if len(t) > 1])

        # Extract katakana terms (Japanese)
        if self.language == "ja":
            kata_terms = re.findall(r'[ァ-ヶー]{2,}', text)
            terms.extend(kata_terms)

        return terms

    def _extract_terms_with_llm(
        self,
        research_topic: str,
        existing_terms: List[str],
    ) -> Dict[str, Any]:
        """Use LLM to identify and define key terms."""
        if not self.llm:
            return {}

        existing_str = ", ".join(existing_terms[:20]) if existing_terms else "なし"

        if self.language == "ja":
            prompt = f"""調査テーマ「{research_topic}」に関連する重要な専門用語を抽出し、定義してください。

既に特定された用語: {existing_str}

以下の形式でJSON配列として出力してください：
[
  {{"term": "用語", "definition": "簡潔な定義（1-2文）", "aliases": ["別名1", "別名2"], "preferred_form": "優先表記"}}
]

10-15個の重要な用語を抽出してください。
JSONのみを出力:"""
        else:
            prompt = f"""Extract and define key technical terms related to the research topic "{research_topic}".

Already identified terms: {existing_str}

Output as JSON array:
[
  {{"term": "term", "definition": "brief definition (1-2 sentences)", "aliases": ["alias1", "alias2"], "preferred_form": "preferred form"}}
]

Extract 10-15 important terms.
Output only JSON:"""

        try:
            response = self.llm.generate(prompt)
            import json
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]

            terms_data = json.loads(content)
            result = {}
            for t in terms_data:
                key = t.get("term", "").lower()
                if key:
                    result[key] = {
                        "term": t.get("term", ""),
                        "definition": t.get("definition", ""),
                        "aliases": t.get("aliases", []),
                        "preferred_form": t.get("preferred_form", t.get("term", "")),
                        "source": "llm",
                    }
            return result

        except Exception as e:
            print(f"[GlossaryManager] LLM term extraction failed: {e}")
            return {}

    def standardize_terms(
        self,
        content: str,
        glossary: Dict[str, Any],
    ) -> str:
        """
        Standardize terminology in content based on glossary.

        Args:
            content: Original content
            glossary: Glossary dict with preferred forms

        Returns:
            Content with standardized terminology
        """
        result = content

        for key, entry in glossary.items():
            preferred = entry.get("preferred_form", entry.get("term", ""))
            aliases = entry.get("aliases", [])

            for alias in aliases:
                if alias.lower() != preferred.lower():
                    # Replace alias with preferred form (case-insensitive)
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    result = re.sub(pattern, preferred, result, flags=re.IGNORECASE)

        return result

    def generate_glossary_section(
        self,
        glossary: Dict[str, Any],
        title: str = "用語集",
    ) -> str:
        """
        Generate a glossary section for the report.

        Args:
            glossary: Glossary dict
            title: Section title

        Returns:
            Formatted glossary section
        """
        if not glossary:
            return ""

        lines = [f"## {title}", ""]

        # Sort by term
        sorted_terms = sorted(glossary.items(), key=lambda x: x[1].get("term", x[0]))

        for key, entry in sorted_terms:
            term = entry.get("term", key)
            definition = entry.get("definition", "")
            aliases = entry.get("aliases", [])

            if aliases:
                alias_str = f"（別名: {', '.join(aliases)}）"
            else:
                alias_str = ""

            lines.append(f"**{term}**{alias_str}")
            if definition:
                lines.append(f": {definition}")
            lines.append("")

        return "\n".join(lines)

    def get_first_mention_format(
        self,
        term: str,
        glossary: Dict[str, Any],
    ) -> str:
        """
        Get the format for first mention of a term.

        For acronyms, returns "Expanded Form (ACRONYM)" format.

        Args:
            term: The term
            glossary: Glossary dict

        Returns:
            Formatted first mention string
        """
        key = term.lower()
        if key not in glossary:
            return term

        entry = glossary[key]
        aliases = entry.get("aliases", [])
        preferred = entry.get("preferred_form", term)

        # Check if it's an acronym
        if re.match(r'^[A-Z]{2,6}$', preferred):
            # Find the expanded form in aliases
            for alias in aliases:
                if not re.match(r'^[A-Z]{2,6}$', alias):
                    if self.language == "ja":
                        return f"{alias}（{preferred}）"
                    else:
                        return f"{alias} ({preferred})"

        return preferred

    def create_initial_glossary(
        self,
        research_topic: str,
        research_plan: Any = None,
    ) -> Dict[str, Any]:
        """
        Create initial glossary from research topic and plan.

        Args:
            research_topic: Original research topic
            research_plan: Optional ResearchPlan object

        Returns:
            Initial glossary dict
        """
        glossary = {}

        # Extract from research plan if available
        if research_plan:
            plan_terms = self.extract_terms_from_plan(research_plan, research_topic)
            glossary.update(plan_terms)

        # Use LLM to enhance
        if self.llm:
            llm_terms = self._extract_terms_with_llm(research_topic, list(glossary.keys()))
            # Merge without overwriting existing definitions
            for key, value in llm_terms.items():
                if key not in glossary:
                    glossary[key] = value
                elif not glossary[key].get("definition") and value.get("definition"):
                    glossary[key]["definition"] = value["definition"]

        return glossary
