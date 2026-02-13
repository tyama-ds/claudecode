"""
Core prompt optimization engine.

Takes a user prompt and selected PE techniques, then uses an LLM
to produce an optimized prompt incorporating those techniques.
"""

from .techniques import Technique, get_techniques_by_ids


def build_optimization_system_prompt(techniques: list[Technique], language: str = "en") -> str:
    """Build the system prompt for the optimizer LLM call."""

    technique_blocks = []
    for i, t in enumerate(techniques, 1):
        technique_blocks.append(
            f"### Technique {i}: {t.name}\n"
            f"**Description**: {t.description}\n"
            f"**How to apply**: {t.instruction}\n"
            f"**Pattern**: {t.example_pattern}"
        )
    techniques_section = "\n\n".join(technique_blocks)

    lang_instruction = ""
    if language == "ja":
        lang_instruction = (
            "\n\nIMPORTANT: The optimized prompt MUST be written in Japanese. "
            "The structural elements (section headings, instructions) should be in Japanese. "
            "Technical terms may remain in English where appropriate."
        )
    elif language == "en":
        lang_instruction = "\n\nThe optimized prompt should be written in English."

    return f"""You are an expert Prompt Engineer specializing in optimizing prompts for Large Language Models.

Your task is to take a user's original prompt and transform it into a highly optimized version
that incorporates specific prompt engineering techniques selected by the user.

## Your Optimization Process

1. **Analyze** the original prompt to understand its intent, target audience, and desired outcome.
2. **Apply** each selected technique thoughtfully — do not just mechanically append keywords.
3. **Integrate** the techniques naturally so the final prompt reads as a cohesive whole.
4. **Preserve** the original intent — the optimized prompt must accomplish the same goal.
5. **Enhance** clarity, specificity, and effectiveness while keeping the prompt practical.

## Selected Techniques to Apply

{techniques_section}

## Output Requirements

- Return ONLY the optimized prompt text. Do not include explanations, metadata, or commentary.
- The optimized prompt should be immediately usable — ready to paste into an LLM.
- If multiple techniques overlap, merge them elegantly rather than creating redundant sections.
- The optimized prompt should be significantly better than the original while remaining natural.
{lang_instruction}"""


def build_optimization_user_prompt(original_prompt: str) -> str:
    """Build the user message for the optimizer call."""
    return (
        f"## Original Prompt to Optimize\n\n"
        f"{original_prompt}\n\n"
        f"---\n\n"
        f"Please produce the optimized version of this prompt incorporating all the "
        f"selected techniques. Return ONLY the optimized prompt text."
    )


def build_analysis_system_prompt() -> str:
    """Build system prompt for the analysis/explanation step."""
    return """You are an expert Prompt Engineer. The user has just received an optimized prompt.
Your job is to provide a brief analysis explaining:

1. **Techniques Applied**: For each technique, explain specifically how it was integrated.
2. **Key Improvements**: What are the main improvements over the original?
3. **Usage Tips**: Any tips for using the optimized prompt effectively.

Keep the analysis concise and actionable. Use markdown formatting.
Respond in the same language as the optimized prompt."""


def build_analysis_user_prompt(original: str, optimized: str, technique_names: list[str]) -> str:
    """Build user prompt for analysis."""
    techniques_str = ", ".join(technique_names)
    return (
        f"## Original Prompt\n{original}\n\n"
        f"## Optimized Prompt\n{optimized}\n\n"
        f"## Applied Techniques\n{techniques_str}\n\n"
        f"Please provide a brief analysis of the optimization."
    )
