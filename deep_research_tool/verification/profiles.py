"""
Verification profiles - one backend resolver for GUI / Web UI / CLI.

Four profiles control how the finalization verification runs:

- fast:     no research / no revision rounds; critical+important claims
            are always verified, minor claims are sampled; cache,
            parallel workers and batch judging on. Unresolved claims may
            remain and are surfaced in the result summary.
- balanced: 1 research + 1 revision round, ALL claims verified, cache /
            parallel / batching on. The recommended default.
- strict:   the legacy-equivalent budget (2/2/1) with ALL claims,
            required_critical_coverage=1.0 and
            min_claim_support_score=0.85. Cache and parallelism are
            allowed but NO acceptance threshold is relaxed.
- custom:   every knob user-settable, with hard bounds validated here.

GUI and CLI never duplicate this logic: they submit the profile name
plus (for custom) overrides, and ``resolve_verification_settings`` is
the single source of truth.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

VERIFICATION_PROFILES = ("fast", "balanced", "strict", "custom")

# hard bounds for custom values (also applied to config-level overrides)
_BOUNDS = {
    "max_final_research_rounds": (0, 5),
    "max_final_revision_rounds": (0, 5),
    "max_no_improvement_rounds": (0, 5),
    "min_claim_support_score": (0.0, 1.0),
    "required_critical_coverage": (0.0, 1.0),
    "max_workers": (1, 16),
    "batch_size": (1, 32),
    "minor_claim_sample_rate": (0.0, 1.0),
    "timeout_seconds": (0, 86400),
}


@dataclass
class VerificationSettings:
    """Resolved, validated verification behavior for one run."""
    profile: str = "balanced"
    max_final_research_rounds: int = 1
    max_final_revision_rounds: int = 1
    max_no_improvement_rounds: int = 1
    min_claim_support_score: float = 0.85
    required_critical_coverage: float = 1.0
    max_workers: int = 4
    batch_size: int = 10
    cache_enabled: bool = True
    # fraction of MINOR claims verified (critical/important are ALWAYS
    # verified in every profile — fast mode never skips them)
    minor_claim_sample_rate: float = 1.0
    timeout_seconds: int = 0            # 0 = no timeout

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": dict(
        max_final_research_rounds=0,
        max_final_revision_rounds=0,
        max_no_improvement_rounds=1,
        minor_claim_sample_rate=0.34,
        cache_enabled=True,
        max_workers=8,
        batch_size=12,
    ),
    "balanced": dict(
        max_final_research_rounds=1,
        max_final_revision_rounds=1,
        max_no_improvement_rounds=1,
        minor_claim_sample_rate=1.0,
        cache_enabled=True,
        max_workers=8,
        batch_size=10,
    ),
    "strict": dict(
        max_final_research_rounds=2,
        max_final_revision_rounds=2,
        max_no_improvement_rounds=1,
        min_claim_support_score=0.85,
        required_critical_coverage=1.0,
        minor_claim_sample_rate=1.0,
        cache_enabled=True,
        max_workers=8,
        batch_size=10,
    ),
    # custom starts from balanced and applies the user's overrides
    "custom": dict(
        max_final_research_rounds=1,
        max_final_revision_rounds=1,
        max_no_improvement_rounds=1,
        minor_claim_sample_rate=1.0,
        cache_enabled=True,
        max_workers=8,
        batch_size=10,
    ),
}

# human-facing descriptions (shared by GUI/CLI help)
PROFILE_DESCRIPTIONS = {
    "fast": "最短時間で検証します。追加調査や自動修正を行わないため、"
            "未解決のクレームが残る場合があります。",
    "balanced": "速度と検証品質のバランスを取った推奨設定です。",
    "strict": "すべてのクレームを厳格に検証します。レポートの内容によっては"
              "完了まで長時間かかる場合があります。",
    "custom": "検証条件を個別に設定できます。設定内容によっては非常に"
              "長い時間がかかる可能性があります。",
}

CUSTOM_WARNING = ("注意: 設定内容やレポートの長さによっては、Verificationに"
                  "非常に長い時間がかかる可能性があります。特に追加調査回数、"
                  "修正回数、検証対象クレーム数、並列数を変更する場合は"
                  "ご注意ください。")


def _check_bounds(name: str, value, integer: bool = False):
    lo, hi = _BOUNDS[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    if integer:
        if isinstance(value, float):
            raise ValueError(f"{name} must be an integer (got {value!r})")
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer (got {value!r})")
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number (got {value!r})")
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi} "
                         f"(got {value})")
    return value


def resolve_verification_settings(
    profile: str = "balanced",
    overrides: Optional[Dict[str, Any]] = None,
) -> VerificationSettings:
    """Resolve a profile (+ overrides) into validated settings.

    - Unknown profiles raise ValueError.
    - Overrides are validated against hard bounds; invalid values raise
      (execution must not start with invalid values).
    - For non-custom profiles, overrides still apply (they represent
      values the user explicitly changed from the defaults — required
      for backward compatibility with existing configs/tests), except
      that STRICT never relaxes its acceptance thresholds:
      min_claim_support_score / required_critical_coverage can only be
      made stricter, and all claims are always verified.
    """
    profile = (profile or "balanced").strip().lower()
    if profile not in VERIFICATION_PROFILES:
        raise ValueError(
            f"verification_profile must be one of {VERIFICATION_PROFILES} "
            f"(got {profile!r})")

    settings = VerificationSettings(profile=profile)
    for key, value in _PRESETS[profile].items():
        setattr(settings, key, value)

    integer_fields = {"max_final_research_rounds",
                      "max_final_revision_rounds",
                      "max_no_improvement_rounds",
                      "max_workers", "batch_size", "timeout_seconds"}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "cache_enabled":
            if not isinstance(value, bool):
                raise ValueError("cache_enabled must be a bool")
            settings.cache_enabled = value
            continue
        if key not in _BOUNDS:
            raise ValueError(f"unknown verification setting: {key}")
        checked = _check_bounds(key, value, integer=key in integer_fields)
        setattr(settings, key, checked)

    if profile == "strict":
        # strict NEVER weakens acceptance criteria or skips claims
        settings.min_claim_support_score = max(
            settings.min_claim_support_score, 0.85)
        settings.required_critical_coverage = max(
            settings.required_critical_coverage, 1.0)
        settings.minor_claim_sample_rate = 1.0

    return settings


def settings_from_research_config(rc) -> VerificationSettings:
    """Resolve settings from a ResearchConfig (the ONE backend path).

    UNSET semantics: every override field on the config defaults to
    ``None`` (= the user did not set it, the profile preset applies).
    A non-None value is an EXPLICIT override. Overrides are never
    inferred by diffing a value against a declared default — a user who
    explicitly picks the default value has still made an explicit
    choice, and a preset that happens to share the default is never
    mistaken for a user override.
    """
    overrides: Dict[str, Any] = {}

    field_map = {
        "max_final_research_rounds": "max_final_research_rounds",
        "max_final_revision_rounds": "max_final_revision_rounds",
        "max_no_improvement_rounds": "max_no_improvement_rounds",
        "min_claim_support_score": "min_claim_support_score",
        "required_critical_coverage": "required_critical_coverage",
        "verification_max_workers": "max_workers",
        "verification_batch_size": "batch_size",
        "verification_minor_claim_sample_rate": "minor_claim_sample_rate",
        "verification_timeout_seconds": "timeout_seconds",
    }
    for cfg_name, key in field_map.items():
        value = getattr(rc, cfg_name, None)
        if value is not None:
            overrides[key] = value
    cache = getattr(rc, "verification_cache_enabled", None)
    if cache is not None:
        overrides["cache_enabled"] = cache

    profile = getattr(rc, "verification_profile", "balanced")
    return resolve_verification_settings(profile, overrides)
