"""
GUI configuration builder — PURE, tkinter-free.

Separated from gui.py so the Tkinter GUI -> create_config wiring is
testable headlessly (CI containers have no display and often no
tkinter). The GUI collects plain values from its Tk variables and calls
``build_gui_config``.
"""


def build_gui_config(v: dict) -> dict:
    """Build the run_research kwargs from plain GUI values.

    Pure function (no Tk dependency) so the GUI->create_config wiring is
    testable headlessly. ``v`` holds plain Python values of the Tk
    variables. Raises ValueError on invalid parallel_max_workers.
    """
    from .utils.concurrency import validate_parallel_max_workers

    config = {
        "topic": (v.get("topic") or "").strip(),
        "provider": v.get("provider", "openai"),
        "search_method": v.get("search_method", "duckduckgo"),
        # run_research()'s contract is ``iterations`` (it forwards it to
        # create_config as research_iterations). Emitting research_iterations
        # here made run_research receive BOTH via **kwargs -> TypeError.
        "iterations": v.get("min_iterations", 3),
        "max_iterations": v.get("max_iterations", 10),
        "output_format": v.get("output_format", "markdown"),
        "output_dir": v.get("output_dir", "./output"),
        "enable_verification": v.get("enable_verification", True),
        "verification_profile": v.get("verification_profile", "balanced"),
        "verbose": v.get("verbose", False),
        "language": v.get("language", "ja"),
        "max_results": v.get("max_results", 10),

        # Extended Mode
        "extended_mode": v.get("extended_mode", False),
        "crawl_max_pages": v.get("crawl_max_pages", 10),
        "crawl_max_depth": v.get("crawl_max_depth", 2),
        "crawl_max_sites": v.get("crawl_max_sites", 3),

        # DeepThink
        "deep_think": v.get("deep_think", False),
        "deep_think_level": v.get("deep_think_level", 0.5),
        "reasoning_iterations": v.get("reasoning_iterations", 3),
        "consistency_threshold": v.get("consistency_threshold", 0.3),
        "consistency_mode": v.get("consistency_mode", "warn"),
        "fidelity_threshold": v.get("fidelity_threshold", 0.7),
    }

    # App-wide parallelism (strictly validated; 1..16, no clamping)
    config["parallel_max_workers"] = validate_parallel_max_workers(
        v.get("parallel_max_workers", 8), source="Parallel workers")

    # LLM sampling knobs (were collected by the GUI but never forwarded)
    if v.get("temperature") is not None:
        temperature = float(v["temperature"])
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        config["temperature"] = temperature
    if v.get("max_tokens") not in (None, ""):
        max_tokens = int(v["max_tokens"])
        if max_tokens < 1:
            raise ValueError("Max Tokens must be a positive integer")
        config["max_tokens"] = max_tokens
    # search region (DuckDuckGo kl code); "wt-wt" = worldwide default
    region = (v.get("region") or "").strip()
    if region and region != "wt-wt":
        config["search_region"] = region
    # report rendering toggles
    for key in ("include_images", "include_citations", "include_toc"):
        if key in v:
            config[key] = bool(v[key])
    # NOTE: verification_strictness is NOT forwarded on purpose: it only
    # applies to the standalone `verify` CLI command, never to a research
    # run (the research verification is controlled by verification_profile)

    # Provider-specific settings. IMPORTANT: "local" is its own provider —
    # it must never fall into the Anthropic branch.
    provider = config["provider"]
    if provider == "openai":
        if (v.get("openai_api_key") or "").strip():
            config["openai_api_key"] = v["openai_api_key"].strip()
        config["model"] = v.get("openai_model") or None
    elif provider == "anthropic":
        if (v.get("anthropic_api_key") or "").strip():
            config["anthropic_api_key"] = v["anthropic_api_key"].strip()
        config["model"] = v.get("anthropic_model") or None
    elif provider == "local":
        # API key is OPTIONAL for local servers. Empty input falls back
        # to the LOCAL_LLM_API_KEY environment variable inside the client.
        key = (v.get("local_api_key") or "").strip()
        if key:
            config["local_api_key"] = key
        if (v.get("local_model") or "").strip():
            config["model"] = v["local_model"].strip()
        if (v.get("local_base_url") or "").strip():
            config["local_base_url"] = v["local_base_url"].strip()
        config["local_backend"] = v.get("local_backend") or "ollama"

    # Target length
    for src_key, dst_key in (("target_pages", "target_pages"),
                             ("target_characters", "target_characters")):
        raw = str(v.get(src_key) or "").strip()
        if raw:
            try:
                config[dst_key] = int(raw)
            except ValueError:
                pass

    # Multilingual
    config["multilingual"] = v.get("multilingual", False)
    if config["multilingual"]:
        langs = v.get("search_languages") or []
        config["search_languages"] = langs if langs else ["ja", "en"]
    config["results_per_language"] = v.get("results_per_language", 10)
    config["translate_results"] = v.get("translate_results", True)

    # Proxy
    for key in ("http_proxy", "https_proxy", "proxy_username",
                "proxy_password"):
        if (v.get(key) or "").strip():
            config[key] = v[key].strip()
    config["verify_ssl"] = v.get("verify_ssl", True)

    return config


def describe_outcome(result) -> dict:
    """Classify a run result on THREE axes for display (pure, Tk-free).

    Returns {"headline", "ok", "process", "verification", "quality"}.
    A run whose artifact check failed, whose verification was cancelled,
    or that ended with limitations is NEVER shown as a plain success.
    """
    result = result if isinstance(result, dict) else {}
    axes = result.get("status") or {}
    process = axes.get("process") or (
        "cancelled" if result.get("verification_cancelled") else "completed")
    verification = axes.get("verification") or (
        "cancelled" if result.get("verification_cancelled") else "unknown")
    quality = axes.get("quality") or "unknown"
    run_status = result.get("run_status", "completed")
    if run_status not in ("completed",) or quality == "failed":
        headline, ok = "成果物検査に失敗しました（要確認）", False
    elif process == "cancelled" or verification == "cancelled":
        headline, ok = "中止されました（部分成果物）", False
    elif verification in ("skipped", "unknown"):
        headline, ok = "処理は完了しました（未検証）", True
    elif verification in ("timeout", "failed"):
        headline, ok = ("処理は完了しましたが検証が完了していません（要確認）",
                        False)
    elif quality == "limitations":
        headline, ok = "検証済み・制約付きで完了（要確認事項あり）", True
    else:
        headline, ok = "検証済み・品質基準を達成して完了", True
    return {"headline": headline, "ok": ok, "process": process,
            "verification": verification, "quality": quality}
