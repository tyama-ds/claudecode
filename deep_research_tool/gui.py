"""
GUI for Deep Research Tool.

Provides a graphical interface for configuring and running research.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from .api.base import get_token_stats, reset_token_stats


@dataclass
class GUIConfig:
    """GUI configuration values."""
    # Research
    topic: str = ""
    language: str = "ja"
    min_iterations: int = 3
    max_iterations: int = 10

    # API
    provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Search
    search_method: str = "duckduckgo"
    max_results: int = 10
    region: str = "wt-wt"

    # Extended Mode
    extended_mode: bool = False
    crawl_max_pages: int = 10
    crawl_max_depth: int = 2
    crawl_max_sites: int = 3

    # DeepThink
    deep_think_enabled: bool = False
    deep_think_level: float = 0.5
    reasoning_iterations: int = 3
    consistency_threshold: float = 0.3
    consistency_mode: str = "warn"
    fidelity_threshold: float = 0.7

    # Report
    output_format: str = "markdown"
    output_dir: str = "./output"
    target_pages: Optional[int] = None
    target_characters: Optional[int] = None
    include_images: bool = True
    include_citations: bool = True
    include_toc: bool = True

    # Verification
    enable_verification: bool = True
    verification_strictness: str = "medium"

    # Proxy
    http_proxy: str = ""
    https_proxy: str = ""
    proxy_username: str = ""
    proxy_password: str = ""
    verify_ssl: bool = True

    # Misc
    verbose: bool = False


class DeepResearchGUI:
    """Main GUI application for Deep Research Tool."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Deep Research Tool")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Configuration
        self.config = GUIConfig()

        # Variables for widgets
        self._init_variables()

        # Build UI
        self._build_ui()

        # Research state
        self.is_running = False
        self.research_thread: Optional[threading.Thread] = None

    def _init_variables(self):
        """Initialize tkinter variables."""
        # Research
        self.var_topic = tk.StringVar(value="")
        self.var_language = tk.StringVar(value="ja")
        self.var_min_iterations = tk.IntVar(value=3)
        self.var_max_iterations = tk.IntVar(value=10)

        # API
        self.var_provider = tk.StringVar(value="openai")
        self.var_openai_key = tk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        self.var_anthropic_key = tk.StringVar(value=os.getenv("ANTHROPIC_API_KEY", ""))
        self.var_openai_model = tk.StringVar(value="gpt-5-mini")
        self.var_anthropic_model = tk.StringVar(value="claude-3-5-sonnet-20241022")
        self.var_temperature = tk.DoubleVar(value=0.7)
        self.var_max_tokens = tk.IntVar(value=4096)

        # Search
        self.var_search_method = tk.StringVar(value="duckduckgo")
        self.var_max_results = tk.IntVar(value=10)
        self.var_region = tk.StringVar(value="wt-wt")

        # Extended Mode
        self.var_extended_mode = tk.BooleanVar(value=False)
        self.var_crawl_max_pages = tk.IntVar(value=10)
        self.var_crawl_max_depth = tk.IntVar(value=2)
        self.var_crawl_max_sites = tk.IntVar(value=3)

        # DeepThink
        self.var_deep_think = tk.BooleanVar(value=False)
        self.var_deep_think_level = tk.DoubleVar(value=0.5)
        self.var_reasoning_iterations = tk.IntVar(value=3)
        self.var_consistency_threshold = tk.DoubleVar(value=0.3)
        self.var_consistency_mode = tk.StringVar(value="warn")
        self.var_fidelity_threshold = tk.DoubleVar(value=0.7)

        # Report
        self.var_output_format = tk.StringVar(value="markdown")
        self.var_output_dir = tk.StringVar(value="./output")
        self.var_target_pages = tk.StringVar(value="")
        self.var_target_characters = tk.StringVar(value="")
        self.var_include_images = tk.BooleanVar(value=True)
        self.var_include_citations = tk.BooleanVar(value=True)
        self.var_include_toc = tk.BooleanVar(value=True)

        # Verification
        self.var_enable_verification = tk.BooleanVar(value=True)
        self.var_verification_strictness = tk.StringVar(value="medium")

        # Proxy
        self.var_http_proxy = tk.StringVar(value=os.getenv("HTTP_PROXY", ""))
        self.var_https_proxy = tk.StringVar(value=os.getenv("HTTPS_PROXY", ""))
        self.var_proxy_username = tk.StringVar(value="")
        self.var_proxy_password = tk.StringVar(value="")
        self.var_verify_ssl = tk.BooleanVar(value=True)

        # Misc
        self.var_verbose = tk.BooleanVar(value=False)

        # Multilingual
        self.var_multilingual = tk.BooleanVar(value=False)
        self.var_search_languages = {
            "ja": tk.BooleanVar(value=True),
            "en": tk.BooleanVar(value=True),
            "zh": tk.BooleanVar(value=False),
            "ko": tk.BooleanVar(value=False),
            "de": tk.BooleanVar(value=False),
            "fr": tk.BooleanVar(value=False),
            "es": tk.BooleanVar(value=False),
            "ru": tk.BooleanVar(value=False),
        }
        self.var_results_per_language = tk.IntVar(value=10)
        self.var_translate_results = tk.BooleanVar(value=True)

    def _build_ui(self):
        """Build the main user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Topic input at top
        self._build_topic_section(main_frame)

        # Notebook for settings tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create tabs
        self._build_research_tab(notebook)
        self._build_api_tab(notebook)
        self._build_search_tab(notebook)
        self._build_deep_think_tab(notebook)
        self._build_report_tab(notebook)
        self._build_multilingual_tab(notebook)
        self._build_proxy_tab(notebook)

        # Bottom section with buttons and log
        self._build_bottom_section(main_frame)

    def _build_topic_section(self, parent):
        """Build the topic input section."""
        frame = ttk.LabelFrame(parent, text="Research Topic", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="Topic:").pack(side=tk.LEFT)
        topic_entry = ttk.Entry(frame, textvariable=self.var_topic, width=80)
        topic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

    def _build_research_tab(self, notebook):
        """Build the Research Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Research")

        # Language
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Language:", width=20, anchor="w").pack(side=tk.LEFT)
        lang_combo = ttk.Combobox(row, textvariable=self.var_language,
                                  values=["ja", "en", "zh", "ko", "de", "fr", "es"],
                                  state="readonly", width=15)
        lang_combo.pack(side=tk.LEFT)
        ttk.Label(row, text="(ja=Japanese, en=English)", foreground="gray").pack(side=tk.LEFT, padx=10)

        # Min Iterations
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Min Iterations:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=10, variable=self.var_min_iterations,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_min_iterations, width=5).pack(side=tk.LEFT)

        # Max Iterations
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Iterations:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=20, variable=self.var_max_iterations,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_max_iterations, width=5).pack(side=tk.LEFT)

        # Extended Mode Section
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Extended Mode (Deep Site Crawling)",
                  font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Enable Extended Mode",
                        variable=self.var_extended_mode).pack(side=tk.LEFT)

        # Crawl settings
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Pages per Site:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=50, variable=self.var_crawl_max_pages,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_crawl_max_pages, width=5).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Crawl Depth:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=5, variable=self.var_crawl_max_depth,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_crawl_max_depth, width=5).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Sites to Crawl:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=10, variable=self.var_crawl_max_sites,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_crawl_max_sites, width=5).pack(side=tk.LEFT)

        # Verification Section
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Verification Settings",
                  font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Enable Verification",
                        variable=self.var_enable_verification).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Strictness:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_verification_strictness,
                     values=["low", "medium", "high"], state="readonly",
                     width=15).pack(side=tk.LEFT)

    def _build_api_tab(self, notebook):
        """Build the API Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="API")

        # Provider
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="LLM Provider:", width=20, anchor="w").pack(side=tk.LEFT)
        provider_combo = ttk.Combobox(row, textvariable=self.var_provider,
                                      values=["openai", "anthropic"],
                                      state="readonly", width=15)
        provider_combo.pack(side=tk.LEFT)
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # OpenAI Settings
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="OpenAI Settings", font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="API Key:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_openai_key, width=50, show="*").pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Model:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_openai_model,
                     values=["gpt-5", "gpt-5-mini", "gpt-5-thinking", "gpt-5-thinking-mini",
                            "gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
                     width=25).pack(side=tk.LEFT)

        # Anthropic Settings
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Anthropic Settings", font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="API Key:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_anthropic_key, width=50, show="*").pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Model:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_anthropic_model,
                     values=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
                            "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                     width=30).pack(side=tk.LEFT)

        # Common Settings
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Common Settings", font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Temperature:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.var_temperature,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        temp_label = ttk.Label(row, text="0.70", width=5)
        temp_label.pack(side=tk.LEFT)
        self.var_temperature.trace_add("write",
            lambda *args: temp_label.config(text=f"{self.var_temperature.get():.2f}"))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Tokens:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1024, to=16384, increment=1024,
                    textvariable=self.var_max_tokens, width=10).pack(side=tk.LEFT)

    def _build_search_tab(self, notebook):
        """Build the Search Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Search")

        # Search Method
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Search Method:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_search_method,
                     values=["duckduckgo", "selenium"],
                     state="readonly", width=15).pack(side=tk.LEFT)

        # Max Results
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Max Results:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=5, to=30, variable=self.var_max_results,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_max_results, width=5).pack(side=tk.LEFT)

        # Region
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Region:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_region,
                     values=["wt-wt", "jp-jp", "us-en", "uk-en", "de-de", "fr-fr"],
                     width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="(wt-wt=Worldwide)", foreground="gray").pack(side=tk.LEFT, padx=10)

    def _build_deep_think_tab(self, notebook):
        """Build the DeepThink Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="DeepThink")

        # Enable DeepThink
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Enable DeepThink",
                        variable=self.var_deep_think).pack(side=tk.LEFT)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # Level
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Reasoning Level:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.var_deep_think_level,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        level_label = ttk.Label(row, text="0.50", width=5)
        level_label.pack(side=tk.LEFT)
        self.var_deep_think_level.trace_add("write",
            lambda *args: level_label.config(text=f"{self.var_deep_think_level.get():.2f}"))
        ttk.Label(row, text="(0=Conservative, 1=Exploratory)",
                  foreground="gray").pack(side=tk.LEFT, padx=10)

        # Reasoning Iterations
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Reasoning Iterations:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=1, to=10, variable=self.var_reasoning_iterations,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_reasoning_iterations, width=5).pack(side=tk.LEFT)

        # Consistency Threshold
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Consistency Threshold:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.var_consistency_threshold,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ct_label = ttk.Label(row, text="0.30", width=5)
        ct_label.pack(side=tk.LEFT)
        self.var_consistency_threshold.trace_add("write",
            lambda *args: ct_label.config(text=f"{self.var_consistency_threshold.get():.2f}"))

        # Consistency Mode
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Consistency Mode:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_consistency_mode,
                     values=["warn", "revise", "strict"],
                     state="readonly", width=15).pack(side=tk.LEFT)
        ttk.Label(row, text="(warn=Log only, revise=Auto-fix, strict=Fail)",
                  foreground="gray").pack(side=tk.LEFT, padx=10)

        # Fidelity Threshold
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Fidelity Threshold:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0.0, to=1.0, variable=self.var_fidelity_threshold,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ft_label = ttk.Label(row, text="0.70", width=5)
        ft_label.pack(side=tk.LEFT)
        self.var_fidelity_threshold.trace_add("write",
            lambda *args: ft_label.config(text=f"{self.var_fidelity_threshold.get():.2f}"))

        # Description
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        desc_text = """DeepThink enhances research with statistical and logical reasoning:

- Reasoning Level: Higher values allow more exploratory reasoning
- Reasoning Iterations: Number of reasoning cycles per section
- Consistency Threshold: Lower values are stricter about logical consistency
- Consistency Mode: How to handle detected inconsistencies
- Fidelity Threshold: Minimum source fidelity score required"""

        desc_label = ttk.Label(frame, text=desc_text, justify=tk.LEFT, foreground="gray")
        desc_label.pack(anchor="w")

    def _build_report_tab(self, notebook):
        """Build the Report Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Report")

        # Output Format
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Output Format:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.var_output_format,
                     values=["markdown", "pdf", "docx"],
                     state="readonly", width=15).pack(side=tk.LEFT)

        # Output Directory
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Output Directory:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_output_dir, width=40).pack(side=tk.LEFT)
        ttk.Button(row, text="Browse...", command=self._browse_output_dir).pack(side=tk.LEFT, padx=5)

        # Target Length
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Target Length (Optional)", font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Target Pages:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_target_pages, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text="(Leave empty for auto)", foreground="gray").pack(side=tk.LEFT, padx=10)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Target Characters:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_target_characters, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text="(Leave empty for auto)", foreground="gray").pack(side=tk.LEFT, padx=10)

        # Include Options
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Include Options", font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Include Images",
                        variable=self.var_include_images).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row, text="Include Citations",
                        variable=self.var_include_citations).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(row, text="Include Table of Contents",
                        variable=self.var_include_toc).pack(side=tk.LEFT)

        # Verbose
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Verbose Output",
                        variable=self.var_verbose).pack(side=tk.LEFT)

    def _build_multilingual_tab(self, notebook):
        """Build the Multilingual Search tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Multilingual")

        # Enable Multilingual
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Enable Multilingual Search",
                        variable=self.var_multilingual).pack(side=tk.LEFT)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # Language Selection
        ttk.Label(frame, text="Search Languages", font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Select languages to search in:",
                  foreground="gray").pack(anchor="w", pady=(0, 10))

        # Language checkboxes in a grid
        lang_frame = ttk.Frame(frame)
        lang_frame.pack(fill=tk.X, pady=5)

        languages = [
            ("ja", "Japanese (日本語)"),
            ("en", "English"),
            ("zh", "Chinese (中文)"),
            ("ko", "Korean (한국어)"),
            ("de", "German (Deutsch)"),
            ("fr", "French (Français)"),
            ("es", "Spanish (Español)"),
            ("ru", "Russian (Русский)"),
        ]

        for i, (code, name) in enumerate(languages):
            row_num = i // 2
            col_num = i % 2
            cb = ttk.Checkbutton(lang_frame, text=name,
                                 variable=self.var_search_languages[code])
            cb.grid(row=row_num, column=col_num, sticky="w", padx=10, pady=2)

        # Results per language
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Results per Language:", width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Scale(row, from_=5, to=20, variable=self.var_results_per_language,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.var_results_per_language, width=5).pack(side=tk.LEFT)

        # Translate results
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Translate results to output language",
                        variable=self.var_translate_results).pack(side=tk.LEFT)

        # Description
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        desc_text = """Multilingual search enables searching across multiple languages:

- Query Translation: Your search query is translated to each selected language
- Parallel Search: Searches are performed in all languages simultaneously
- Result Aggregation: Results are combined and deduplicated
- Source Tracking: Each source's original language is recorded

This helps gather more comprehensive information from diverse sources."""

        desc_label = ttk.Label(frame, text=desc_text, justify=tk.LEFT, foreground="gray")
        desc_label.pack(anchor="w")

    def _build_proxy_tab(self, notebook):
        """Build the Proxy Settings tab."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Proxy")

        # HTTP Proxy
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="HTTP Proxy:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_http_proxy, width=50).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="", width=20).pack(side=tk.LEFT)
        ttk.Label(row, text="e.g., http://proxy.example.com:8080",
                  foreground="gray").pack(side=tk.LEFT)

        # HTTPS Proxy
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="HTTPS Proxy:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_https_proxy, width=50).pack(side=tk.LEFT)

        # Authentication
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(frame, text="Proxy Authentication (Optional)",
                  font=("", 10, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Username:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_proxy_username, width=30).pack(side=tk.LEFT)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Password:", width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_proxy_password, width=30, show="*").pack(side=tk.LEFT)

        # SSL
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row, text="Verify SSL Certificates",
                        variable=self.var_verify_ssl).pack(side=tk.LEFT)
        ttk.Label(row, text="(Disable for self-signed certificates)",
                  foreground="gray").pack(side=tk.LEFT, padx=10)

    def _build_bottom_section(self, parent):
        """Build the bottom section with buttons and log."""
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_start = ttk.Button(btn_frame, text="Start Research",
                                    command=self._start_research)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(btn_frame, text="Stop",
                                   command=self._stop_research, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(btn_frame, text="Reset to Defaults",
                   command=self._reset_defaults).pack(side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(btn_frame, variable=self.progress_var,
                                            maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT)

        self.progress_label = ttk.Label(btn_frame, text="Ready")
        self.progress_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                   state=tk.DISABLED,
                                                   wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _on_provider_change(self, event=None):
        """Handle provider change."""
        pass  # Can be extended to show/hide provider-specific settings

    def _browse_output_dir(self):
        """Open directory browser for output directory."""
        directory = filedialog.askdirectory(initialdir=self.var_output_dir.get())
        if directory:
            self.var_output_dir.set(directory)

    def _reset_defaults(self):
        """Reset all settings to defaults."""
        self.var_language.set("ja")
        self.var_min_iterations.set(3)
        self.var_max_iterations.set(10)
        self.var_provider.set("openai")
        self.var_openai_model.set("gpt-5-mini")
        self.var_anthropic_model.set("claude-3-5-sonnet-20241022")
        self.var_temperature.set(0.7)
        self.var_max_tokens.set(4096)
        self.var_search_method.set("duckduckgo")
        self.var_max_results.set(10)
        self.var_region.set("wt-wt")
        self.var_extended_mode.set(False)
        self.var_crawl_max_pages.set(10)
        self.var_crawl_max_depth.set(2)
        self.var_crawl_max_sites.set(3)
        self.var_deep_think.set(False)
        self.var_deep_think_level.set(0.5)
        self.var_reasoning_iterations.set(3)
        self.var_consistency_threshold.set(0.3)
        self.var_consistency_mode.set("warn")
        self.var_fidelity_threshold.set(0.7)
        self.var_output_format.set("markdown")
        self.var_output_dir.set("./output")
        self.var_target_pages.set("")
        self.var_target_characters.set("")
        self.var_include_images.set(True)
        self.var_include_citations.set(True)
        self.var_include_toc.set(True)
        self.var_enable_verification.set(True)
        self.var_verification_strictness.set("medium")
        self.var_verify_ssl.set(True)
        self.var_verbose.set(False)

        # Multilingual defaults
        self.var_multilingual.set(False)
        for code, var in self.var_search_languages.items():
            var.set(code in ["ja", "en"])  # Only ja and en by default
        self.var_results_per_language.set(10)
        self.var_translate_results.set(True)

        self._log("Settings reset to defaults")

    def _log(self, message: str):
        """Add message to log area."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _update_progress(self, message: str, progress: float):
        """Update progress bar and label."""
        self.progress_label.config(text=message)
        self.progress_var.set(progress)
        self._log(f"[{progress:.0f}%] {message}")

    def _validate_config(self) -> bool:
        """Validate configuration before starting research."""
        topic = self.var_topic.get().strip()
        if not topic:
            messagebox.showerror("Validation Error", "Please enter a research topic.")
            return False

        provider = self.var_provider.get()
        if provider == "openai" and not self.var_openai_key.get().strip():
            messagebox.showerror("Validation Error",
                               "OpenAI API key is required.\nSet it in the API tab or via OPENAI_API_KEY environment variable.")
            return False

        if provider == "anthropic" and not self.var_anthropic_key.get().strip():
            messagebox.showerror("Validation Error",
                               "Anthropic API key is required.\nSet it in the API tab or via ANTHROPIC_API_KEY environment variable.")
            return False

        return True

    def _get_config_dict(self) -> dict:
        """Get configuration as a dictionary for run_research."""
        config = {
            "topic": self.var_topic.get().strip(),
            "provider": self.var_provider.get(),
            "search_method": self.var_search_method.get(),
            "research_iterations": self.var_min_iterations.get(),
            "max_iterations": self.var_max_iterations.get(),
            "output_format": self.var_output_format.get(),
            "output_dir": self.var_output_dir.get(),
            "enable_verification": self.var_enable_verification.get(),
            "verbose": self.var_verbose.get(),
            "language": self.var_language.get(),
            "max_results": self.var_max_results.get(),

            # Extended Mode
            "extended_mode": self.var_extended_mode.get(),
            "crawl_max_pages": self.var_crawl_max_pages.get(),
            "crawl_max_depth": self.var_crawl_max_depth.get(),
            "crawl_max_sites": self.var_crawl_max_sites.get(),

            # DeepThink
            "deep_think": self.var_deep_think.get(),
            "deep_think_level": self.var_deep_think_level.get(),
            "reasoning_iterations": self.var_reasoning_iterations.get(),
            "consistency_threshold": self.var_consistency_threshold.get(),
            "consistency_mode": self.var_consistency_mode.get(),
            "fidelity_threshold": self.var_fidelity_threshold.get(),
        }

        # API keys
        if self.var_provider.get() == "openai":
            config["openai_api_key"] = self.var_openai_key.get().strip()
            config["model"] = self.var_openai_model.get()
        else:
            config["anthropic_api_key"] = self.var_anthropic_key.get().strip()
            config["model"] = self.var_anthropic_model.get()

        # Target length
        if self.var_target_pages.get().strip():
            try:
                config["target_pages"] = int(self.var_target_pages.get().strip())
            except ValueError:
                pass

        if self.var_target_characters.get().strip():
            try:
                config["target_characters"] = int(self.var_target_characters.get().strip())
            except ValueError:
                pass

        # Multilingual
        config["multilingual"] = self.var_multilingual.get()
        if self.var_multilingual.get():
            selected_languages = [
                code for code, var in self.var_search_languages.items()
                if var.get()
            ]
            config["search_languages"] = selected_languages if selected_languages else ["ja", "en"]
        config["results_per_language"] = self.var_results_per_language.get()
        config["translate_results"] = self.var_translate_results.get()

        # Proxy
        if self.var_http_proxy.get().strip():
            config["http_proxy"] = self.var_http_proxy.get().strip()
        if self.var_https_proxy.get().strip():
            config["https_proxy"] = self.var_https_proxy.get().strip()
        if self.var_proxy_username.get().strip():
            config["proxy_username"] = self.var_proxy_username.get().strip()
        if self.var_proxy_password.get().strip():
            config["proxy_password"] = self.var_proxy_password.get().strip()
        config["verify_ssl"] = self.var_verify_ssl.get()

        return config

    def _start_research(self):
        """Start the research process."""
        if not self._validate_config():
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self._log("=" * 50)
        self._log("Starting research...")

        config = self._get_config_dict()
        self._log(f"Topic: {config['topic']}")
        self._log(f"Provider: {config['provider']}")
        self._log(f"DeepThink: {'Enabled' if config['deep_think'] else 'Disabled'}")
        self._log(f"Extended Mode: {'Enabled' if config['extended_mode'] else 'Disabled'}")
        self._log(f"Multilingual: {'Enabled' if config['multilingual'] else 'Disabled'}")
        if config['multilingual'] and 'search_languages' in config:
            self._log(f"  Languages: {', '.join(config['search_languages'])}")

        # Run research in background thread
        self.research_thread = threading.Thread(
            target=self._run_research_thread,
            args=(config,),
            daemon=True
        )
        self.research_thread.start()

    def _run_research_thread(self, config: dict):
        """Run research in a background thread."""
        try:
            from .main import run_research

            # Reset token stats before running
            reset_token_stats()

            def progress_callback(message: str, progress: float):
                self.root.after(0, lambda: self._update_progress(message, progress))

            topic = config.pop("topic")
            result = run_research(
                topic=topic,
                progress_callback=progress_callback,
                **config
            )

            self.root.after(0, lambda: self._on_research_complete(result))

        except Exception as e:
            self.root.after(0, lambda: self._on_research_error(str(e)))

    def _on_research_complete(self, result):
        """Handle research completion."""
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_var.set(100)
        self.progress_label.config(text="Complete")

        self._log("=" * 50)
        self._log("Research completed!")

        if result:
            if isinstance(result, dict) and "report_path" in result:
                self._log(f"Report saved to: {result['report_path']}")
            elif hasattr(result, "report_path"):
                self._log(f"Report saved to: {result.report_path}")

        # Display token usage summary
        token_stats = get_token_stats()
        language = self.var_language.get()
        self._log("")
        self._log(token_stats.get_summary(language))

        # Build completion message with token info
        completion_msg = "Research completed successfully!\n\n"
        completion_msg += f"Total tokens used: {token_stats.total_tokens:,}\n"
        completion_msg += f"  - Input: {token_stats.total_prompt_tokens:,}\n"
        completion_msg += f"  - Output: {token_stats.total_completion_tokens:,}\n"
        completion_msg += f"API calls: {token_stats.total_calls}"

        messagebox.showinfo("Complete", completion_msg)

    def _on_research_error(self, error: str):
        """Handle research error."""
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_label.config(text="Error")

        self._log(f"ERROR: {error}")
        messagebox.showerror("Error", f"Research failed:\n{error}")

    def _stop_research(self):
        """Stop the research process."""
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_label.config(text="Stopped")
        self._log("Research stopped by user")


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()

    # Set theme
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except tk.TclError:
        pass  # Theme not available, use default

    app = DeepResearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
