"""
Provisional GUI for Fermi estimation (フェルミ推定 仮GUI).

A minimal Tkinter interface for running Fermi estimations with the
FermiEstimator. Intended as a provisional tool; the main research GUI
lives in gui.py.

Usage:
    python -m deep_research_tool.fermi_gui
    or
    from deep_research_tool import launch_fermi_gui; launch_fermi_gui()
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


class FermiEstimatorGUI:
    """Provisional Tkinter GUI for Fermi estimation."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("フェルミ推定ツール (仮) - Deep Research Tool")
        self.root.geometry("820x680")
        self.last_estimate = None  # Most recent FermiEstimate for export
        self._build_widgets()

    def _build_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Question ---
        ttk.Label(main, text="質問（定量的な問い）:").pack(anchor=tk.W)
        self.question_entry = ttk.Entry(main, width=100)
        self.question_entry.pack(fill=tk.X, pady=(2, 8))
        self.question_entry.insert(0, "日本国内のピアノ調律師の人数は？")

        # --- API settings row ---
        api_frame = ttk.LabelFrame(main, text="API設定", padding=8)
        api_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(api_frame, text="プロバイダー:").grid(row=0, column=0, sticky=tk.W)
        self.provider_var = tk.StringVar(value="openai")
        ttk.Combobox(
            api_frame, textvariable=self.provider_var,
            values=["openai", "anthropic", "local"],
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky=tk.W, padx=(4, 16))

        ttk.Label(api_frame, text="モデル (任意):").grid(row=0, column=2, sticky=tk.W)
        self.model_entry = ttk.Entry(api_frame, width=28)
        self.model_entry.grid(row=0, column=3, sticky=tk.W, padx=(4, 16))

        ttk.Label(api_frame, text="言語:").grid(row=0, column=4, sticky=tk.W)
        self.language_var = tk.StringVar(value="ja")
        ttk.Combobox(
            api_frame, textvariable=self.language_var,
            values=["ja", "en"], state="readonly", width=6,
        ).grid(row=0, column=5, sticky=tk.W, padx=4)

        ttk.Label(api_frame, text="APIキー (任意、未入力時は環境変数):").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        self.api_key_entry = ttk.Entry(api_frame, width=60, show="*")
        self.api_key_entry.grid(row=1, column=2, columnspan=4, sticky=tk.W + tk.E, pady=(6, 0))

        # --- Known values ---
        known_frame = ttk.LabelFrame(
            main, text="既知の値 (任意、1行に「名前=数値」形式)", padding=8)
        known_frame.pack(fill=tk.X, pady=(0, 8))
        self.known_text = tk.Text(known_frame, height=3, width=100)
        self.known_text.pack(fill=tk.X)
        self.known_text.insert("1.0", "# 例: 日本の人口=125000000")

        # --- Context ---
        ttk.Label(main, text="参考情報 (任意):").pack(anchor=tk.W)
        self.context_text = tk.Text(main, height=3, width=100)
        self.context_text.pack(fill=tk.X, pady=(2, 8))

        # --- Run/save buttons + status ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 8))
        self.run_button = ttk.Button(btn_frame, text="推定実行", command=self._on_run)
        self.run_button.pack(side=tk.LEFT)
        self.save_docx_button = ttk.Button(
            btn_frame, text="Word保存", command=self._on_save_docx, state=tk.DISABLED)
        self.save_docx_button.pack(side=tk.LEFT, padx=(8, 0))
        self.save_pdf_button = ttk.Button(
            btn_frame, text="PDF保存", command=self._on_save_pdf, state=tk.DISABLED)
        self.save_pdf_button.pack(side=tk.LEFT, padx=(8, 0))
        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(btn_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        # --- Results ---
        ttk.Label(main, text="推定結果:").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(main, height=18, width=100)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

    def _parse_known_values(self):
        values = {}
        for line in self.known_text.get("1.0", tk.END).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, _, num = line.partition("=")
                try:
                    values[name.strip()] = float(num.strip().replace(",", ""))
                except ValueError:
                    pass
        return values or None

    def _on_run(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("入力エラー", "質問を入力してください。")
            return

        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("推定中... (LLM呼び出し)")
        self.result_text.delete("1.0", tk.END)

        thread = threading.Thread(target=self._run_estimation, args=(question,), daemon=True)
        thread.start()

    def _run_estimation(self, question: str):
        try:
            from deep_research_tool.api import get_client
            from deep_research_tool.thinking import FermiEstimator

            api_key = self.api_key_entry.get().strip() or None
            model = self.model_entry.get().strip() or None

            llm = get_client(
                provider=self.provider_var.get(),
                api_key=api_key,
                model=model,
            )
            estimator = FermiEstimator(llm_client=llm, language=self.language_var.get())

            result = estimator.estimate(
                question=question,
                context=self.context_text.get("1.0", tk.END).strip(),
                known_values=self._parse_known_values(),
            )
            self.root.after(0, self._show_result, result)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _show_result(self, estimate):
        self.last_estimate = estimate
        self.result_text.insert("1.0", estimate.to_markdown())
        self.status_var.set("完了")
        self.run_button.config(state=tk.NORMAL)
        self.save_docx_button.config(state=tk.NORMAL)
        self.save_pdf_button.config(state=tk.NORMAL)

    def _show_error(self, message: str):
        self.result_text.insert("1.0", f"エラー: {message}\n")
        self.status_var.set("エラー")
        self.run_button.config(state=tk.NORMAL)
        messagebox.showerror("推定エラー", message)

    def _on_save_docx(self):
        self._save_estimate(
            extension=".docx",
            filetypes=[("Word文書", "*.docx")],
            save_func=lambda est, path: est.save_docx(path),
        )

    def _on_save_pdf(self):
        self._save_estimate(
            extension=".pdf",
            filetypes=[("PDF文書", "*.pdf")],
            save_func=lambda est, path: est.save_pdf(path),
        )

    def _save_estimate(self, extension: str, filetypes, save_func):
        if self.last_estimate is None:
            messagebox.showwarning("保存エラー", "先に推定を実行してください。")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=extension,
            filetypes=filetypes,
            initialfile=f"fermi_estimate{extension}",
        )
        if not filepath:
            return

        try:
            saved = save_func(self.last_estimate, filepath)
            self.status_var.set(f"保存しました: {saved}")
            messagebox.showinfo("保存完了", f"保存しました:\n{saved}")
        except ImportError as e:
            # Missing optional dependency (python-docx / reportlab)
            messagebox.showerror("依存パッケージ不足", str(e))
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))


def main():
    """Launch the provisional Fermi estimation GUI."""
    root = tk.Tk()
    FermiEstimatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
