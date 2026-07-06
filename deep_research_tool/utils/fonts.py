"""
Font utilities for PDF generation (reportlab).

Shared by the report generator and other PDF exporters
(e.g., Fermi estimation export).
"""

import os


def register_japanese_font():
    """
    Register a Japanese font with reportlab for PDF generation.

    Tries common system font paths first, then falls back to
    reportlab's built-in CID fonts.

    Returns:
        Registered font name, or None if no Japanese font is available
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # Common Japanese font paths on different systems
    japanese_fonts = [
        # Linux - Noto fonts (most common)
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK"),
        ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "NotoSansCJK"),
        # Linux - other fonts
        ("/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf", "TakaoPGothic"),
        ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "IPAGothic"),
        ("/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf", "VLGothic"),
        ("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", "JapaneseGothic"),
        # Google fonts location
        ("/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf", "NotoSansJP"),
        # macOS
        ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "Hiragino"),
        ("/Library/Fonts/Arial Unicode.ttf", "ArialUnicode"),
        # Windows
        ("C:/Windows/Fonts/msgothic.ttc", "MSGothic"),
        ("C:/Windows/Fonts/meiryo.ttc", "Meiryo"),
        ("C:/Windows/Fonts/YuGothic.ttc", "YuGothic"),
    ]

    for font_path, font_name in japanese_fonts:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception:
                # Font might be in use or incompatible
                continue

    # Try CID fonts as fallback (built into reportlab for Asian languages)
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    for cid_font in ("HeiseiKakuGo-W5", "HeiseiMin-W3"):
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(cid_font))
            return cid_font
        except Exception:
            continue

    return None
