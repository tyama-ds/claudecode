"""
v1.13.0 — charts in the REPORT (group "report"):

- item 7: every placed figure gets 「図N　タイトル」 above the image (English
  'Figure N. Title'), numbered in report order across chapters, and ONE
  caption block below it (caption text, then 「見るべき点: …」 once) —
  Markdown, Word (V1/V2 insertion and V3 inline), PDF and HTML. The
  number is never stored in Figure.caption (frozen into the manifest),
  and the semantic artifact check still passes.
- item 8: the report-wide chart cap (chart_max_total) keeps panels
  round-robin across chapters and records what it left out.
- item 9: Word placement — the last chapter's figures end its body,
  before the back matter and before w:sectPr, title / image / caption in
  order and kept together (keep_with_next).
- item 14: Appendix D never prints '=None', prints the judgment once,
  line_xy points as label=value, one line per series.
- item 15: V2 + PDF makes a real PDF (Japanese font, figures), falls back
  to Markdown with a CRITICAL warning only when rendering fails, and
  honours strict_format.

No paid API, no network: fake clients only.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.report.figure_table_generator import (
    Figure, FigureTableCollection, FigureTableGenerator, FigureType, TableData,
    figure_caption_lines, figure_heading, judgment_label)
from deep_research_tool.report.semantic_manifest import (
    build_semantic_manifest, check_artifact, figure_semantics, figure_semantics_markdown,
    manifest_hash)


def _png(path: Path, size=(800, 500)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _chart(fid, title, caption, section, png, **kw):
    return Figure(figure_id=fid, figure_type=FigureType.CHART, title=title, caption=caption,
                  section_id=section, image_path=png, **kw)


def _collection(png):
    """Two charts in chapter 2 (the LAST one), one in chapter 1, a table
    in chapter 2 and an orphan."""
    return FigureTableCollection(
        charts=[_chart("c1", "市場規模の推移（億円）", "出所 2 件（公表）　見るべき点: 3年で1.5倍に拡大した。",
                       "1", png),
                _chart("c2", "主要3社の売上高（億円）", "出所 1 件　見るべき点: 首位と2位の差が大きい。", "2", png),
                _chart("c3", "地域別の構成比（%）", "地域別の構成。", "2", png),
                _chart("c9", "どの章にも属さない図", "章外の図の説明。", "9", png)],
        tables=[TableData(table_id="t1", title="主要企業のシェア", caption="出所: 各社IR",
                          headers=["社名", "シェア"], rows=[["A社", "52%"], ["B社", "0"]], section_id="2")])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class TestHelpers:

    def test_heading_and_caption_block(self):
        assert figure_heading(3, "市場規模（億円）") == "図3　市場規模（億円）"
        assert figure_heading(3, "Market size", "en") == "Figure 3. Market size"
        assert figure_heading(1, "") == "図1"
        assert judgment_label("ja") == "見るべき点: " and judgment_label("en") == "Key point: "
        fig = _chart("x", "t", "出所 2 件　見るべき点: 回復基調。", "1", None)
        assert figure_caption_lines(fig) == ["出所 2 件", "見るべき点: 回復基調。"]
        # judgment only / caption only / a caption that just repeats the title
        assert figure_caption_lines(_chart("x", "t", "見るべき点: 回復。", "1", None)) == ["見るべき点: 回復。"]
        assert figure_caption_lines(_chart("x", "t", "出所のみ。", "1", None)) == ["出所のみ。"]
        assert figure_caption_lines(_chart("x", "同じ", "同じ", "1", None)) == []
        en = _chart("x", "t", "Source: 2 items　Key point: recovering.", "1", None)
        assert figure_caption_lines(en) == ["Source: 2 items", "Key point: recovering."]


# --------------------------------------------------------------------------
# Markdown (item 7 + chapter-end placement)
# --------------------------------------------------------------------------

MD = """# 調査レポート

## 1. 市場規模

第1章の最初の段落。

第1章の二つ目の段落。

## 2. 競合

第2章の本文。

### 小見出し

第2章の続き。

## 付録D. 図表一覧

付録の本文。

---

## References

[^1]: 出典
"""


class TestMarkdown:

    def test_numbered_in_report_order_and_placed_at_each_chapter_end(self, tmp_path):
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        captions = [c.caption for c in coll.charts]
        gen = FigureTableGenerator(llm_client=None, output_dir=tmp_path, language="ja")
        out = gen.add_figures_to_markdown(MD, coll)
        # 図N above each image, numbered across chapters (orphan last)
        assert out.index("**図1　市場規模の推移（億円）**") < out.index("![市場規模の推移（億円）]")
        assert out.index("**図2　主要3社の売上高（億円）**") < out.index("**図3　地域別の構成比（%）**")
        assert "**図4　どの章にも属さない図**" in out
        # chapter 1: after ALL its text, before chapter 2
        assert out.index("第1章の二つ目の段落。") < out.index("**図1") < out.index("## 2. 競合")
        # the LAST chapter: after its sub-section, before the appendix and References
        assert out.index("第2章の続き。") < out.index("**図2") < out.index("## 付録D")
        assert out.index("主要企業のシェア") < out.index("## 付録D")
        assert out.index("## References") < out.index("**図4")
        # ONE caption block: the caption, then 見るべき点 once, under the image
        block = out[out.index("![市場規模の推移（億円）]"):out.index("## 2. 競合")]
        assert "*出所 2 件（公表）*\n*見るべき点: 3年で1.5倍に拡大した。*" in block
        assert out.count("見るべき点: 3年で1.5倍に拡大した。") == 1
        # the number is a rendering detail: nothing stored in the collection changed
        assert [c.caption for c in coll.charts] == captions
        assert not any(c.caption.startswith("図") or c.title.startswith("図") for c in coll.charts)

    def test_v1_markdown_without_blank_line_after_heading(self, tmp_path):
        # V1 writes the chapter text right under its heading: the figures
        # used to land after the FIRST paragraph; now after the chapter
        png = _png(tmp_path / "c.png")
        md = "## 1. 市場\n段落1。\n\n段落2。\n\n## 2. 競合\n本文2。\n"
        coll = FigureTableCollection(charts=[_chart("c1", "図題", "説明。", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_markdown(md, coll)
        assert out.index("段落2。") < out.index("**図1　図題**") < out.index("## 2. 競合")

    def test_missing_image_gets_no_number(self, tmp_path):
        png = _png(tmp_path / "c.png")
        coll = FigureTableCollection(charts=[
            _chart("a", "無い図", "無い図の説明。", "1", tmp_path / "missing.png"),
            _chart("b", "ある図", "ある図の説明。", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_markdown(
            "## 1. 章\n\n本文。\n", coll)
        assert "<!-- Chart not found:" in out and "*無い図の説明。*" in out
        assert "**図1　ある図**" in out and "図2" not in out

    def test_english_numbering(self, tmp_path):
        png = _png(tmp_path / "c.png")
        coll = FigureTableCollection(charts=[_chart("a", "Market size", "Source: stats　Key point: up.", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path, language="en").add_figures_to_markdown(
            "## 1. Market\n\nText.\n", coll)
        assert "**Figure 1. Market size**" in out and "*Source: stats*\n*Key point: up.*" in out

    def test_artifact_check_passes_with_numbers(self, tmp_path):
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        gen = FigureTableGenerator(llm_client=None, output_dir=tmp_path, language="ja")
        path = tmp_path / "r.md"
        path.write_text(gen.add_figures_to_markdown(MD, coll), encoding="utf-8")
        manifest = build_semantic_manifest({"1": "第1章の最初の段落。\n\n第1章の二つ目の段落。"},
                                           figure_collection=coll)
        res = check_artifact(manifest, path)
        assert res["status"] == "pass", res


# --------------------------------------------------------------------------
# Word: V1 / V2 post-render insertion (items 7 + 9)
# --------------------------------------------------------------------------

def _body_rows(docx_path):
    """[(kind, text, has_image, keep_with_next)] for every body child."""
    import docx
    from docx.text.paragraph import Paragraph
    d = docx.Document(str(docx_path))
    rows = []
    for el in d.element.body:
        tag = el.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(el, d)
            rows.append(("p", p.text, bool(el.xpath(".//pic:pic")),
                         bool(p.paragraph_format.keep_with_next)))
        else:
            rows.append((tag, "", False, False))
    return rows


class TestWordInsertion:

    def _v1_like(self, tmp_path):
        import docx
        doc = docx.Document()
        doc.add_heading("調査レポート", 0)
        doc.add_heading("Table of Contents", level=1)
        doc.add_heading("1. 市場規模", level=1)
        doc.add_paragraph("本文1。")
        doc.add_heading("2. 競合", level=1)
        doc.add_paragraph("本文2。")
        doc.add_heading("付録D. 図表一覧", level=1)
        doc.add_heading("図: 市場規模の推移（億円）", level=2)
        doc.add_paragraph("付録の本文。")
        doc.add_heading("Verification Summary", level=1)
        doc.add_heading("Source Quality Summary", level=1)
        doc.add_heading("References", level=1)
        doc.add_paragraph("1. [A] 出典")
        src = tmp_path / "r.docx"
        doc.save(src)
        return src

    def test_last_chapter_figures_end_its_body_before_back_matter_and_sectpr(self, tmp_path):
        pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        gen = FigureTableGenerator(llm_client=None, output_dir=tmp_path, language="ja")
        out = gen.add_figures_to_docx(docx_path=self._v1_like(tmp_path), collection=coll)
        rows = _body_rows(out)
        texts = [r[1] for r in rows]
        kinds = [r[0] for r in rows]
        # nothing after the section properties
        assert kinds[-1] == "sectPr"
        i2 = texts.index("2. 競合")
        i_app = texts.index("付録D. 図表一覧")
        t2, t3 = texts.index("図2　主要3社の売上高（億円）"), texts.index("図3　地域別の構成比（%）")
        assert texts.index("本文2。") < t2 < t3 < i_app
        assert i_app < texts.index("References") < texts.index("図4　どの章にも属さない図")
        # chapter 1's figure before chapter 2
        t1 = texts.index("図1　市場規模の推移（億円）")
        assert texts.index("本文1。") < t1 < i2
        # title -> image -> caption, in order; title and image keep with next
        for t in (t1, t2, t3):
            assert rows[t][3] and rows[t + 1][2] and rows[t + 1][3]
            assert not rows[t + 2][2]
        assert texts[t1 + 2] == "出所 2 件（公表）\n見るべき点: 3年で1.5倍に拡大した。"
        assert texts[t3 + 2] == "地域別の構成。"
        # the table follows the charts, still inside chapter 2
        assert t3 < texts.index("主要企業のシェア") < i_app
        assert kinds[texts.index("主要企業のシェア") + 1] == "tbl"
        # one image per chart, the judgment printed once
        assert sum(r[2] for r in rows) == 4
        assert sum("見るべき点: 3年で1.5倍" in t for t in texts) == 1

    def test_v2_style_document_and_repeated_numbers_in_back_matter(self, tmp_path):
        """V2 converts '## 1. …' to Heading 1 and '### …' to Heading 2; a
        numbered heading inside the back matter is not a chapter again."""
        docx = pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        doc = docx.Document()
        doc.add_heading("調査レポート", 0)
        doc.add_heading("目次", level=1)
        doc.add_heading("1. 市場規模", level=1)
        doc.add_heading("背景", level=2)
        doc.add_paragraph("本文1。")
        doc.add_heading("参考文献", level=2)          # a chapter's own list: back matter
        doc.add_paragraph("文献。")
        doc.add_heading("付録A. フェルミ推定", level=1)
        doc.add_heading("1. 前提", level=2)            # number already used: not a chapter
        doc.add_paragraph("前提の本文。")
        src = tmp_path / "v2.docx"
        doc.save(src)
        coll = FigureTableCollection(charts=[_chart("c1", "図題", "説明。", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_docx(
            docx_path=src, collection=coll)
        texts = [r[1] for r in _body_rows(out)]
        assert texts.count("図1　図題") == 1
        assert texts.index("本文1。") < texts.index("図1　図題") < texts.index("参考文献")

    def test_document_without_back_matter(self, tmp_path):
        docx = pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        doc = docx.Document()
        doc.add_heading("1. 市場規模", level=1)
        doc.add_paragraph("本文1。")
        src = tmp_path / "plain.docx"
        doc.save(src)
        coll = FigureTableCollection(charts=[_chart("a", "図A", "A。", "1", png),
                                             _chart("b", "図B", "B。", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_docx(
            docx_path=src, collection=coll)
        rows = _body_rows(out)
        texts = [r[1] for r in rows]
        assert rows[-1][0] == "sectPr"
        assert texts.index("図1　図A") < texts.index("A。") < texts.index("図2　図B") \
            < texts.index("B。")

    def test_a_broken_image_leaves_no_title_behind(self, tmp_path):
        pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a png")
        coll = FigureTableCollection(charts=[_chart("a", "壊れた図", "壊れた。", "1", bad),
                                             _chart("b", "正しい図", "正しい。", "1", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_docx(
            docx_path=self._v1_like(tmp_path), collection=coll)
        texts = [r[1] for r in _body_rows(out)]
        assert not any("壊れた" in t for t in texts)
        assert "図1　正しい図" in texts            # numbering skips the broken one

    def test_artifact_check_passes(self, tmp_path):
        pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_docx(
            docx_path=self._v1_like(tmp_path), collection=coll)
        manifest = build_semantic_manifest({"1": "本文1。 本文2。"}, figure_collection=coll)
        res = check_artifact(manifest, out)
        assert res["status"] == "pass", res


# --------------------------------------------------------------------------
# Word: V3 inline
# --------------------------------------------------------------------------

class TestV3Inline:

    def _gen(self, language="ja"):
        from deep_research_tool.report.v3.docx_generator import DocxReportGeneratorV3
        gen = DocxReportGeneratorV3.__new__(DocxReportGeneratorV3)
        gen.language = language
        gen._figure_no = 0
        return gen

    def test_numbered_title_image_caption_kept_together(self, tmp_path):
        docx = pytest.importorskip("docx")
        png = _png(tmp_path / "c.png")
        gen = self._gen()
        doc = docx.Document()
        gen._insert_section_figures(doc, {
            "charts": [_chart("a", "図A", "出所A　見るべき点: Aを見る。", "1", png)],
            "tables": [], "figures": []})
        gen._add_figure_to_doc(doc, _chart("m", "無い図", "無い図の説明。", "1", tmp_path / "missing.png"))
        gen._insert_section_figures(doc, {
            "charts": [_chart("b", "図B", "出所B", "2", png)], "tables": [], "figures": []})
        doc.save(tmp_path / "v3.docx")
        rows = _body_rows(tmp_path / "v3.docx")
        texts = [r[1] for r in rows]
        a = texts.index("図1　図A")
        assert rows[a][3] and rows[a + 1][2] and rows[a + 1][3]
        assert texts[a + 2] == "出所A\n見るべき点: Aを見る。"
        # a missing image: placeholder + caption, no number used
        assert "[Image: 無い図]" in texts and "無い図の説明。" in texts
        assert not any(t.startswith("図2　無い図") for t in texts)
        b = texts.index("図2　図B")
        assert rows[b + 1][2] and texts[b + 2] == "出所B"

    def test_generate_and_save_numbers_across_chapters_before_back_matter(self, tmp_path):
        pytest.importorskip("docx")
        from unittest.mock import MagicMock
        from deep_research_tool.report.v2.context import ReportContext
        from deep_research_tool.report.v2.generator import ChapterContent, GenerationResult
        from deep_research_tool.report.v3.docx_generator import DocxReportGeneratorV3
        png = _png(tmp_path / "c.png")
        gen = DocxReportGeneratorV3(llm_client=MagicMock(), language="ja")
        ctx = ReportContext(research_topic="炭素繊維")
        ctx.report_title = "炭素繊維の調査"
        chapters = {
            "1": ChapterContent(section_number="1", section_title="市場", content="本文1。", word_count=4),
            "2": ChapterContent(section_number="2", section_title="競合", content="本文2。", word_count=4),
            "付録D": ChapterContent(section_number="付録D", section_title="図表一覧",
                                  content="## 付録D. 図表一覧\n\n付録の本文。", word_count=10),
        }
        result = GenerationResult(chapters=chapters, context=ctx)
        coll = _collection(png)
        path = gen.generate_and_save(result, tmp_path, "v3r", figure_collection=coll,
                                     include_glossary=False)
        rows = _body_rows(path)
        texts = [r[1] for r in rows]
        assert rows[-1][0] == "sectPr"
        heading2 = len(texts) - 1 - texts[::-1].index("2. 競合")    # (the TOC lists it first)
        appendix = texts.index("付録D. 図表一覧", heading2)
        assert texts.index("本文1。") < texts.index("図1　市場規模の推移（億円）") \
            < heading2 < texts.index("図2　主要3社の売上高（億円）") \
            < texts.index("図3　地域別の構成比（%）") < texts.index("主要企業のシェア") < appendix
        assert "図4　どの章にも属さない図" in texts          # orphans continue the numbering
        manifest = build_semantic_manifest({k: c.content for k, c in chapters.items()},
                                           figure_collection=coll)
        res = check_artifact(manifest, path)
        assert res["status"] == "pass", res


# --------------------------------------------------------------------------
# V1 PDF / HTML
# --------------------------------------------------------------------------

class TestV1PdfHtml:

    def _gen(self, tmp_path, coll):
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker, make_plan, make_session)
        ev = [("https://ex.com/a", "x " * 30)]
        plan = make_plan([("1", "市場規模の推移", "d"), ("2", "競合", "d")])
        session = make_session(plan, {"1": {"title": "市場規模の推移", "content": "本文 [1]", "evidence": ev},
                                      "2": {"title": "競合", "content": "本文2 [1]", "evidence": ev}})
        gen = ReportGenerator(output_dir=tmp_path / "reports", language="ja")
        gen.figure_collection = coll
        return gen, session, make_locker(tmp_path, ev)

    def test_pdf_numbers_titles_and_caption_blocks(self, tmp_path):
        pypdf = pytest.importorskip("pypdf")
        pytest.importorskip("reportlab")
        from deep_research_tool.config import ReportFormat
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        gen, session, locker = self._gen(tmp_path, coll)
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.PDF, verification_result=None)
        text = "\n".join(p.extract_text() for p in pypdf.PdfReader(str(path)).pages)
        flat = text.replace(" ", "").replace("　", "")
        assert flat.index("図1市場規模の推移") < flat.index("見るべき点:3年で1.5倍") < flat.index("2.競合")
        assert flat.index("2.競合") < flat.index("図2主要3社") < flat.index("図3地域別") < flat.index("図4どの章")
        assert flat.count("見るべき点:3年で1.5倍") == 1
        assert gen.figures_inserted == 5                   # 4 charts + 1 table
        manifest = build_semantic_manifest({"1": "本文 [1]", "2": "本文2 [1]"}, figure_collection=coll)
        assert check_artifact(manifest, path, locker)["status"] == "pass"

    def test_html_numbers_titles_and_caption_blocks(self, tmp_path):
        from deep_research_tool.config import ReportFormat
        png = _png(tmp_path / "c.png")
        coll = _collection(png)
        gen, session, locker = self._gen(tmp_path, coll)
        path = gen.generate_report(session=session, evidence_locker=locker,
                                   format=ReportFormat.HTML, verification_result=None)
        page = path.read_text(encoding="utf-8")
        first = page.index('<div class="figure-title">図1　市場規模の推移（億円）</div><img')
        assert page.index("<h2>1. 市場規模の推移</h2>") < first < page.index("<h2>2. 競合</h2>")
        assert "<figcaption>出所 2 件（公表）<br>見るべき点: 3年で1.5倍に拡大した。</figcaption>" in page
        assert page.count("見るべき点: 3年で1.5倍に拡大した。") == 1
        assert page.index("図2　主要3社") < page.index("図3　地域別") < page.index("図4　どの章")
        assert "break-inside: avoid" in page
        manifest = build_semantic_manifest({"1": "本文 [1]", "2": "本文2 [1]"}, figure_collection=coll)
        assert check_artifact(manifest, path, locker)["status"] == "pass"

    def test_numbering_restarts_for_every_file(self, tmp_path):
        from deep_research_tool.config import ReportFormat
        png = _png(tmp_path / "c.png")
        gen, session, locker = self._gen(tmp_path, _collection(png))
        for _ in range(2):
            page = gen.generate_report(session=session, evidence_locker=locker,
                                       format=ReportFormat.HTML,
                                       verification_result=None).read_text(encoding="utf-8")
            assert "図1　市場規模" in page and "図5" not in page


# --------------------------------------------------------------------------
# Appendix D (item 14)
# --------------------------------------------------------------------------

class TestAppendixD:

    def _coll(self):
        bars = {"form": "bar", "title": "出荷（百万缶）", "unit": "百万缶",
                "labels": ["25Q1", "25Q2", "26Q2"],
                "series": {"出荷": [202, 190, None], "実売ベース推定": [180, None, 150]},
                "caption": "出所 2 件", "judgment": "回復基調。"}
        xy = {"form": "line_xy", "title": "市場予測", "unit": None,
              "labels": ["2023", "2024", "2025"],
              "series": {"A": ["2023.0:50.0", "2025.0:70.0"], "B": ["2024.0:None", "2025.0:65.5"]},
              "judgment": None}
        return FigureTableCollection(charts=[
            _chart("s1_bar", "出荷（百万缶）", "出所 2 件　見るべき点: 回復基調。", "1", None,
                   alt_text="出荷（百万缶）: 回復基調。", chart_data=bars),
            _chart("s1_xy", "市場予測", "推計の比較。", "1", None, alt_text="市場予測: ", chart_data=xy)])

    def test_no_none_judgment_once_line_xy_as_label_value(self):
        coll = self._coll()
        md = figure_semantics_markdown(coll, language="ja", section_id="付録D")
        assert "None" not in md and "=None" not in md
        assert md.count("回復基調。") == 1                     # caption only: no alt line, no 2nd judgment
        assert "- 出荷: 25Q1=202; 25Q2=190\n" in md           # missing 26Q2 skipped
        assert "- 実売ベース推定: 25Q1=180; 26Q2=150" in md
        assert "- A: 2023=50.0; 2025=70.0" in md              # x is the label, not repeated
        assert "- B: 2025=65.5" in md
        assert "2023=2023" not in md and "軸・単位: 百万缶" in md
        # one line per series
        assert sum(1 for line in md.splitlines() if line.startswith("- ")) == 4

    def test_judgment_printed_when_the_caption_lacks_it(self):
        coll = FigureTableCollection(charts=[_chart(
            "x", "t", "出所のみ。", "1", None, alt_text="t: 推移を見る。",
            chart_data={"form": "bar", "labels": ["a"], "series": {"s": [1]}, "judgment": "推移を見る。"})])
        md = figure_semantics_markdown(coll)
        assert md.count("推移を見る。") == 1 and "見るべき点: 推移を見る。" in md
        en = figure_semantics_markdown(coll, language="en")
        assert "Key point: 推移を見る。" in en and "- s: a=1" in en

    def test_image_alt_text_kept_when_it_adds_something(self):
        coll = FigureTableCollection(figures=[Figure(
            figure_id="i", figure_type=FigureType.IMAGE, title="工場の写真", caption="出典: A社",
            alt_text="生産ラインの全景", section_id="1")])
        md = figure_semantics_markdown(coll)
        assert "生産ラインの全景" in md and "出典: A社" in md

    def test_the_frozen_semantics_are_untouched(self):
        # only the markdown rendering changed: figure_semantics (the freeze
        # hash input) still carries the gaps exactly as before
        coll = self._coll()
        sem = figure_semantics(coll)
        bar = next(i for i in sem if i["id"] == "s1_bar")
        assert bar["chart_data"]["series"]["出荷"] == ["202", "190", "None"]
        assert bar["caption"] == "出所 2 件　見るべき点: 回復基調。"
        h1 = manifest_hash(build_semantic_manifest({"1": "x"}, coll))
        figure_semantics_markdown(coll)
        assert manifest_hash(build_semantic_manifest({"1": "x"}, coll)) == h1

    def test_appendix_passes_the_artifact_check_in_every_v1_format(self, tmp_path):
        pytest.importorskip("reportlab")
        from deep_research_tool.config import ReportFormat
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker, make_plan, make_session)
        coll = self._coll()
        app = figure_semantics_markdown(coll, language="ja", section_id="付録D")
        ev = [("https://ex.com/a", "x " * 30)]
        for fmt in ("markdown", "docx", "pdf"):
            plan = make_plan([("1", "市場", "d")])
            session = make_session(plan, {"1": {"title": "市場", "content": "本文 [1]", "evidence": ev}})
            session.section_contents["付録D"] = {"title": "付録D", "content": app, "sources": [],
                                                "extracted_content": []}
            gen = ReportGenerator(output_dir=tmp_path / fmt, language="ja")
            locker = make_locker(tmp_path / fmt, ev)
            path = gen.generate_report(session=session, evidence_locker=locker,
                                       format=ReportFormat(fmt), verification_result=None)
            manifest = build_semantic_manifest({"1": "本文 [1]", "付録D": app}, figure_collection=coll)
            res = check_artifact(manifest, path, locker)
            assert res["status"] == "pass", (fmt, res)


# --------------------------------------------------------------------------
# the chart pipeline: caption composition + report-wide cap (items 7, 8)
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, content):
        self.content = content


class _SpecLLM:
    """Fake chart designer: n panels per chapter from the catalog keys."""

    def __init__(self, n_panels=3):
        self.n = n_panels

    def generate(self, prompt, system_prompt=None, **kw):
        cat = json.loads(prompt.split("値は含めない）:\n", 1)[1] if "値は含めない）:\n" in prompt
                         else prompt.split("no values):\n", 1)[1])
        panels = []
        # the same series in the same form twice is a duplicate: vary them
        plan = [(0, "bar"), (1 % len(cat), "bar" if len(cat) > 1 else "hbar"), (0, "hbar")]
        for i in range(self.n):
            idx, form = plan[i % len(plan)]
            it = cat[idx]
            panels.append({"id": f"p{i + 1}", "form": form, "title": f"{it['name']}（{it['unit']}）#{i + 1}",
                           "labels_ref": it["labels_ref"],
                           "series": [{"name": it["name"], "values_ref": it["values_ref"], "role": "emphasis"}],
                           "unit": it["unit"], "caption": f"出所 {it['sources']} 件",
                           "judgment": "推移を見る。"})
        return _Resp(json.dumps(panels, ensure_ascii=False))


def _dp(subject, metric, year, value, unit, url, **kw):
    from deep_research_tool.evidence.numerical_extractor import NumericalDataPoint
    kw.setdefault("raw_text", f"{value}{unit}")
    return NumericalDataPoint(value=value, unit=unit, metric_name=metric, subject=subject, year=year,
                              source_url=url, source_title=url.split("/")[2], **kw)


def _store():
    from deep_research_tool.evidence.numerical_extractor import NumericalDataStore
    store = NumericalDataStore(research_topic="炭素繊維")
    gov = "https://www.stat.go.jp/x"
    for sid in ("1", "2", "3"):
        for y, v in ((2021, 40), (2022, 50), (2023, 55), (2024, 61)):
            store.add(_dp(f"市場{sid}", "市場規模", y, v + int(sid), "億円", gov, section_id=sid))
        for e, v in (("Toray", 300), ("Teijin", 200), ("Mitsubishi", 150)):
            store.add(_dp(e, "売上高", 2024, v + int(sid), "億円", gov, section_id=sid))
    return store


def _tool(tmp_path, language="ja", **over):
    from deep_research_tool.config import create_config
    from deep_research_tool.main import DeepResearchTool
    cfg = create_config(provider="openai", openai_api_key="t", output_dir=str(tmp_path),
                        auto_figures=True, chart_max_per_section=3)
    cfg.research.language = language
    for k, v in over.items():
        setattr(cfg.report, k, v)          # chart_max_total: read with getattr(…, 0)
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config, tool.llm_client, tool.stage_llm_clients = cfg, _SpecLLM(), {}
    return tool


def _session():
    return SimpleNamespace(query="炭素繊維", research_plan=None, section_contents={
        "3": {"title": "需要", "content": "本文", "sources": []},     # out of order on purpose
        "1": {"title": "市場", "content": "本文", "sources": []},
        "2": {"title": "競合", "content": "本文", "sources": []}})


class TestPipeline:

    def test_total_cap_round_robin_and_recorded(self, tmp_path):
        pytest.importorskip("matplotlib")
        from deep_research_tool.evidence.locker import EvidenceLocker
        tool = _tool(tmp_path, chart_max_total=4)
        figs = tool._spec_charts(_session(), EvidenceLocker(research_id="r", output_dir=tmp_path / "ev"),
                                 _store(), tmp_path / "figures")
        by = {}
        for f in figs:
            by.setdefault(f.section_id, []).append(f.figure_id)
        # every chapter keeps its first chart before any gets a second;
        # the earliest chapter (in report order, not dict order) wins the 4th
        assert len(figs) == 4
        assert sorted(len(v) for v in by.values()) == [1, 1, 2] and len(by["1"]) == 2
        assert all(ids[0].endswith("_p1") for ids in by.values())
        y = json.loads((tmp_path / "figures" / "chart_yield.json").read_text(encoding="utf-8"))
        cap = y["total_cap"]
        assert cap["max_total"] == 4 and cap["kept"] == 4 and cap["dropped"] == cap["validated"] - 4
        assert cap["dropped"] > 0 and len(cap["dropped_panels"]) == cap["dropped"]
        assert {d["chapter"] for d in cap["dropped_panels"]} <= {"1", "2", "3"}
        assert y["chapters"]["3"]["over_total"] >= 1
        assert any(p["over_total"] for p in y["chapters"]["3"]["panels"])
        assert y["charts"] == 4

    def test_no_cap_when_the_setting_is_absent_or_zero(self, tmp_path):
        pytest.importorskip("matplotlib")
        from deep_research_tool.evidence.locker import EvidenceLocker
        from deep_research_tool.main import DeepResearchTool
        tool = _tool(tmp_path, chart_max_total=0)
        figs = tool._spec_charts(_session(), EvidenceLocker(research_id="r", output_dir=tmp_path / "ev"),
                                 _store(), tmp_path / "figures")
        y = json.loads((tmp_path / "figures" / "chart_yield.json").read_text(encoding="utf-8"))
        assert "total_cap" not in y and len(figs) == sum(c["valid"] for c in y["chapters"].values())
        assert DeepResearchTool._chart_total_cap([("1", "t", [{"id": "a"}], [], None, {})], 0) == {}

    def test_cap_smaller_than_the_chapter_count(self):
        from deep_research_tool.main import DeepResearchTool
        p = lambda s, i: {"id": f"s{s}_p{i}"}                      # noqa: E731
        results = [(s, "t", [p(s, 1), p(s, 2)], [], None, {}) for s in ("10", "2", "1")]
        over = DeepResearchTool._chart_total_cap(results, 2)
        # chapters 1 and 2 keep their first panel; chapter 10 keeps nothing
        assert [q["id"] for q in over["10"]] == ["s10_p1", "s10_p2"]
        assert [q["id"] for q in over["1"]] == ["s1_p2"] and [q["id"] for q in over["2"]] == ["s2_p2"]

    def test_whole_chapter_over_the_cap_is_explained(self, tmp_path):
        pytest.importorskip("matplotlib")
        from deep_research_tool.evidence.locker import EvidenceLocker
        tool = _tool(tmp_path, chart_max_total=2)
        figs = tool._spec_charts(_session(), EvidenceLocker(research_id="r", output_dir=tmp_path / "ev"),
                                 _store(), tmp_path / "figures")
        y = json.loads((tmp_path / "figures" / "chart_yield.json").read_text(encoding="utf-8"))
        assert len(figs) == 2 and {f.section_id for f in figs} == {"1", "2"}
        assert "上限" in y["chapters"]["3"]["no_chart_reason"]

    def test_caption_carries_the_judgment_once_and_no_number(self, tmp_path):
        pytest.importorskip("matplotlib")
        from deep_research_tool.evidence.locker import EvidenceLocker
        figs = _tool(tmp_path)._spec_charts(
            _session(), EvidenceLocker(research_id="r", output_dir=tmp_path / "ev"), _store(),
            tmp_path / "figures")
        assert figs
        for f in figs:
            # once; the improved style (the default) appends a data fact
            assert f.caption.count("見るべき点: ") == 1 and "見るべき点: 推移を見る。" in f.caption
            assert not f.caption.startswith("図") and not f.title.startswith("図")
            assert figure_caption_lines(f)[-1].startswith("見るべき点: 推移を見る。")
        en = _tool(tmp_path / "en", language="en")._spec_charts(
            _session(), EvidenceLocker(research_id="r", output_dir=tmp_path / "ev2"), _store(),
            tmp_path / "en" / "figures")
        assert en and all("Key point: 推移を見る。" in f.caption for f in en)


# --------------------------------------------------------------------------
# V2 + PDF (item 15)
# --------------------------------------------------------------------------

V2_MD = """# 炭素繊維市場の調査

*Generated: 2026-11-11*

## 目次

- 1. 市場規模

## 1. 市場規模

国内市場は **拡大** している [1]。2024年は60億円に達した。

| 用途 | シェア |
|---|---|
| 航空 | 40% |

## 参考文献

1. [A] 統計 https://www.stat.go.jp/x
"""


class TestV2Pdf:

    def _gen(self):
        from unittest.mock import MagicMock
        from deep_research_tool.report.v2.generator import ReportGeneratorV2
        return ReportGeneratorV2(llm_client=MagicMock(), language="ja")

    def _warnings(self):
        from deep_research_tool.utils.helpers import ResearchWarnings
        return ResearchWarnings.get_instance().to_dict_list()

    def test_real_pdf_with_japanese_text_and_figures(self, tmp_path):
        pypdf = pytest.importorskip("pypdf")
        pytest.importorskip("reportlab")
        from deep_research_tool.utils.helpers import ResearchWarnings
        png = _png(tmp_path / "c.png")
        coll = FigureTableCollection(charts=[_chart("c1", "市場規模の推移（億円）",
                                                    "出所 2 件　見るべき点: 拡大した。", "1", png)])
        md = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_markdown(V2_MD, coll)
        ResearchWarnings.reset()
        gen = self._gen()
        path = gen.save_report(md, tmp_path / "out", "r", format="pdf")
        assert path.suffix == ".pdf" and path.read_bytes()[:5] == b"%PDF-"
        reader = pypdf.PdfReader(str(path))
        assert sum(len(p.images) for p in reader.pages) == 1 and gen.pdf_images_embedded == 1
        text = "".join(p.extract_text() for p in reader.pages).replace(" ", "")
        assert "国内市場は拡大している" in text and "図1" in text and "見るべき点:拡大した。" in text
        assert text.index("図1") < text.index("見るべき点") < text.index("参考文献")
        assert not any(w["severity"] == "CRITICAL" for w in self._warnings())
        manifest = build_semantic_manifest(
            {"1": "## 1. 市場規模\n\n国内市場は **拡大** している [1]。2024年は60億円に達した。\n\n"
                  "| 用途 | シェア |\n|---|---|\n| 航空 | 40% |"}, figure_collection=coll)
        assert check_artifact(manifest, path, extra_allowed=["炭素繊維市場の調査"])["status"] == "pass"

    def test_failure_falls_back_to_markdown_with_a_critical_warning(self, tmp_path, monkeypatch):
        from deep_research_tool.utils.helpers import ResearchWarnings
        from deep_research_tool.report.v2.generator import ReportGeneratorV2

        def boom(self, md, path):
            path.write_bytes(b"%PDF-partial")
            raise RuntimeError("layout exploded")
        monkeypatch.setattr(ReportGeneratorV2, "_render_markdown_pdf", boom)
        ResearchWarnings.reset()
        path = self._gen().save_report(V2_MD, tmp_path, "r", format="pdf")
        assert path.suffix == ".md" and path.read_text(encoding="utf-8") == V2_MD
        assert not (tmp_path / "r.pdf").exists()                 # no partial PDF left
        crit = [w for w in self._warnings() if w["severity"] == "CRITICAL"]
        assert len(crit) == 1 and "layout exploded" in crit[0]["message"]

    def test_strict_format_raises(self, tmp_path, monkeypatch):
        from deep_research_tool.report.v2.generator import ReportFormatError, ReportGeneratorV2
        monkeypatch.setattr(ReportGeneratorV2, "_render_markdown_pdf",
                            lambda self, md, path: (_ for _ in ()).throw(RuntimeError("nope")))
        with pytest.raises(ReportFormatError) as e:
            self._gen().save_report(V2_MD, tmp_path, "r", format="pdf", strict_format=True)
        assert (tmp_path / "r_debug.md").exists() and "nope" in str(e.value)

    def test_layout_failure_is_retried_without_images(self, tmp_path, monkeypatch):
        from deep_research_tool.utils.helpers import ResearchWarnings
        from deep_research_tool.report.v2.generator import ReportGeneratorV2
        real = ReportGeneratorV2._render_markdown_pdf
        modes = []

        def flaky(self, md, path):
            modes.append(self._pdf_safe_mode)
            if not self._pdf_safe_mode:
                raise ValueError("too large on page")
            return real(self, md, path)
        monkeypatch.setattr(ReportGeneratorV2, "_render_markdown_pdf", flaky)
        png = _png(tmp_path / "c.png")
        md = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_markdown(
            V2_MD, FigureTableCollection(charts=[_chart("c1", "図題", "説明。", "1", png)]))
        ResearchWarnings.reset()
        gen = self._gen()
        path = gen.save_report(md, tmp_path, "r", format="pdf")
        assert modes == [False, True] and path.suffix == ".pdf"
        assert gen.pdf_images_embedded == 0 and "too large" in gen.pdf_figures_dropped
        assert not any(w["severity"] == "CRITICAL" for w in self._warnings())


# --------------------------------------------------------------------------
# run(): V2 + PDF and V3 carry the numbered figures end to end
# --------------------------------------------------------------------------

def _run(tmp_path, version, fmt):
    from unittest.mock import patch
    from deep_research_tool.tests import test_finalization_integration as T
    from deep_research_tool.utils.helpers import ResearchWarnings
    ResearchWarnings.reset()
    png = _png(tmp_path / "figs" / "chart.png")
    coll = FigureTableCollection(
        charts=[_chart("c1", "シェアの推移", "A社のシェアの推移を示す図。出所: 統計　見るべき点: 首位を保った。",
                       "1", png)])
    llm = T.make_llm()
    if version != "v1":
        llm.on("chapter", lambda p: T.long_body("1", "市場シェア", T.CLAIM_SENTENCE)
               + "\n===CHAPTER_META===\n" + T.j({"key_points": [], "terms_used": [], "facts_stated": []}))
    locker = T.make_locker(tmp_path, T.SUPPORTED_EVIDENCE)
    tool = T.build_tool(tmp_path, llm, T.FakeSearch(), auto_figures=True, output_format=fmt,
                        report_generator_version=version)
    gen = FigureTableGenerator(llm_client=llm, output_dir=tmp_path / "figs")
    base = T.Base()
    base.tmp = tmp_path
    with patch.object(type(tool), "_generate_figure_collection", lambda self, **kw: (coll, gen)):
        return T.run_tool(tool, base.default_session(), locker)


def test_run_v2_pdf_is_a_real_pdf_with_the_numbered_chart(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    res = _run(tmp_path, "v2", "pdf")
    path = Path(res["report_path"])
    assert path.suffix == ".pdf", path
    reader = pypdf.PdfReader(str(path))
    assert sum(len(p.images) for p in reader.pages) == 1
    text = "".join(p.extract_text() for p in reader.pages).replace(" ", "")
    assert "図1" in text and "見るべき点:首位を保った。" in text
    assert res["semantic_artifact_check"] == "pass"
    assert res["figures"]["placed"] == 1 and res["figures"]["total"] == 1
    assert not any("PDF format requested" in str(w.get("message", "")) for w in res.get("warnings") or [])


def test_run_v3_docx_numbers_the_chart(tmp_path):
    pytest.importorskip("docx")
    res = _run(tmp_path, "v3", "docx")
    path = Path(res["report_path"])
    texts = [r[1] for r in _body_rows(path)]
    assert "図1　シェアの推移" in texts
    i = texts.index("図1　シェアの推移")
    assert texts[i + 2] == "A社のシェアの推移を示す図。出所: 統計\n見るべき点: 首位を保った。"
    assert res["semantic_artifact_check"] == "pass"


class TestReviewFixes:
    """Fixes from the independent review of v1.13.0."""

    def test_last_chapter_figures_come_before_the_limitations_note(self, tmp_path):
        png = _png(tmp_path / "c.png")
        md = ("## 1. 市場\n\n本文1。\n\n## 2. 需要\n\n本文2。\n\n### 調査上の限界\n\n- 未解決: 何か\n\n"
              "## References\n\n1. x\n")
        coll = FigureTableCollection(charts=[_chart("c1", "需要の推移", "説明。", "2", png)])
        out = FigureTableGenerator(llm_client=None, output_dir=tmp_path).add_figures_to_markdown(md, coll)
        assert out.index("本文2。") < out.index("**図1　需要の推移**") < out.index("### 調査上の限界")

    def test_a_wrapped_figure_title_is_not_an_unverified_addition(self):
        from deep_research_tool.report.semantic_manifest import verify_frozen_in_artifact
        title = "国内炭素繊維市場における用途別需要量の推移と2030年までの見通し（航空・自動車、千トン）"
        body = "本文の一文目である。" * 8
        manifest = build_semantic_manifest({"1": body}, FigureTableCollection(charts=[
            _chart("c1", title, "出所 3 件", "1", Path("x.png"))]))
        # the PDF wraps '図3 <title>' over two lines
        art = f"{body}\n図3 {title[:42]}\n{title[42:]}\n出所 3 件\n"
        res = verify_frozen_in_artifact(manifest, art)
        assert not res.get("additions"), res.get("additions")
