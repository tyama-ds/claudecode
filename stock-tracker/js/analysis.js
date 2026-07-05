/* ==========================================================================
   analysis.js — 簡易財務分析 & アドバイス生成
   ========================================================================== */

const Analysis = (() => {

  // 値を0〜5のスコアに変換(lo以下=0, hi以上=5)。invert=trueで低いほど良い指標。
  function scale(v, lo, hi, invert = false) {
    if (v == null) return null;
    let s = ((v - lo) / (hi - lo)) * 5;
    s = Math.max(0, Math.min(5, s));
    return invert ? 5 - s : s;
  }

  // 銘柄タイプ別のスコアリング。ETFは対象外(null)。
  function computeScores(sym) {
    const f = sym.financials;
    if (!f || sym.type === "etf") return null;
    const isUS = sym.market === "US";

    // 割安性: PER(市場平均を跨ぐレンジで低いほど高スコア) + PBR
    const perScore = scale(f.per, isUS ? 10 : 5, isUS ? 45 : 30, true);
    const pbrScore = scale(f.pbr, 0.5, isUS ? 12 : 4, true);
    const value = avg([perScore, pbrScore]);

    // 収益性: ROE
    const profit = scale(f.roe, 2, isUS ? 40 : 18);

    // 健全性: 自己資本比率(銀行等は除外表示)
    const stability = f.equityRatio != null ? scale(f.equityRatio, 10, 60) : null;

    // 株主還元: 配当利回り
    const dividend = scale(f.dividendYield, 0, 4.5);

    // モメンタム: 直近3ヶ月の株価トレンド
    const momentum = computeMomentumScore(sym.prices);

    return [
      { key: "value", label: "割安性", score: nz(value) },
      { key: "profit", label: "収益性", score: nz(profit) },
      { key: "stability", label: "健全性", score: nz(stability, 2.5) },
      { key: "dividend", label: "配当", score: nz(dividend) },
      { key: "momentum", label: "勢い", score: nz(momentum) },
    ];
  }

  function computeMomentumScore(prices) {
    if (!prices || prices.length < 70) return 2.5;
    const last = prices[prices.length - 1].close;
    const m3 = prices[prices.length - 63].close; // 約3ヶ月前
    const chg = ((last - m3) / m3) * 100;
    return scale(chg, -20, 20);
  }

  function avg(arr) {
    const v = arr.filter(x => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }
  function nz(v, def = 0) { return v == null ? def : v; }

  /* ---------- アドバイス文生成 ---------- */
  function buildAdvice(sym) {
    const f = sym.financials;
    if (sym.type === "etf") return buildEtfAdvice(sym);
    if (!f) return { verdict: "データ不足", icon: "❓", sections: [{ h: "見解", p: "財務データが不足しているため分析できません。" }] };

    const scores = computeScores(sym);
    const total = avg(scores.map(s => s.score));
    const S = Object.fromEntries(scores.map(s => [s.key, s.score]));
    const isUS = sym.market === "US";
    const sections = [];

    // バリュエーション
    let vp = "";
    if (f.per != null) {
      const perLv = f.per < (isUS ? 18 : 12) ? "低め(割安圏)" : f.per < (isUS ? 30 : 20) ? "標準的な水準" : "高め(成長期待の織り込みが大きい水準)";
      vp += `PER ${f.per}倍は${isUS ? "米国" : "日本"}市場では${perLv}です。`;
    }
    if (f.pbr != null) {
      vp += f.pbr < 1 ? `PBR ${f.pbr}倍と解散価値を下回っており、資産面からは割安感があります。`
        : `PBRは${f.pbr}倍です。`;
    }
    if (vp) sections.push({ h: "🏷️ バリュエーション(割安性)", p: vp });

    // 収益性
    if (f.roe != null) {
      const roeLv = f.roe >= (isUS ? 25 : 12) ? "高く、資本効率に優れています" : f.roe >= 8 ? "まずまずの水準です" : "低めで、資本効率に課題があります";
      sections.push({ h: "💰 収益性", p: `ROE(自己資本利益率)は ${f.roe}% と${roeLv}。` + (f.operatingIncome != null ? `直近通期の営業利益は ${fmtMoney(f.operatingIncome, sym)} です。` : "") });
    }

    // 財務健全性
    if (f.equityRatio != null) {
      const eqLv = f.equityRatio >= 50 ? "非常に健全" : f.equityRatio >= 30 ? "おおむね健全" : "やや低め(金融業では標準的な場合もあります)";
      sections.push({ h: "🏦 財務健全性", p: `自己資本比率は ${f.equityRatio}% で${eqLv}です。` });
    }

    // 配当
    if (f.dividendYield != null) {
      const dLv = f.dividendYield >= 3 ? "高配当の部類に入ります" : f.dividendYield >= 1.5 ? "平均的な水準です" : "低め(成長投資優先の企業に多い水準)です";
      sections.push({ h: "🎁 株主還元", p: `配当利回りは ${f.dividendYield}% と${dLv}。` });
    }

    // 直近の勢い + 注目イベント
    const mom = S.momentum;
    const momTxt = mom >= 3.5 ? "直近3ヶ月の株価は上昇基調にあります。" : mom <= 1.5 ? "直近3ヶ月の株価は下落基調で、逆風が意識されます。" : "直近3ヶ月の株価はもみ合いです。";
    const recent = (sym.events || []).slice(-2).map(e => e.title).join("、");
    sections.push({ h: "📈 直近の動き", p: momTxt + (recent ? ` 注目材料として「${recent}」などが挙げられます。` : "") });

    // 総合見解
    let verdict, icon;
    if (total >= 3.5) { verdict = "総合評価: 良好"; icon = "🟢"; }
    else if (total >= 2.4) { verdict = "総合評価: 中立"; icon = "🟡"; }
    else { verdict = "総合評価: 慎重"; icon = "🔴"; }

    const strengths = scores.filter(s => s.score >= 3.5).map(s => s.label);
    const weaknesses = scores.filter(s => s.score < 2).map(s => s.label);
    let summary = `5項目スコアの平均は ${total.toFixed(1)} / 5.0。`;
    if (strengths.length) summary += `強みは「${strengths.join("・")}」、`;
    if (weaknesses.length) summary += `弱みは「${weaknesses.join("・")}」です。`;
    else if (strengths.length) summary += `目立った弱みは見当たりません。`;
    sections.push({ h: "📝 総合見解", p: summary });

    return { verdict, icon, sections, scores, total };
  }

  function buildEtfAdvice(sym) {
    const f = sym.financials || {};
    const sections = [];
    sections.push({
      h: "📦 ETFの特徴",
      p: `${sym.name}は${sym.index || "指数"}に連動するETFです。個別銘柄リスクを避けながら市場全体へ分散投資できます。`
    });
    if (f.expenseRatio != null) {
      sections.push({ h: "💸 コスト", p: `信託報酬(経費率)は年 ${f.expenseRatio}% ${f.expenseRatio <= 0.2 ? "と低コストで、長期保有に向いています" : "です"}。` });
    }
    if (f.distributionYield != null) {
      sections.push({ h: "🎁 分配金", p: `直近の分配金利回りは約 ${f.distributionYield}% です。` });
    }
    const mom = computeMomentumScore(sym.prices);
    sections.push({ h: "📈 直近の動き", p: mom >= 3.5 ? "連動指数は直近3ヶ月で上昇基調です。" : mom <= 1.5 ? "連動指数は直近3ヶ月で調整局面にあります。" : "連動指数は直近3ヶ月もみ合いです。" });
    return { verdict: "インデックス連動型", icon: "📦", sections, scores: null };
  }

  function fmtMoney(v, sym) {
    if (sym.market === "US") return `$${v}B(約${Math.round(v * 10) / 10}十億ドル)`;
    return `${v.toLocaleString()}億円`;
  }

  return { computeScores, buildAdvice };
})();
