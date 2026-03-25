"""
html_report.py — Interactive HTML experiment report.

Generates a self-contained HTML file with:
  - Metadata header (timestamp, mode, models)
  - Per-model statistics table with confidence intervals
  - Colour-coded variance heatmap (matches PNG)
  - Semantic variance comparison table
  - Top-10 most volatile prompts (cross-model)
  - Cross-model divergence table

No external dependencies — pure Python stdlib.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Colour helpers (YlOrRd palette approximation)
# ---------------------------------------------------------------------------

def _yor_css(v: float) -> str:
    """Return a CSS rgb() colour on the YlOrRd gradient for v ∈ [0, 1]."""
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        t = v * 2
        r = int(255)
        g = int(255 - t * (255 - 166))
        b = int(204 - t * 204)
    else:
        t = (v - 0.5) * 2
        r = int(255 - t * (255 - 128))
        g = int(166 - t * 166)
        b = 0
    return f"rgb({r},{g},{b})"


def _text_color(v: float) -> str:
    return "#fff" if v > 0.55 else "#222"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#f5f5f5;color:#222;}
h1{font-size:1.6rem;margin-bottom:4px;}
h2{font-size:1.2rem;margin-top:28px;border-bottom:2px solid #ccc;padding-bottom:4px;}
.meta{font-size:.85rem;color:#555;margin-bottom:20px;}
table{border-collapse:collapse;width:100%;margin-bottom:16px;font-size:.82rem;}
th{background:#333;color:#fff;padding:6px 8px;text-align:left;cursor:pointer;user-select:none;}
th:hover{background:#555;}
td{padding:4px 7px;border:1px solid #ddd;background:#fff;}
tr:hover td{filter:brightness(.95);}
.heatmap-table td{padding:1px;min-width:34px;text-align:center;font-size:.68rem;border:none;}
.heatmap-table th{font-size:.72rem;padding:3px 4px;}
.verdict-vc{color:#1a7c2e;font-weight:600;}
.verdict-c{color:#2c7fbe;font-weight:600;}
.verdict-mv{color:#d97b00;font-weight:600;}
.verdict-hv{color:#c0392b;font-weight:600;}
.footer{margin-top:40px;font-size:.78rem;color:#888;text-align:center;}
.tag{display:inline-block;background:#e0e7ff;color:#3730a3;border-radius:4px;
     padding:1px 6px;font-size:.75rem;margin-right:4px;}
"""

_SORT_JS = """
function sortTable(tableId,col){
  var t=document.getElementById(tableId),dir=t._dir||{};
  var asc=dir[col]!==true;dir[col]=asc;t._dir=dir;
  var rows=Array.from(t.tBodies[0].rows);
  rows.sort(function(a,b){
    var av=a.cells[col].dataset.v||a.cells[col].innerText;
    var bv=b.cells[col].dataset.v||b.cells[col].innerText;
    var an=parseFloat(av),bn=parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn))return asc?an-bn:bn-an;
    return asc?av.localeCompare(bv):bv.localeCompare(av);
  });
  rows.forEach(function(r){t.tBodies[0].appendChild(r);});
}
"""


def _verdict_class(mean_v: float) -> str:
    if mean_v < 0.1:
        return "verdict-vc"
    if mean_v < 0.2:
        return "verdict-c"
    if mean_v < 0.35:
        return "verdict-mv"
    return "verdict-hv"


def _verdict_text(mean_v: float) -> str:
    if mean_v < 0.1:
        return "Very consistent"
    if mean_v < 0.2:
        return "Consistent"
    if mean_v < 0.35:
        return "Moderate variance"
    return "High variance"


def generate_html_report(
    variance_data: Dict[str, Dict[str, float]],
    prompts: List[str],
    output_path: Path,
    *,
    mode: str = "mock",
    num_runs: int = 10,
    models: Optional[Dict[str, str]] = None,
    semantic_data: Optional[Dict[str, Dict[str, float]]] = None,
    model_stats: Optional[Dict[str, dict]] = None,
    cross_model_divergence: Optional[Dict[str, float]] = None,
    title: str = "Consistency Heatmap — Experiment Report",
) -> Path:
    """
    Generate a self-contained HTML report and write it to output_path.

    Returns the path to the written file.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_names = sorted(variance_data.keys())
    num_prompts = len(prompts)

    lines: List[str] = []
    a = lines.append  # shorthand

    a("<!DOCTYPE html>")
    a("<html lang='en'><head>")
    a(f"<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    a(f"<title>{title}</title>")
    a(f"<style>{_CSS}</style>")
    a("</head><body>")
    a(f"<h1>{title}</h1>")

    # --- Metadata badges ---
    mode_tag = "LIVE" if mode == "live" else "MOCK"
    a(f"<div class='meta'>")
    a(f"  <span class='tag'>{mode_tag} mode</span>")
    a(f"  <span class='tag'>{num_prompts} prompts</span>")
    a(f"  <span class='tag'>{num_runs} runs</span>")
    a(f"  <span class='tag'>{len(model_names)} models</span>")
    a(f"  Generated {ts}")
    a(f"</div>")

    if models:
        a("<p style='font-size:.82rem;color:#444;'>")
        for name, mid in sorted(models.items()):
            a(f"  <strong>{name}</strong>: <code>{mid}</code>&nbsp;&nbsp;")
        a("</p>")

    # --- Per-model stats table ---
    a("<h2>Per-Model Statistics</h2>")
    a("<table id='stats-table'>")
    a("<thead><tr>")
    for i, h in enumerate(["Model", "Mean ▲", "Std Dev", "Min", "Max", "Median",
                            "95% CI Low", "95% CI High", "Verdict"]):
        a(f"<th onclick='sortTable(\"stats-table\",{i})'>{h}</th>")
    a("</tr></thead><tbody>")

    for mname in model_names:
        scores = list(variance_data[mname].values())
        if not scores:
            continue
        mean_v = sum(scores) / len(scores)

        # Use pre-computed stats if available, otherwise compute inline
        if model_stats and mname in model_stats:
            st = model_stats[mname]
        else:
            import math
            n = len(scores)
            std = math.sqrt(sum((x - mean_v) ** 2 for x in scores) / max(n - 1, 1)) if n > 1 else 0.0
            sorted_s = sorted(scores)
            mid = n // 2
            median = sorted_s[mid] if n % 2 == 1 else (sorted_s[mid - 1] + sorted_s[mid]) / 2.0
            se = std / math.sqrt(n) if n > 1 else 0.0
            st = {
                "mean": mean_v, "std": std, "min": min(scores), "max": max(scores),
                "median": median,
                "ci_95_low": max(0.0, mean_v - 1.96 * se),
                "ci_95_high": min(1.0, mean_v + 1.96 * se),
            }

        vc = _verdict_class(float(st["mean"]))
        vt = _verdict_text(float(st["mean"]))
        bg = _yor_css(float(st["mean"]))

        a("<tr>")
        a(f"<td><strong>{mname}</strong></td>")
        a(f"<td data-v='{st['mean']:.6f}' style='background:{bg};color:{_text_color(float(st['mean']))}'>{st['mean']:.4f}</td>")
        a(f"<td data-v='{st['std']:.6f}'>{st['std']:.4f}</td>")
        a(f"<td data-v='{st['min']:.6f}'>{st['min']:.4f}</td>")
        a(f"<td data-v='{st['max']:.6f}'>{st['max']:.4f}</td>")
        a(f"<td data-v='{st['median']:.6f}'>{st['median']:.4f}</td>")
        a(f"<td data-v='{st['ci_95_low']:.6f}'>{st['ci_95_low']:.4f}</td>")
        a(f"<td data-v='{st['ci_95_high']:.6f}'>{st['ci_95_high']:.4f}</td>")
        a(f"<td class='{vc}'>{vt}</td>")
        a("</tr>")
    a("</tbody></table>")

    # --- Variance heatmap table ---
    a("<h2>Levenshtein Variance Heatmap</h2>")
    a("<div style='overflow-x:auto;'>")
    a("<table class='heatmap-table'>")
    a("<thead><tr><th>Model \\ Prompt</th>")
    for i in range(num_prompts):
        prompt_title = prompts[i] if i < len(prompts) else ""
        a(f"<th title='{prompt_title}'>{i+1}</th>")
    a("<th>Mean</th></tr></thead><tbody>")

    for mname in model_names:
        scores = variance_data[mname]
        vals = [float(scores.get(str(i), 0.0)) for i in range(num_prompts)]
        mean_v = sum(vals) / len(vals) if vals else 0.0
        a(f"<tr><td style='font-weight:600;white-space:nowrap;'>{mname}</td>")
        for v in vals:
            bg = _yor_css(v)
            fg = _text_color(v)
            a(f"<td style='background:{bg};color:{fg}'>{v:.2f}</td>")
        mbg = _yor_css(mean_v)
        mfg = _text_color(mean_v)
        a(f"<td style='background:{mbg};color:{mfg};font-weight:700'>{mean_v:.3f}</td>")
        a("</tr>")
    a("</tbody></table></div>")

    # --- Semantic variance table (optional) ---
    if semantic_data:
        a("<h2>Semantic (Jaccard) Variance Heatmap</h2>")
        a("<p style='font-size:.82rem;color:#555;'>Vocabulary-overlap distance — complements character-level Levenshtein.</p>")
        a("<div style='overflow-x:auto;'>")
        a("<table class='heatmap-table'>")
        a("<thead><tr><th>Model \\ Prompt</th>")
        for i in range(num_prompts):
            prompt_title = prompts[i] if i < len(prompts) else ""
            a(f"<th title='{prompt_title}'>{i+1}</th>")
        a("<th>Mean</th></tr></thead><tbody>")

        for mname in model_names:
            scores = semantic_data.get(mname, {})
            vals = [float(scores.get(str(i), 0.0)) for i in range(num_prompts)]
            mean_v = sum(vals) / len(vals) if vals else 0.0
            a(f"<tr><td style='font-weight:600;white-space:nowrap;'>{mname}</td>")
            for v in vals:
                bg = _yor_css(v)
                fg = _text_color(v)
                a(f"<td style='background:{bg};color:{fg}'>{v:.2f}</td>")
            mbg = _yor_css(mean_v)
            mfg = _text_color(mean_v)
            a(f"<td style='background:{mbg};color:{mfg};font-weight:700'>{mean_v:.3f}</td>")
            a("</tr>")
        a("</tbody></table></div>")

    # --- Top volatile prompts ---
    a("<h2>Most Volatile Prompts (top 10, averaged across models)</h2>")
    prompt_scores: Dict[int, List[float]] = {}
    for mname, scores in variance_data.items():
        for p_str, v in scores.items():
            prompt_scores.setdefault(int(p_str), []).append(float(v))
    avg_scores = {idx: sum(vs) / len(vs) for idx, vs in prompt_scores.items()}
    top10 = sorted(avg_scores.items(), key=lambda x: -x[1])[:10]

    a("<table id='volatile-table'>")
    a("<thead><tr>")
    for i, h in enumerate(["Rank", "Prompt #", "Prompt Text", "Mean Variance (cross-model)"]):
        a(f"<th onclick='sortTable(\"volatile-table\",{i})'>{h}</th>")
    a("</tr></thead><tbody>")
    for rank, (p_idx, avg_v) in enumerate(top10, 1):
        pt = prompts[p_idx] if p_idx < len(prompts) else f"Prompt {p_idx}"
        bg = _yor_css(avg_v)
        fg = _text_color(avg_v)
        a("<tr>")
        a(f"<td>{rank}</td>")
        a(f"<td>{p_idx + 1}</td>")
        a(f"<td style='max-width:480px'>{pt}</td>")
        a(f"<td data-v='{avg_v:.6f}' style='background:{bg};color:{fg}'>{avg_v:.4f}</td>")
        a("</tr>")
    a("</tbody></table>")

    # --- Cross-model divergence ---
    if cross_model_divergence:
        a("<h2>Cross-Model Divergence</h2>")
        a("<p style='font-size:.82rem;color:#555;'>How much models disagree with each other per prompt (semantic distance between first runs).</p>")
        sorted_div = sorted(cross_model_divergence.items(), key=lambda x: -float(x[1]))[:15]
        a("<table id='div-table'>")
        a("<thead><tr>")
        for i, h in enumerate(["Prompt #", "Prompt Text", "Cross-Model Divergence"]):
            a(f"<th onclick='sortTable(\"div-table\",{i})'>{h}</th>")
        a("</tr></thead><tbody>")
        for p_str, div_v in sorted_div:
            p_idx = int(p_str)
            pt = prompts[p_idx] if p_idx < len(prompts) else f"Prompt {p_idx}"
            bg = _yor_css(float(div_v))
            fg = _text_color(float(div_v))
            a("<tr>")
            a(f"<td>{p_idx + 1}</td>")
            a(f"<td style='max-width:480px'>{pt}</td>")
            a(f"<td data-v='{float(div_v):.6f}' style='background:{bg};color:{fg}'>{float(div_v):.4f}</td>")
            a("</tr>")
        a("</tbody></table>")

    a(f"<div class='footer'>Generated by Consistency Heatmap / NEO &mdash; {ts}</div>")
    a(f"<script>{_SORT_JS}</script>")
    a("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
