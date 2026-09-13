"""
build_site.py -- assemble docs/index.html, fully self-contained.

Embeds the panel-derived interactive data and the figures as base64 so the
output is a single file that can be dropped on any host with no build step,
no asset directory and no CDN dependency except the two webfonts.

    python scripts/build_site.py
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
from aei_audit import indices as ix  # noqa: E402

MAPS = [("State", None), ("Census region", "census_region"), ("Time zone", "time_zone"),
        ("BEA region", "bea_region"), ("Census division", "census_division"),
        ("Federal region", "federal_region")]

FIGS = [("scale", "F1_scale_decomposition.png"), ("null", "F3_zoning_null.png"),
        ("shift", "F5_shiftshare.png"), ("cater", "F5_state_caterpillar.png"),
        ("brk", "F6_platform_break.png")]


def interactive_data(p):
    D = {}
    for w, g in p.groupby("wave"):
        D[w] = {}
        for nm, col in MAPS:
            a = (g[["state", "usage_share", "pop_share_fixed"]].rename(columns={"state": "unit"})
                 if col is None else
                 g.groupby(col)[["usage_share", "pop_share_fixed"]].sum().reset_index()
                  .rename(columns={col: "unit"}))
            a["aui"] = a.usage_share / a.pop_share_fixed
            a = a.sort_values("aui", ascending=False)
            cx, cy = ix.lorenz_points(a.aui.values, a.pop_share_fixed.values)
            D[w][nm] = {
                "units": a.unit.tolist(),
                "aui": [round(v, 4) for v in a.aui],
                "pop": [round(v, 4) for v in a.pop_share_fixed],
                "gini": round(ix.gini(a.aui.values, a.pop_share_fixed.values), 4),
                "theil": round(ix.theil(a.aui.values, a.pop_share_fixed.values), 4),
                "k": len(a),
                "lx": [round(v, 4) for v in cx], "ly": [round(v, 4) for v in cy],
            }
    return D


def validate_js(html: str) -> None:
    """
    Parse the emitted <script> with node before writing the file.

    Checking that a string like "Plotly.react" appears in the output proves
    nothing: a single syntax error kills the whole script and every control on
    the page renders dead. This catches that class of failure, which is exactly
    what a placeholder substitution can introduce.
    """
    import re
    import subprocess
    import tempfile

    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not blocks:
        raise SystemExit("no inline script found in output")
    for i, js in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
            path = f.name
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"SYNTAX ERROR in inline script {i}:\n{r.stderr}")
    print(f"  node --check passed on {len(blocks)} inline script(s)")


def main():
    p = pd.read_csv("data/processed/panel.csv")
    data = json.dumps(interactive_data(p), separators=(",", ":"))
    figs = {k: "data:image/png;base64," +
            base64.b64encode(Path("results/figures/" + f).read_bytes()).decode()
            for k, f in FIGS}

    tpl = Path("docs/_template.html").read_text()
    out = tpl.replace("__DATA__", data)
    for k, v in figs.items():
        out = out.replace(f"__FIG_{k.upper()}__", v)

    validate_js(out)
    Path("docs/index.html").write_text(out)
    print(f"wrote docs/index.html  ({len(out)//1024} KB, self-contained, JS validated)")


if __name__ == "__main__":
    main()
