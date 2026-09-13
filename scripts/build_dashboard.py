"""Rebuild docs/index.html from data/processed/panel.csv. Self-contained; deploy anywhere."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from aei_audit import indices as ix

MAPS = [("State", None), ("Census region", "census_region"), ("Time zone", "time_zone"),
        ("BEA region", "bea_region"), ("Census division", "census_division"),
        ("Federal region", "federal_region")]

def build_data(p):
    D = {}
    for w, g in p.groupby("wave"):
        D[w] = {}
        for nm, col in MAPS:
            a = (g[["state","usage_share","pop_share_fixed"]].rename(columns={"state":"unit"})
                 if col is None else
                 g.groupby(col)[["usage_share","pop_share_fixed"]].sum().reset_index()
                  .rename(columns={col:"unit"}))
            a["aui"] = a.usage_share / a.pop_share_fixed
            a = a.sort_values("aui", ascending=False)
            cx, cy = ix.lorenz_points(a.aui.values, a.pop_share_fixed.values)
            D[w][nm] = {"units":a.unit.tolist(), "aui":[round(v,4) for v in a.aui],
                        "pop":[round(v,4) for v in a.pop_share_fixed],
                        "gini":round(ix.gini(a.aui.values,a.pop_share_fixed.values),4),
                        "theil":round(ix.theil(a.aui.values,a.pop_share_fixed.values),4),
                        "k":len(a), "lx":[round(v,4) for v in cx], "ly":[round(v,4) for v in cy]}
    return D

if __name__ == "__main__":
    p = pd.read_csv("data/processed/panel.csv")
    data = json.dumps(build_data(p), separators=(",",":"))
    Path("docs/dashboard_data.json").write_text(data)
    tpl = Path("docs/index.html").read_text()
    import re
    tpl = re.sub(r"const D = \{.*?\};", "const D = " + data + ";", tpl, count=1, flags=re.S)
    Path("docs/index.html").write_text(tpl)
    print(f"rebuilt docs/index.html ({len(data)//1024} KB data)")
