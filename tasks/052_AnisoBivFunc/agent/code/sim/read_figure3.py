"""Read the boxplot statistics of Figure 3 of the paper (R_alpha risk) out of
the vector PDF panels in paper/src/plots/alpha_adj_*.pdf.

The panels are tikzDevice/ggplot2 output: y tick labels give the pixel->value
calibration, the grey fill of each box matches the grey of its legend key, so
each box can be attributed to a (sigma, alpha) configuration without relying on
reading the picture by eye.  Output: results/paper_fig/figure3_values.csv
"""
import csv, glob, json, os, re
import pdfplumber

ALPHAS = [0.10472, 0.62832, 0.78540, 1.04720, 1.46608]
SIGMAS = [0.1, 0.5, 1.0]
PANEL_RE = re.compile(r"alpha_adj_N(\d+)_M(\d+)\.pdf$")


def grey(rect):
    c = rect.get("non_stroking_color")
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return round(float(c), 4)
    return round(sum(float(x) for x in c) / len(c), 4)


def panel_stats(path):
    with pdfplumber.open(path) as pdf:
        p = pdf.pages[0]
        words = p.extract_words()
        # --- y calibration from the tick labels (values 0.00, 0.03, 0.06, 0.09)
        ticks = []
        for w in words:
            m = re.fullmatch(r"0\.\d{2,4}", w["text"])
            if m and w["x0"] < 60:
                ticks.append((float(w["text"]), (w["top"] + w["bottom"]) / 2))
        ticks.sort()
        (v0, p0), (v1, p1) = ticks[0], ticks[-1]
        scale = (p1 - p0) / (v1 - v0)          # px per unit of R_alpha
        to_val = lambda y: v0 + (y - p0) / scale
        # --- legend keys: the small filled rects of the legend (right/bottom)
        legend = {}
        for r in p.rects:
            g = grey(r)
            if r["x0"] > 300 and 5 < (r["x1"] - r["x0"]) < 16 and (r["bottom"] - r["top"]) < 12:
                legend.setdefault(g, r["top"])
        legend_greys = [g for g, _ in sorted(legend.items(), key=lambda kv: kv[1])]
        assert len(legend_greys) == 5, legend_greys
        # --- boxes: width ~11 px, left of the legend
        boxes = [r for r in p.rects
                 if 50 < r["x0"] < 300 and r["x1"] < 300 and 8 < (r["x1"] - r["x0"]) < 14]
        boxes.sort(key=lambda r: r["x0"])
        lines = [l for l in p.lines if l["width"] is not None]
        out = []
        for b in boxes:
            cx = (b["x0"] + b["x1"]) / 2
            g = grey(b)
            alpha = ALPHAS[legend_greys.index(g)] if g in legend_greys else None
            # median: horizontal line inside the box
            med = [to_val((l["top"] + l["bottom"]) / 2) for l in lines
                   if b["top"] - 0.6 <= (l["top"] + l["bottom"]) / 2 <= b["bottom"] + 0.6
                   and abs(l["x0"] - b["x0"]) < 1.5 and abs(l["x1"] - b["x1"]) < 1.5
                   and abs(l["top"] - l["bottom"]) < 0.6]
            # whiskers: vertical lines at the box centre, above / below the box
            vert = [l for l in lines
                    if abs((l["x0"] + l["x1"]) / 2 - cx) < 0.6 and abs(l["x1"] - l["x0"]) < 0.6
                    and 0.5 < abs(l["bottom"] - l["top"]) < 150]
            # whiskers are attached to the box: the far end is the hinge
            up = [to_val(l["top"]) for l in vert if abs(l["bottom"] - b["top"]) <= 1.0]
            dn = [to_val(l["bottom"]) for l in vert if abs(l["top"] - b["bottom"]) <= 1.0]
            out.append(dict(n_side=int(PANEL_RE.search(path).group(1)),
                            side=int(PANEL_RE.search(path).group(2)),
                            sigma=None, alpha=alpha, grey=g,
                            q1=to_val(b["bottom"]), q3=to_val(b["top"]),
                            median=max(med) if med else None,
                            whisker_lo=min(dn) if dn else None,
                            whisker_hi=max(up) if up else None,
                            x0=b["x0"]))
        return out


def main():
    rows = []
    for path in sorted(glob.glob("paper/src/plots/alpha_adj_*.pdf")):
        st = panel_stats(path)
        # sigma: group the 15 boxes into 3 clusters of 5 by x gap
        st.sort(key=lambda d: d["x0"])
        groups, cur = [], [st[0]]
        for prev, r in zip(st, st[1:]):
            if r["x0"] - prev["x0"] > 20:
                groups.append(cur); cur = []
            cur.append(r)
        groups.append(cur)
        assert len(groups) == 3 and all(len(g) == 5 for g in groups), [len(g) for g in groups]
        for sg, g in zip(SIGMAS, groups):
            for r in g:
                r["sigma"] = sg
                rows.append(r)
        # within a group the five boxes are dodged left -> right in the order of
        # the alpha factor levels; the grey sequence must agree with the legend
        gseq = [round(r["grey"], 4) for r in g]
        assert gseq == sorted(gseq), gseq
    os.makedirs("results/paper_fig", exist_ok=True)
    with open("results/paper_fig/figure3_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print(f"N={r['n_side']} side={r['side']} sigma={r['sigma']} alpha={r['alpha']:.3f} "
              f"| q1={r['q1']:.4f} med={r['median']:.4f} q3={r['q3']:.4f} "
              f"whisk=[{r['whisker_lo']:.4f},{r['whisker_hi']:.4f}]")


if __name__ == "__main__":
    main()
