"""
Validation_Examples.py  -  check Gaussian_Plume_Stack_Dispersion.py against published worked examples.
Author: Shahed Assad

What it does
  1. Prints the verification table that appears in the README.
  2. Saves two comparison graphs in the "images" folder next to this file and opens them in windows:
       validation_centreline.png    concentration along the plume centreline (y = 0, z = H)
       validation_ground_level.png  ground-level concentration on the centreline (y = 0, z = 0)

How the comparison works
  The published solutions use the Pasquill-Gifford dispersion coefficients (EPA ISC3 curve fits),
  while this project uses the Briggs (1973) formulas. The same Gaussian plume code is run twice:
    * with Pasquill-Gifford sigmas  ->  the published method (tests the equation and the code)
    * with Briggs sigmas            ->  this project's model
  The published values are marked as points on the graphs.

Example 1: Q = 100 g/s, effective height H = 50 m, wind u = 5 m/s, receptor on the centreline
           1 km downwind at 1.5 m height (AgentCalc worked example, EPA ISC3 curves).
Example 2: Q = 110 g/s, H = 100 m, u = 5 m/s, class D: maximum ground-level concentration
           (course notes: 178 ug/m3 from a nomogram at about 3 km, 183 ug/m3 from the equation).

Keep this file in the same folder as Gaussian_Plume_Stack_Dispersion.py.
Run:  python Validation_Examples.py
"""
import os
import sys
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import Gaussian_Plume_Stack_Dispersion as gp
except ModuleNotFoundError:
    sys.exit("Could not find Gaussian_Plume_Stack_Dispersion.py next to this file.\n"
             "If you opened the zip directly, extract it first (right-click > Extract All),\n"
             "open the extracted folder in VS Code (File > Open Folder) and run this file again.")

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- Pasquill-Gifford sigmas
# EPA ISC3 rural curve fits (Turner's Pasquill-Gifford curves); x in km inside the fits.
_PG_Y = {"A": (24.1670, 2.5334), "B": (18.3330, 1.8096), "C": (12.5000, 1.0857),
         "D": (8.3330, 0.72382), "E": (6.2500, 0.54287), "F": (4.1667, 0.36191)}
_PG_Z = {   # (upper end of the distance range in km, a, b):  sigma_z = a * x_km ** b
    "A": [(0.10, 122.800, 0.94470), (0.15, 158.080, 1.05420), (0.20, 170.220, 1.09320), (0.25, 179.520, 1.12620),
          (0.30, 217.410, 1.26440), (0.40, 258.890, 1.40940), (0.50, 346.750, 1.72830), (np.inf, 453.850, 2.11660)],
    "B": [(0.20, 90.673, 0.93198), (0.40, 98.483, 0.98332), (np.inf, 109.300, 1.09710)],
    "C": [(np.inf, 61.141, 0.91465)],
    "D": [(0.30, 34.459, 0.86974), (1.00, 32.093, 0.81066), (3.00, 32.093, 0.64403), (10.00, 33.504, 0.60486),
          (30.00, 36.650, 0.56589), (np.inf, 44.053, 0.51179)],
    "E": [(0.10, 24.260, 0.83660), (0.30, 23.331, 0.81956), (1.00, 21.628, 0.75660), (2.00, 21.628, 0.63077),
          (4.00, 22.534, 0.57154), (10.00, 24.703, 0.50527), (20.00, 26.970, 0.46713), (40.00, 35.420, 0.37615),
          (np.inf, 47.618, 0.29592)],
    "F": [(0.20, 15.209, 0.81558), (0.70, 14.457, 0.78407), (1.00, 13.953, 0.68465), (2.00, 13.953, 0.63227),
          (3.00, 14.823, 0.54503), (7.00, 16.187, 0.46490), (15.00, 17.836, 0.41507), (30.00, 22.651, 0.32681),
          (60.00, 27.074, 0.27436), (np.inf, 34.219, 0.21716)],
}


def sigmas_pg(x, stab, terrain="rural"):
    """Pasquill-Gifford sigma_y, sigma_z in metres (x in metres)."""
    xk = np.maximum(np.asarray(x, float), 1.0) / 1000.0
    c, d = _PG_Y[stab]
    sy = 465.11628 * xk * np.tan(0.017453293 * (c - d * np.log(xk)))
    conds, vals, lower = [], [], 0.0
    for upper, a, b in _PG_Z[stab]:
        conds.append((xk > lower) & (xk <= upper))
        vals.append(a * xk ** b)
        lower = upper
    sz = np.minimum(np.select(conds, vals), 5000.0)
    return sy, sz


@contextmanager
def pasquill_gifford():
    """Temporarily run the main model with Pasquill-Gifford sigmas instead of Briggs."""
    briggs = gp.sigmas
    gp.sigmas = sigmas_pg
    try:
        yield
    finally:
        gp.sigmas = briggs


# ----------------------------------------------------------------------------- the two examples
def example1(stability="D", terrain="rural"):
    src = gp.Source(Q=100.0, h_s=50.0, T_s=298.0)                 # T_s = T_a: no plume rise, so H = 50 m
    met = gp.Met(u_ref=5.0, z_ref=50.0, stability=stability, T_a=298.0, terrain=terrain)   # 5 m/s at 50 m
    return src, met


def example2():
    src = gp.Source(Q=110.0, h_s=100.0, T_s=298.0)                # H = 100 m
    met = gp.Met(u_ref=5.0, z_ref=100.0, stability="D", T_a=298.0)
    return src, met


def ugm3(src, met, x, y=0.0, z=0.0):
    return float(gp.concentration(x, y, z, src, met)) * 1e6


def pct(value, reference):
    d = 100 * (value / reference - 1)
    return "0.0 %" if abs(d) < 0.05 else f"{d:+.1f} %"


def results():
    r = {}
    s, m = example1("D")
    r["ex1_D_pg"] = None
    with pasquill_gifford():
        r["ex1_D_pg"] = ugm3(s, m, 1000.0, 0.0, 1.5)
    r["ex1_D_br"] = ugm3(s, m, 1000.0, 0.0, 1.5)
    s, m = example1("D", "urban")
    r["ex1_U_br"] = ugm3(s, m, 1000.0, 0.0, 1.5)
    s, m = example1("F")
    with pasquill_gifford():
        r["ex1_F_pg"] = ugm3(s, m, 1000.0, 0.0, 1.5)
    r["ex1_F_br"] = ugm3(s, m, 1000.0, 0.0, 1.5)
    s, m = example2()
    with pasquill_gifford():
        r["ex2_pg"] = gp.ground_max(s, m, xmax=10000.0)
    r["ex2_br"] = gp.ground_max(s, m, xmax=10000.0)
    return r


def print_table(r):
    rows = [
        ("Example 1, rural class D", "866",
         f"{r['ex1_D_pg']:.0f} ({pct(r['ex1_D_pg'], 866)})", f"{r['ex1_D_br']:.0f} ({pct(r['ex1_D_br'], 866)})"),
        ("Example 1, urban class D", "353", "same as Briggs *",
         f"{r['ex1_U_br']:.0f} ({pct(r['ex1_U_br'], 353)})"),
        ("Example 1, rural class F", "23",
         f"{r['ex1_F_pg']:.1f} ({pct(r['ex1_F_pg'], 23)})", f"{r['ex1_F_br']:.1f} ({pct(r['ex1_F_br'], 23)})"),
        ("Example 2, maximum (class D)", "178 / 183",
         f"{r['ex2_pg'][0]:.0f} at {r['ex2_pg'][1] / 1000:.1f} km ({pct(r['ex2_pg'][0], 178)} / {pct(r['ex2_pg'][0], 183)})",
         f"{r['ex2_br'][0]:.0f} at {r['ex2_br'][1] / 1000:.1f} km ({pct(r['ex2_br'][0], 178)} / {pct(r['ex2_br'][0], 183)})"),
    ]
    head = ("Test case", "Published", "This code, P-G sigma", "This code, Briggs sigma (this project)")
    w = [max(len(str(row[i])) for row in rows + [head]) for i in range(4)]
    line = "+" + "+".join("-" * (k + 2) for k in w) + "+"
    print("\nVERIFICATION TABLE (all values in ug/m3; difference from the published value in brackets)")
    print(line)
    print("| " + " | ".join(h.ljust(k) for h, k in zip(head, w)) + " |")
    print(line.replace("-", "="))
    for row in rows:
        print("| " + " | ".join(str(c).ljust(k) for c, k in zip(row, w)) + " |")
    print(line)
    print("Example 1: Q = 100 g/s, H = 50 m, u = 5 m/s, receptor on the centreline 1 km downwind at 1.5 m height.")
    print("Example 2: Q = 110 g/s, H = 100 m, u = 5 m/s, maximum ground-level concentration;"
          " published 178 (nomogram) / 183 (equation).")
    print("* Urban sigmas are the Briggs urban formulas in both the published method and this code.")
    print("P-G sigma = Pasquill-Gifford curves (EPA ISC3 curve fits), the method used by the published solutions.")


# ----------------------------------------------------------------------------- graphs
BLUE, ORANGE, INK, INK2, GRIDC, AXC, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LABEL_BR = "this project's model (Briggs σ)"
LABEL_PG = "published method (Pasquill–Gifford σ)"
TITLES = ["Example 1:  Q = 100 g/s,  H = 50 m,  u = 5 m/s,  rural class D",
          "Example 2:  Q = 110 g/s,  H = 100 m,  u = 5 m/s,  rural class D"]


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, which="major", color=GRIDC, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXC)
    ax.tick_params(colors=INK2)


def plot_centreline(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    x = np.logspace(np.log10(gp.X_MIN), 4, 400)
    for ax, (src, met), title in zip(axes, (example1("D"), example2()), TITLES):
        H = src.h_s
        c_br = gp.concentration(x, 0.0, H, src, met) * 1e6
        with pasquill_gifford():
            c_pg = gp.concentration(x, 0.0, H, src, met) * 1e6
        v_br = ugm3(src, met, 1000.0, 0.0, H)
        with pasquill_gifford():
            v_pg = ugm3(src, met, 1000.0, 0.0, H)
        ax.loglog(x, c_br, color=BLUE, lw=2.6, label=f"{LABEL_BR}:  {v_br:,.0f} µg/m³ at 1 km")
        ax.loglog(x, c_pg, color=ORANGE, lw=2.2, label=f"{LABEL_PG}:  {v_pg:,.0f} µg/m³ at 1 km")
        ax.plot(1000.0, v_br, "o", color=BLUE, ms=7, mec="white", mew=1.5, zorder=5)
        ax.plot(1000.0, v_pg, "o", color=ORANGE, ms=7, mec="white", mew=1.5, zorder=5)
        _style(ax); ax.grid(True, which="minor", color=GRIDC, lw=0.4)
        ax.set(xlim=(gp.X_MIN, 1e4), ylim=(c_br.min() / 2, c_pg.max() * 8), xlabel="downwind distance x (m)",
               ylabel=f"concentration at plume height z = H = {H:g} m (µg/m³)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", frameon=False, fontsize=9.2)
    fig.suptitle("Concentration along the plume centreline (y = 0, z = H): this project vs the published method",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def plot_ground_level(path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    cases = ((example1("D"), 5000.0, 1000.0, 866.0, "published value: 866 µg/m³ at 1 km (receptor at 1.5 m)"),
             (example2(), 10000.0, 3000.0, 178.0,
              "published maximum: 178 µg/m³ at ≈ 3 km (nomogram);\n183 µg/m³ by the equation"))
    for ax, ((src, met), X, xp, cp, ptext), title in zip(axes, cases, TITLES):
        x = np.linspace(gp.X_MIN, X, 1200)
        c_br = gp.concentration(x, 0.0, 0.0, src, met) * 1e6
        m_br = gp.ground_max(src, met, xmax=X)
        with pasquill_gifford():
            c_pg = gp.concentration(x, 0.0, 0.0, src, met) * 1e6
            m_pg = gp.ground_max(src, met, xmax=X)
        top = max(c_br.max(), c_pg.max())
        ax.plot(x, c_br, color=BLUE, lw=2.6, label=f"{LABEL_BR}:  max {m_br[0]:.0f} µg/m³ at {m_br[1]:,.0f} m")
        ax.plot(x, c_pg, color=ORANGE, lw=2.2, label=f"{LABEL_PG}:  max {m_pg[0]:.0f} µg/m³ at {m_pg[1]:,.0f} m")
        ax.plot(*m_br[::-1], "o", color=BLUE, ms=7, mec="white", mew=1.5, zorder=5)
        ax.plot(*m_pg[::-1], "o", color=ORANGE, ms=7, mec="white", mew=1.5, zorder=5)
        ax.plot(xp, cp, "D", color=INK, ms=8, zorder=7, label=ptext)
        _style(ax)
        ax.set(xlim=(0, X), ylim=(0, 1.75 * top), xlabel="downwind distance x (m)",
               ylabel="ground-level concentration (µg/m³)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", frameon=False, fontsize=9.2)
    fig.suptitle("Ground-level concentration on the centreline (y = 0, z = 0): this project vs the published examples",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    r = results()
    print_table(r)
    out = os.path.join(HERE, "images"); os.makedirs(out, exist_ok=True)
    plot_centreline(os.path.join(out, "validation_centreline.png"))
    plot_ground_level(os.path.join(out, "validation_ground_level.png"))
    print(f"\nGraphs saved in: {out}")
    print("    validation_centreline.png\n    validation_ground_level.png")
    gp.show_figures()
