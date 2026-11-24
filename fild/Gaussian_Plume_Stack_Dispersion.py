"""
Gaussian_Plume_Stack_Dispersion.py  -  Gaussian plume from an elevated stack: 3D view + x-y, x-z, y-z views.
Author: Shahed Assad

Model (steady state, continuous point source, flat terrain, total ground reflection)

  C(x,y,z) = Q / (2 pi u sy sz) * exp(-y^2 / 2 sy^2)
             * [ exp(-(z-H)^2 / 2 sz^2) + exp(-(z+H)^2 / 2 sz^2) ]

  x downwind, y crosswind, z height (m); wind blows along +x.
  Q in g/s -> C in g/m3, plotted in ug/m3 (x 1e6).
  H = h_stack + Briggs buoyant plume rise (transitional rise, capped at the final rise).
  sy, sz = Briggs (1973) rural/urban formulas for Pasquill classes A-F.
  u = reference wind corrected to stack height (power law, minimum 1 m/s).
  Optional mixing-height lid L (full image series); if H >= L the plume does not reach the ground.

Valid roughly 100 m - 10 km downwind; no chemistry, deposition or terrain.
Concentrations closer than X_MIN = 100 m are NOT computed (formulas not valid there).
Plume rise is the most uncertain input: C_max varies roughly as 1/H^2 (see rise_sensitivity).
Run:  python Gaussian_Plume_Stack_Dispersion.py     -> PNG figures in the "images" folder next to this file
"""
import os
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))       # folder of this script (figures are saved here)
X_MIN = 100.0     # m, nearest distance where the Briggs formulas are valid
trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------- inputs
@dataclass
class Source:
    Q: float = 100.0      # emission rate, g/s
    h_s: float = 50.0     # physical stack height, m
    d: float = 2.0        # stack inner diameter, m
    v_s: float = 15.0     # exit velocity, m/s
    T_s: float = 420.0    # exit gas temperature, K
    rise_factor: float = 1.0   # multiplies the Briggs plume rise (1 = as published); for sensitivity tests


@dataclass
class Met:
    u_ref: float = 4.0            # wind speed at z_ref, m/s
    z_ref: float = 10.0           # anemometer height, m
    stability: str = "C"          # Pasquill class A (very unstable) ... F (stable)
    T_a: float = 298.0            # ambient temperature, K
    L: Optional[float] = None     # mixing height (inversion lid), m; None = no lid
    terrain: str = "rural"        # "rural" or "urban"


# ----------------------------------------------------------------------------- physics
_RURAL = {   # Briggs (1973), open country, x in metres.
             # Class D sigma_z uses exponent -1/2 (Hanna et al. 1982). A CAP88-PC table uses -1, which would
             # freeze sigma_z at 40 m far downwind - not physical for neutral air - so -1/2 is kept.
    "A": (lambda x: 0.22 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.20 * x),
    "B": (lambda x: 0.16 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.12 * x),
    "C": (lambda x: 0.11 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.08 * x * (1 + 2e-4 * x) ** -0.5),
    "D": (lambda x: 0.08 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.06 * x * (1 + 15e-4 * x) ** -0.5),
    "E": (lambda x: 0.06 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.03 * x * (1 + 3e-4 * x) ** -1),
    "F": (lambda x: 0.04 * x * (1 + 1e-4 * x) ** -0.5, lambda x: 0.016 * x * (1 + 3e-4 * x) ** -1),
}
_URBAN = {   # Briggs (1973), urban
    "A": (lambda x: 0.32 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.24 * x * (1 + 1e-3 * x) ** 0.5),
    "B": (lambda x: 0.32 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.24 * x * (1 + 1e-3 * x) ** 0.5),
    "C": (lambda x: 0.22 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.20 * x),
    "D": (lambda x: 0.16 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.14 * x * (1 + 3e-4 * x) ** -0.5),
    "E": (lambda x: 0.11 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.08 * x * (1 + 15e-4 * x) ** -0.5),
    "F": (lambda x: 0.11 * x * (1 + 4e-4 * x) ** -0.5, lambda x: 0.08 * x * (1 + 15e-4 * x) ** -0.5),
}
_P_RURAL = dict(A=0.07, B=0.07, C=0.10, D=0.15, E=0.35, F=0.55)
_P_URBAN = dict(A=0.15, B=0.15, C=0.20, D=0.25, E=0.30, F=0.30)
_DTHETA_DZ = dict(E=0.02, F=0.035)     # K/m, stable classes


def sigmas(x, stab, terrain="rural"):
    fy, fz = (_RURAL if terrain == "rural" else _URBAN)[stab]
    x = np.asarray(x, float)
    return fy(x), fz(x)


def stack_wind(src, met):
    p = (_P_RURAL if met.terrain == "rural" else _P_URBAN)[met.stability]
    return max(met.u_ref * (src.h_s / met.z_ref) ** p, 1.0)


def plume_rise(x, src, met):
    x = np.maximum(np.asarray(x, float), 0.0)
    F = G * src.v_s * src.d ** 2 * (src.T_s - met.T_a) / (4 * src.T_s)    # buoyancy flux, m4/s3
    if F <= 0:
        return np.zeros_like(x)
    u = stack_wind(src, met)
    trans = 1.6 * F ** (1 / 3) * x ** (2 / 3) / u
    if met.stability in _DTHETA_DZ:
        s = G / met.T_a * _DTHETA_DZ[met.stability]
        final = 2.6 * (F / (u * s)) ** (1 / 3)
    else:
        final = 21.425 * F ** 0.75 / u if F < 55 else 38.71 * F ** 0.6 / u
    return src.rise_factor * np.minimum(trans, final)


def concentration(x, y, z, src, met, x_min=X_MIN, nmax=5):
    """Concentration in g/m3 at (x, y, z); zero upwind of x_min."""
    x, y, z = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, y, z)))
    xs = np.maximum(x, 1.0)
    u = stack_wind(src, met)
    H = src.h_s + plume_rise(xs, src, met)
    sy, sz = sigmas(xs, met.stability, met.terrain)
    lateral = np.exp(-y ** 2 / (2 * sy ** 2))
    if met.L is None:
        vert = np.exp(-(z - H) ** 2 / (2 * sz ** 2)) + np.exp(-(z + H) ** 2 / (2 * sz ** 2))
        valid = x >= x_min
    else:
        L = met.L
        vert = 0.0
        for n in range(-nmax, nmax + 1):
            vert = vert + np.exp(-(z - H + 2 * n * L) ** 2 / (2 * sz ** 2)) \
                        + np.exp(-(z + H + 2 * n * L) ** 2 / (2 * sz ** 2))
        vert = np.where(sz >= 1.6 * L, np.sqrt(2 * np.pi) * sz / L, vert)
        valid = (x >= x_min) & (z <= L) & (H < L)
    C = src.Q / (2 * np.pi * u * sy * sz) * lateral * vert
    return np.where(valid, C, 0.0)


def ground_max(src, met, xmax=10000.0):
    x = np.linspace(X_MIN, xmax, 5000)
    c = concentration(x, 0.0, 0.0, src, met) * 1e6
    i = int(np.argmax(c))
    return float(c[i]), float(x[i])


def mass_balance(src, met, distances=(200, 500, 1000, 2000, 5000)):
    """Flux of pollutant through each vertical plane divided by Q (must be 1.000)."""
    u = stack_wind(src, met)
    rows = []
    for x in distances:
        sy, sz = sigmas(x, met.stability, met.terrain)
        H = src.h_s + float(plume_rise(x, src, met))
        zmax = met.L if met.L else H + 8 * sz
        ys = np.linspace(-8 * sy, 8 * sy, 801)
        zs = np.linspace(0, zmax, 1601)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        C = concentration(x, Y, Z, src, met)
        rows.append((x, trapz(trapz(C, zs, axis=1), ys) * u / src.Q))
    return rows


_MAX_WIND = {"A": 3.0, "B": 5.0, "E": 5.0, "F": 3.0}    # m/s at 10 m, upper wind limits in the Pasquill-Turner table


def check_inputs(src, met):
    """Return a list of warnings about unrealistic inputs (empty list = nothing suspicious)."""
    w = []
    lim = _MAX_WIND.get(met.stability)
    if lim and met.u_ref > lim:
        w.append(f"class {met.stability} normally goes with wind <= {lim:g} m/s at 10 m (you used {met.u_ref:g}).")
    c, x = ground_max(src, met, xmax=10000.0)
    if x > 9500:
        w.append("the ground-level maximum is at or beyond ~10 km, outside the valid range of these formulas.")
    return w


def rise_sensitivity(src, met, factors=(0.5, 0.75, 1.0, 1.25)):
    """Ground-level maximum if the Briggs plume rise were too low / too high."""
    from dataclasses import replace
    return [(f, *ground_max(replace(src, rise_factor=f), met)) for f in factors]


# ----------------------------------------------------------------------------- plotting helpers
BANDS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]     # ug/m3, same in every figure
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})


def _style():
    cols = plt.get_cmap("turbo")(np.linspace(0.06, 0.96, len(BANDS)))
    cmap = ListedColormap(cols[:-1]); cmap.set_over(cols[-1])
    return cmap, BoundaryNorm(BANDS, cmap.N)


def _fill(ax, X, Y, C, **kw):
    cmap, norm = _style()
    return ax.contourf(X, Y, np.where(C >= BANDS[0], C, np.nan), levels=BANDS,
                       cmap=cmap, norm=norm, extend="max", **kw)


def _colorbar(fig, cs, ax=None, **kw):
    cb = fig.colorbar(cs, ax=ax, ticks=BANDS, spacing="uniform", **kw)
    cb.ax.set_yticklabels([f"{b:g}" for b in BANDS])
    cb.set_label("Concentration  (µg/m³)")
    return cb


def _subtitle(src, met):
    Hf = src.h_s + float(plume_rise(1e5, src, met))
    lid = f" | mixing height {met.L:g} m" if met.L else ""
    return (f"Q = {src.Q:g} g/s | stack {src.h_s:g} m | wind {met.u_ref:g} m/s at {met.z_ref:g} m | "
            f"stability class {met.stability} | effective height ≈ {Hf:.0f} m{lid}")


def _extent(src, met):
    cmax, xmax = ground_max(src, met)
    X = float(min(10000, max(3000, np.ceil(1.5 * xmax / 500) * 500)))
    Hf = src.h_s + float(plume_rise(1e5, src, met))
    Z = float(max(250, np.ceil(3.5 * Hf / 50) * 50))
    return X, Z, cmax, xmax


def _draw_stack_side(ax, src, label=True):
    ax.plot([0, 0], [0, src.h_s], color="black", lw=7, solid_capstyle="butt", zorder=10)
    ax.plot(0, src.h_s, "o", color="red", ms=7, zorder=11)
    if label:
        ax.annotate(f"STACK\n{src.h_s:g} m", xy=(0, src.h_s * 0.55), xytext=(-12, 0),
                    textcoords="offset points", fontsize=11, fontweight="bold", ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black"), zorder=12)


# ----------------------------------------------------------------------------- the four figures
def plot_xz(src, met, path):
    X, Z, cmax, xmax = _extent(src, met)
    x = np.linspace(X_MIN, X, 600); z = np.linspace(0, Z, 300)
    Xg, Zg = np.meshgrid(x, z)
    C = concentration(Xg, 0.0, Zg, src, met) * 1e6
    fig, ax = plt.subplots(figsize=(12, 5.4))
    cs = _fill(ax, Xg, Zg, C)
    ax.axvspan(0, X_MIN, color="lightgrey", zorder=1.5)
    ax.text(X_MIN / 2, 0.55 * Z, "not computed (< 100 m)", rotation=90, ha="center", va="center", fontsize=9, zorder=2)
    ax.plot(x, src.h_s + plume_rise(x, src, met), "k--", lw=1.4, label="Plume centreline (effective height H)")
    ax.plot(xmax, 0, "v", color="white", mec="black", ms=12, zorder=11,
            label=f"Max ground-level: {cmax:.0f} µg/m³ at x = {xmax:.0f} m")
    if met.L and met.L < Z:
        ax.axhline(met.L, color="red", ls=":", lw=1.5, label=f"Mixing height {met.L:g} m")
    _draw_stack_side(ax, src)
    ax.axhline(0, color="saddlebrown", lw=3)
    ax.set(xlim=(-0.10 * X, X), ylim=(0, Z), xlabel="Downwind distance x  (m)", ylabel="Height z  (m)")
    ax.set_title("SIDE VIEW (x–z plane, along the plume centreline y = 0)\n" + _subtitle(src, met), fontsize=11)
    ax.annotate("", xy=(0.20, 0.93), xytext=(0.07, 0.93), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"))
    ax.text(0.135, 0.955, "wind", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    _colorbar(fig, cs, ax=ax, pad=0.015)
    fig.tight_layout(); fig.savefig(path, dpi=140)


def plot_xy(src, met, path, z=0.0):
    X, Z, cmax, xmax = _extent(src, met)
    sy_end, _ = sigmas(X, met.stability, met.terrain)
    W = float(np.ceil(3.2 * sy_end / 100) * 100)
    x = np.linspace(X_MIN, X, 700); y = np.linspace(-W, W, 300)
    Xg, Yg = np.meshgrid(x, y)
    C = concentration(Xg, Yg, z, src, met) * 1e6
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    cs = _fill(ax, Xg, Yg, C)
    ax.axvspan(0, X_MIN, color="lightgrey", zorder=1.5)
    ax.text(X_MIN / 2, 0.45 * W, "not computed (< 100 m)", rotation=90, ha="center", va="center", fontsize=8, zorder=2)
    ax.plot(0, 0, marker="^", color="black", ms=16, zorder=11)
    ax.plot(0, 0, marker="^", color="red", ms=7, zorder=12)
    ax.annotate("STACK", xy=(0, 0), xytext=(8, 14), textcoords="offset points", fontweight="bold")
    ax.plot(xmax, 0, "*", color="white", mec="black", ms=17, zorder=11,
            label=f"Max ground-level: {cmax:.0f} µg/m³ at x = {xmax:.0f} m")
    ax.annotate("", xy=(0.20, 0.90), xytext=(0.07, 0.90), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"))
    ax.text(0.135, 0.925, "wind", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    ax.set(xlim=(-0.04 * X, X), ylim=(-W, W), xlabel="Downwind distance x  (m)",
           ylabel="Crosswind distance y  (m)", aspect="equal")
    ax.set_title(f"TOP VIEW (x–y plane) – ground-level concentration, z = {z:g} m\n" + _subtitle(src, met),
                 fontsize=11)
    ax.legend(loc="upper right", framealpha=0.95)
    _colorbar(fig, cs, ax=ax, pad=0.015)
    fig.tight_layout(); fig.savefig(path, dpi=140)


def plot_yz(src, met, path, xs=(500, 1000, 2000)):
    X, Z, cmax, xmax = _extent(src, met)
    fig, axes = plt.subplots(1, len(xs), figsize=(5.0 * len(xs) + 1.0, 5.4), sharey=True)
    cs = None
    for ax, x in zip(np.atleast_1d(axes), xs):
        sy, _ = sigmas(x, met.stability, met.terrain)
        W = float(np.ceil(3.5 * sy / 50) * 50)
        y = np.linspace(-W, W, 300); z = np.linspace(0, Z, 300)
        Yg, Zg = np.meshgrid(y, z)
        C = concentration(x, Yg, Zg, src, met) * 1e6
        cs = _fill(ax, Yg, Zg, C)
        H = src.h_s + float(plume_rise(x, src, met))
        ax.axhline(src.h_s, color="black", ls="-", lw=1.0)
        ax.text(-W * 0.96, src.h_s + 0.015 * Z, f"stack height {src.h_s:g} m", fontsize=9.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.plot(0, H, "k+", ms=14, mew=2.5)
        ax.text(W * 0.06, H + 0.02 * Z, "plume centre", fontsize=9.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.axhline(0, color="saddlebrown", lw=3)
        ax.set(xlim=(-W, W), ylim=(0, Z), xlabel="Crosswind distance y  (m)")
        ax.set_title(f"x = {x:g} m downwind\npeak {C.max():.0f} µg/m³ · ground {C[0].max():.0f} µg/m³", fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel("Height z  (m)")
    fig.suptitle("CROSS-SECTIONS (y–z planes, looking downwind)\n" + _subtitle(src, met), fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.90, 0.94))
    cax = fig.add_axes([0.925, 0.14, 0.014, 0.70])
    _colorbar(fig, cs, ax=None, cax=cax)
    fig.savefig(path, dpi=140)


def plot_3d(src, met, path):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    X, Z, cmax, xmax = _extent(src, met)
    sy_end, _ = sigmas(X, met.stability, met.terrain)
    W = float(np.ceil(3.0 * sy_end / 100) * 100)
    nx, ny, nz = 110, 80, 60
    x = np.linspace(X_MIN, X, nx); y = np.linspace(-W, W, ny); z = np.linspace(0, Z, nz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    C = concentration(Xg, Yg, Zg, src, met) * 1e6

    cmap, norm = _style()
    fig = plt.figure(figsize=(13, 8.2))
    ax = fig.add_subplot(projection="3d")
    ax.set_position([0.0, 0.14, 1.0, 0.78])

    # ground footprint (z = 0)
    Xs, Ys = np.meshgrid(x, y, indexing="ij")
    Cg = C[:, :, 0]
    cs = ax.contourf(Xs, Ys, np.where(Cg >= BANDS[0], Cg, np.nan), levels=BANDS, cmap=cmap, norm=norm,
                     extend="max", zdir="z", offset=0, alpha=0.95)

    # translucent shells of constant concentration
    shells = [(20, 0.20), (100, 0.34), (1000, 0.60)]
    handles = []
    try:
        from skimage.measure import marching_cubes
        for level, alpha in shells:
            if C.max() <= level:
                continue
            verts, faces, _, _ = marching_cubes(C, level=level, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
            verts += np.array([x[0], y[0], z[0]])
            col = cmap(norm(level))
            poly = Poly3DCollection(verts[faces], facecolor=col, edgecolor="none", alpha=alpha)
            ax.add_collection3d(poly)
            handles.append(Patch(facecolor=col, alpha=min(1, alpha + 0.3), label=f"{level:g} µg/m³ surface"))
    except ImportError:               # no scikit-image: show the plume as a point cloud instead
        m = C > 20
        ax.scatter(Xg[m], Yg[m], Zg[m], c=np.log10(C[m]), s=2, alpha=0.2, cmap="turbo")
        handles.append(Patch(facecolor="grey", label="plume cloud (> 10 µg/m³)"))

    # stack and plume centreline
    xl = np.linspace(0, X, 200)
    ax.plot([0, 0], [0, 0], [0, src.h_s], color="black", lw=8, solid_capstyle="butt", zorder=20)
    ax.scatter([0], [0], [src.h_s], color="red", s=60, zorder=21, depthshade=False)
    ax.text(0, 0, src.h_s + 0.16 * Z, f"STACK ({src.h_s:g} m)", fontweight="bold", ha="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black"), zorder=30)
    ax.plot(xl, 0 * xl, src.h_s + plume_rise(xl, src, met), "k--", lw=1.5, zorder=19)
    ax.scatter([xmax], [0], [0], marker="*", color="white", edgecolor="black", s=260, depthshade=False, zorder=22)
    handles += [Patch(facecolor="none", edgecolor="black", ls="--", label="plume centreline"),
                plt.Line2D([], [], marker="*", color="white", mec="black", ms=14, ls="",
                           label=f"max ground level {cmax:.0f} µg/m³ at x = {xmax:.0f} m")]

    ax.set(xlim=(0, X), ylim=(-W, W), zlim=(0, Z), xlabel="Downwind x (m)", ylabel="Crosswind y (m)",
           zlabel="Height z (m)")
    ax.set_box_aspect((3.0, 1.5, 1.0), zoom=1.12); ax.view_init(elev=24, azim=-58)
    ax.zaxis.labelpad = 8
    ax.set_title("3D VIEW – plume from the stack (vertical scale exaggerated)\n" + _subtitle(src, met), fontsize=11)
    ax.legend(handles=handles, loc="upper left", framealpha=0.95, fontsize=10)
    cax = fig.add_axes([0.25, 0.085, 0.50, 0.022])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=BANDS, spacing="uniform")
    cb.ax.set_xticklabels([f"{b:g}" for b in BANDS]); cb.set_label("Ground-level concentration  (µg/m³)")
    fig.savefig(path, dpi=140)


def plot_ground_profile(src, met, path, X=5000.0):
    """Ground-level concentration along the centreline, with a +/-25 % plume-rise sensitivity test."""
    L_SURF, L_TXT, L_TXT2, L_GRID, L_AX = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    cols = {1.0: "#2a78d6", 0.75: "#eb6834", 1.25: "#1baf7a"}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_facecolor(L_SURF)
    x = np.linspace(X_MIN, X, 1500)
    top = 0.0
    for f in (1.0, 0.75, 1.25):
        c = concentration(x, 0.0, 0.0, replace(src, rise_factor=f), met) * 1e6
        lab = "this project (Briggs plume rise)" if f == 1.0 else f"plume rise × {f:g}  (sensitivity test)"
        ax.plot(x, c, color=cols[f], lw=2.6 if f == 1.0 else 2.0, label=lab, zorder=3 if f == 1.0 else 2)
        i = int(np.argmax(c)); top = max(top, c[i])
        ax.plot(x[i], c[i], "o", color=cols[f], ms=8, mec="white", mew=2, zorder=5)
        ax.text(x[i] + 0.016 * X, c[i] + 0.013 * top, f"{c[i]:.0f} µg/m³ at {x[i]:.0f} m", color=L_TXT, fontsize=10.5,
                zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc=L_SURF, ec="none", alpha=0.9))
    H = src.h_s + float(plume_rise(1e6, src, met))
    xs = x[np.argmin(np.abs(sigmas(x, met.stability, met.terrain)[1] - H / np.sqrt(2)))]
    ax.axvline(xs, color=L_AX, lw=1.2)
    ax.text(xs + 0.008 * X, 0.04 * top, r"$\sigma_z = H/\sqrt{2}$" + f"\n(x ≈ {xs:.0f} m)", fontsize=10.5, color=L_TXT2)
    ax.text(0.03 * X, 0.86 * top, "plume has not yet\nreached the ground", fontsize=10.5, color=L_TXT2, va="top")
    ax.text(0.66 * X, 0.46 * top, "dilution: the plume keeps\nspreading, so C falls", fontsize=10.5, color=L_TXT2)
    ax.set(xlim=(0, X), ylim=(0, 1.55 * top), xlabel="downwind distance x (m)",
           ylabel="ground-level concentration on the centreline (µg/m³)")
    ax.set_title("Ground-level concentration along the plume centreline (y = 0, z = 0)\n" + _subtitle(src, met), fontsize=11.5)
    ax.grid(True, color=L_GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(L_AX)
    ax.tick_params(colors=L_TXT2)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def show_figures():
    """Open every figure in its own window (they are already saved as PNG files).
    Does nothing when Python has no screen to draw on (for example on a server)."""
    if not plt.get_fignums():
        return
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo", "template"):
        return
    print("Showing the figures in separate windows (they may open behind VS Code). Close them to finish.")
    plt.show()


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    src = Source(Q=100, h_s=50, d=2.0, v_s=15.0, T_s=420.0)
    met = Met(u_ref=4.0, z_ref=10.0, stability="C", T_a=298.0, L=None, terrain="rural")
    out = os.path.join(HERE, "images"); os.makedirs(out, exist_ok=True)

    cmax, xmax = ground_max(src, met)
    print(f"Wind at stack height   : {stack_wind(src, met):.2f} m/s")
    print(f"Effective stack height : {src.h_s + float(plume_rise(1e5, src, met)):.0f} m")
    print(f"Max ground-level conc. : {cmax:.0f} ug/m3 at x = {xmax:.0f} m")
    print("Mass balance (should be 1.0000):", ", ".join(f"{r:.4f}" for _, r in mass_balance(src, met)))
    for w in check_inputs(src, met):
        print("WARNING:", w)
    print("Sensitivity to plume rise (ground max):",
          "; ".join(f"x{f:g} -> {c:.0f} ug/m3" for f, c, _ in rise_sensitivity(src, met)))

    plot_3d(src, met, f"{out}/plume_3d_view.png")
    plot_xy(src, met, f"{out}/top_view_xy.png")
    plot_xz(src, met, f"{out}/side_view_xz.png")
    plot_yz(src, met, f"{out}/cross_sections_yz.png")
    plot_ground_profile(src, met, f"{out}/ground_level_profile.png")
    print(f"Figures saved in: {out}")
    for name in ("plume_3d_view.png", "top_view_xy.png", "side_view_xz.png", "cross_sections_yz.png",
                 "ground_level_profile.png"):
        print("   ", name)
    show_figures()
