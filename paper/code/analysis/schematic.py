"""Figure: what TVT is and why S = TVT + Z, drawn from the real data of one training well.

    python -m analysis.schematic                 # well 5bd25f59 (layers rising)  -> figures/s_tvt_z.pdf / .png
    python -m analysis.schematic --well 8cc800b3 # layers dipping down            -> figures/s_tvt_z_8cc800b3.pdf / .png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ssv.data import data_dir

DEFAULT_WELL = "5bd25f59"
ALL_TOPS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]   # formation tops in the horizontal-well files
COLORS = ["#f2d49b", "#b8d8a8", "#9cc3e6", "#d9a6c8"]
OTHER = "#e6e6e6"
SKY = "#eef5fb"
OUT = os.environ.get("FIGURES", "figures")


def broken_band(ax, x0, x1, ylo, yhi, n=40, amp=None):
    """White band with two wavy edges: the part of the vertical axis that is left out."""
    amp = amp if amp is not None else 0.12 * (yhi - ylo)
    xx = np.linspace(x0, x1, 400)
    wave = amp * np.sin(np.linspace(0, n * np.pi, 400))
    ax.fill_between(xx, ylo + wave, yhi + wave, color="w", lw=0, zorder=4)
    for y in (ylo, yhi):
        ax.plot(xx, y + wave, color="0.4", lw=0.8, zorder=4)


def pick_tops(hw, K):
    """5 consecutive tops (4 layers) that contain as much of the lateral as possible (ties: the deeper window)."""
    avail = [t for t in ALL_TOPS if t in hw and hw[t].notna().all()]
    Z = hw.Z.to_numpy()[K:]
    best = None
    for i in range(len(avail) - 4):
        tops = avail[i:i + 5]
        inside = np.mean((Z <= hw[tops[0]].to_numpy()[K:]) & (Z >= hw[tops[-1]].to_numpy()[K:]))
        if best is None or inside >= best[0]:
            best = (inside, tops)
    return best[1]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--well", default=DEFAULT_WELL)
    WELL = ap.parse_args(argv).well
    base = data_dir()
    hw = pd.read_csv(f"{base}/train/{WELL}__horizontal_well.csv")
    tw = pd.read_csv(f"{base}/train/{WELL}__typewell.csv")
    K = int(hw.TVT_input.notna().sum())
    TOPS = pick_tops(hw, K)
    x = (hw.MD - hw.MD.iloc[0]).to_numpy()
    Z, gr = hw.Z.to_numpy(), hw.GR.to_numpy()
    H = np.array([hw[t].to_numpy() for t in TOPS])                   # elevations of the layer tops (ft, < 0)
    c = np.array([np.median(hw.TVT + hw.Z - hw[t]) for t in TOPS])  # TVT of each top in the typewell
    S = hw.TVT.to_numpy() + Z                                        # elevation of the TVT = 0 surface

    # x: the typewell sits at the left edge; every surface (layer tops and TVT = 0) is extended linearly to the plot edges
    XL, XR = -250.0, x[-1] * 1.08
    sl = lambda h, i0, i1: (h[i1] - h[i0]) / (x[i1] - x[i0])
    xe = np.r_[np.linspace(XL, -1, 30), x, np.linspace(x[-1] + 1, XR, 40)]
    ext = lambda h: np.r_[h[0] + sl(h, 0, 2000) * (xe[:30] - x[0]), h, h[-1] + sl(h, -200, -1) * (xe[-40:] - x[-1])]
    He = np.array([ext(h) for h in H])
    Se = He[0] + c[0]                                               # TVT = 0 surface = top of layer 1 raised by its TVT

    # display coordinates: true elevations for the geology; two breaks remove ~7,700 ft (geology -> sea level)
    # and ~1,700 ft (sea level -> TVT = 0 surface)
    ygeo, ybot = max(H[0].max(), Z.max()) + 140, H[-1].min() - 40
    bA = (ygeo + 20, ygeo + 70)
    ysea = ygeo + 125
    bB = (ysea + 40, ysea + 90)
    if Se.min() <= 0:
        raise SystemExit("this layout draws the TVT = 0 surface above sea level; pick a well where S = TVT + Z > 0")
    s0 = ysea + 135 - Se.min()                                      # display shift of the TVT = 0 surface
    Sd = Se + s0
    ytop = Sd.max() + 45

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})  # text stays text (Canva, Illustrator)
    fig = plt.figure(figsize=(10.5, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 4.4], height_ratios=[1.0, 5.0], hspace=0.08, wspace=0.0)
    ax_gr, ax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_tw = fig.add_subplot(gs[1, 0])

    # ---- cross-section ----
    ax.fill_between([XL, XR], ysea, ytop + 2000, color=SKY, lw=0)
    ax.fill_between(xe, He[0], ysea, color=OTHER, lw=0)
    ax.fill_between(xe, ybot, He[-1], color=OTHER, lw=0)
    for k in range(4):
        ax.fill_between(xe, He[k], He[k + 1], color=COLORS[k], lw=0)
        ax.plot(xe, He[k], color="0.35", lw=0.6)
        xl = int(len(x) * 0.42)
        ax.text(x[xl], (H[k][xl] + H[k + 1][xl]) / 2, f"layer {k + 1}", ha="center", va="center", fontsize=9, weight="bold")
    ax.plot(xe, He[-1], color="0.35", lw=0.6)
    ax.plot(xe, Sd, color="k", lw=1.4, ls=(0, (6, 3)), zorder=6)
    rising = Se[-1] > Se[0]
    xlab = 60 if rising else XR - 60
    span = 0.27 * (XR - XL)                                          # approximate width of the two-line label
    xr = np.linspace(xlab, xlab + span if rising else xlab - span, 50)
    ylab = np.interp(xr, xe, Sd).max() + 30                         # clear the line over the whole label width
    ax.text(xlab, ylab, "TVT = 0 surface\n(parallel to the layers)", ha="left" if rising else "right", va="bottom",
            fontsize=8.5, weight="bold", zorder=7)
    ytop = max(ytop, ylab + 60)
    xo = x[len(x) // 7] if rising else x[len(x) * 5 // 7]
    ax.text(xo, ygeo - 25, "overlying layers", fontsize=8, color="0.35", va="top")
    ax.plot(x[:K], Z[:K], "k", lw=2.2, label="horizontal well, known TVT")
    ax.plot(x[K - 1:], Z[K - 1:], color="k", lw=2.2, ls=(0, (4, 1.5)), label="horizontal well, TVT to predict")
    ax.axvline(x[K - 1], ymax=(ygeo - ybot) / (ytop - ybot), color="0.3", lw=0.8, ls="--")
    ax.text(x[K - 1] + 40, ygeo - 30, "end of the known section", fontsize=8, rotation=90, va="top", color="0.3")
    ax.axhline(ysea, color="C0", ls="--", lw=1.2, zorder=5)
    ax.text(60, ysea + 6, "sea level (elevation 0)", color="C0", ha="left", va="bottom", fontsize=8, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc=SKY, ec="none"))
    omitA = round((-ygeo - (ysea - ygeo)) / 100) * 100              # elevation span left out below sea level
    omitB = round((Se.min() - (Sd.min() - ysea)) / 100) * 100       # and between sea level and the TVT = 0 surface
    for lo_, hi_, txt in [(bA[0], bA[1], f"about {omitA:,.0f} ft omitted"), (bB[0], bB[1], f"about {omitB:,.0f} ft omitted")]:
        broken_band(ax, XL, XR, lo_, hi_)
        ax.text(XR - 60, (lo_ + hi_) / 2, txt, ha="right", va="center", fontsize=7, color="0.35", zorder=6)
    ax.set_xlim(XL, XR); ax.set_ylim(ybot, ytop)
    ticks = [t for t in np.arange(np.floor(ybot / 200) * 200, ygeo, 200) if ybot < t < bA[0] - 10]
    ax.set_yticks(ticks + [ysea]); ax.set_yticklabels([f"{t:.0f}" for t in ticks] + ["0"])
    ax.set_xticks([t for t in range(0, int(XR), 1000)])
    ax.set_xlabel("measured depth from the first row of the horizontal well (ft)")
    ax.set_ylabel("elevation relative to sea level (ft)")
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right()
    ax.legend(loc="lower right" if rising else "lower left", frameon=False, fontsize=8)

    # at the middle of the section to predict: S (sea level -> TVT = 0 surface), Z (sea level -> well),
    # TVT (TVT = 0 surface -> well)
    xm = (x[K - 1] + x[-1]) / 2
    xs, xz, xt = xm - 230, xm, xm + 150
    kw = dict(mutation_scale=12, lw=1.6, shrinkA=0, shrinkB=0, zorder=8)
    ax.add_patch(FancyArrowPatch((xs, ysea), (xs, np.interp(xs, xe, Sd)), arrowstyle="-|>", color="C0", **kw))
    ax.add_patch(FancyArrowPatch((xz, ysea), (xz, np.interp(xz, x, Z)), arrowstyle="-|>", color="C3", **kw))
    ax.add_patch(FancyArrowPatch((xt, np.interp(xt, xe, Sd)), (xt, np.interp(xt, x, Z)), arrowstyle="<|-|>", color="k", **kw))
    ax.plot([xz, xt + 15], [np.interp(xz, x, Z), np.interp(xt, x, Z)], color="k", lw=0.8, ls=":")
    ax.plot(xz, np.interp(xz, x, Z), "o", ms=5, color="C3", zorder=9)
    ax.text(xs - 30, (ysea + np.interp(xs, xe, Sd)) / 2, r"$S$", color="C0", ha="right", va="center", fontsize=11)
    ax.text(xz - 30, (ygeo + np.interp(xz, xe, He[0])) / 2, r"$Z<0$", color="C3", ha="right", va="center", fontsize=11)
    ax.text(xt + 30, (ygeo + np.interp(xt, xe, He[0])) / 2, "TVT", ha="left", va="center", fontsize=11, weight="bold")
    ax.plot(x[-1], Z[-1], "o", ms=8, mfc="w", mec="k", mew=1.5, zorder=9)
    ax.annotate("drill bit", (x[-1], Z[-1]), xytext=(x[-1] - 250, Z[-1] + 170), fontsize=9, weight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7), zorder=9)

    ax_eq = fig.add_subplot(gs[0, 0]); ax_eq.axis("off")
    ax_eq.text(0.45, 0.5, "TVT = 0: the typewell's zero depth,\ncarried parallel to the layers\n"
               "TVT: depth of the bit below it\n$S$: elevation of that surface\n\n$S$ = TVT + $Z$",
               transform=ax_eq.transAxes, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="w", ec="0.6"))

    # ---- lateral GR along MD ----
    ax_gr.plot(x, gr, color="0.2", lw=0.7)
    ax_gr.set_xlim(ax.get_xlim()); ax_gr.set_xticks(ax.get_xticks()); ax_gr.set_ylabel("lateral GR (API)")
    ax_gr.yaxis.set_label_position("right"); ax_gr.yaxis.tick_right(); ax_gr.tick_params(labelbottom=False)
    lay = np.sum(Z[:, None] < H.T, 1) - 1
    ax_gr.fill_between(x, 0, 1, where=(lay < 0) | (lay > 3), color=OTHER, transform=ax_gr.get_xaxis_transform(), lw=0)
    for k in range(4):
        m = lay == k
        if m.any():
            ax_gr.fill_between(x, 0, 1, where=m, color=COLORS[k], transform=ax_gr.get_xaxis_transform(), lw=0)
    ax_gr.set_title(f"Well {WELL}: GR recorded along MD (shading: layer the bit is in)", fontsize=8, loc="left")
    ax_gr.axvline(x[K - 1], color="0.3", lw=0.8, ls="--")

    # ---- typewell at the left edge: same display coordinates; TVT = 0 is its zero depth ----
    e_of = lambda t: He[0][0] + c[0] - t          # typewell TVT -> elevation at the typewell (left edge)
    tvt_t, gr_t = tw.TVT.to_numpy(), tw.GR.to_numpy()
    tmin = c[0] - (ygeo - He[0][0])                  # TVT at the top of the geology window
    m = (tvt_t >= tmin) & (e_of(tvt_t) >= ybot)
    ax_tw.fill_between([0, 1], ybot, ytop, color="w", transform=ax_tw.get_yaxis_transform(), lw=0)
    ax_tw.axhspan(He[0][0], ygeo, color=OTHER, lw=0); ax_tw.axhspan(ybot, He[-1][0], color=OTHER, lw=0)
    ax_tw.axhspan(ysea, ytop + 2000, color=SKY, lw=0)
    for k in range(4):
        ax_tw.axhspan(He[k + 1][0], He[k][0], color=COLORS[k], lw=0)
        ax_tw.text(0.97, (He[k][0] + He[k + 1][0]) / 2, f"layer {k + 1}", transform=ax_tw.get_yaxis_transform(), ha="right",
                   va="center", fontsize=8, weight="bold", bbox=dict(boxstyle="round,pad=0.15", fc=COLORS[k], ec="none", alpha=0.85))
    ax_tw.plot(gr_t[m], e_of(tvt_t[m]), color="0.15", lw=0.7)
    ax_tw.axhline(Sd[0], color="k", lw=1.4, ls=(0, (6, 3)))
    ax_tw.text(0.04, Sd[0] + 10, "TVT = 0 (typewell zero depth)", transform=ax_tw.get_yaxis_transform(), fontsize=7.5, va="bottom",
               weight="bold")
    ax_tw.axhline(ysea, color="C0", ls="--", lw=1.2)
    xl0, xl1 = 0, 200
    ax_tw.set_xlim(xl0, xl1)
    for lo_, hi_ in (bA, bB):
        broken_band(ax_tw, xl0, xl1, lo_, hi_, n=8)
    tt = [t for t in np.arange(np.floor(c[0] / 200) * 200 - 2000, c[-1] + 2000, 200) if e_of(t) < bA[0] - 10 and e_of(t) > ybot]
    ax_tw.set_yticks([e_of(t) for t in tt] + [Sd[0]]); ax_tw.set_yticklabels([f"{t:.0f}" for t in tt] + ["0"])
    ax_tw.set_ylim(ybot, ytop)
    ax_tw.set_xlabel("typewell GR (API)"); ax_tw.set_ylabel("TVT = depth below the TVT = 0 level (ft)")
    ax_tw.set_title("typewell", fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    name = "s_tvt_z" if WELL == DEFAULT_WELL else f"s_tvt_z_{WELL}"
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=110)
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight")
    print("written", os.path.join(OUT, name + ".pdf"), "tops", TOPS)
    print("TVT of the tops", c.round(1), "S = TVT + Z along the well: min/median/max", S.min().round(0), np.median(S).round(0),
          S.max().round(0), "check S - (top 1 elevation + its TVT):", np.abs(S - (H[0] + c[0])).max().round(3))


if __name__ == "__main__":
    main()
