"""
Backward Transfer on Large Number of Tasks.
Two plots designed to perfectly overlap for a smooth before/after reveal.
"""

import matplotlib.pyplot as plt

# (method, BWT %)
METHODS = [
    ("InfLoRA",   -4.9),
    ("CorDA",     -4.5),
    ("LoRM-AB",   -4.1),
    ("O-LoRA",    -4.0),
    ("OPCM",      -3.9),
    ("MagMax",    -3.8),
    ("SLAO",      -3.5),
    ("SAPT-LoRA", -2.9),
    ("CLUE",      +1.0),
]

# GRAY = "#888780"
GRAY = "#FF7276" # light red
TEAL = "#0F6E56"

SCALE = .5

def make_figure(show_clue: bool, savepath: str):
    n = len(METHODS)
    clue_idx = n - 1

    values = [b for _, b in METHODS]
    labels = [m for m, _ in METHODS]

    colors = [GRAY] * n
    colors[clue_idx] = TEAL

    fig, ax = plt.subplots(figsize=(10*SCALE, 5*SCALE), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(n))
    plot_values = list(values)
    plot_colors = list(colors)

    if not show_clue:
        plot_values[clue_idx] = 0       # no visible bar
        plot_colors[clue_idx] = (1, 1, 1, 0)
        labels[clue_idx] = ""           # no x-tick label

    bars = ax.bar(x, plot_values, color=plot_colors, width=0.68,
                  zorder=3, edgecolor="none")

    # Zero line
    ax.axhline(0, color="#222222", lw=1.3*SCALE, zorder=2)

    # Axes — identical in both plots
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11*SCALE, color="#333")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylabel("BWT %", fontsize=12*SCALE, color="#333")
    ax.set_ylim(-5.9, 2.0)
    ax.set_yticks([-5, -4, -3, -2, -1, 0, 1])
    ax.set_yticklabels(["-5", "-4", "-3", "-2", "-1", "0", "+1"],
                       fontsize=10*SCALE, color="#555")

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=6*SCALE)

    ax.grid(axis="y", color="#e6e6e6", lw=0.6*SCALE, zorder=0)
    ax.set_axisbelow(True)

    # Value labels outside each bar
    for i, v in enumerate(values):
        if not show_clue and i == clue_idx:
            continue
        is_clue = (i == clue_idx)
        color = TEAL if is_clue else "#555555"
        weight = "bold" if is_clue else "normal"
        size = 12 if is_clue else 10
        if v < 0:
            ax.text(i, v - 0.14, f"{v:+.1f}", ha="center", va="top",
                    fontsize=size*SCALE, color=color, fontweight=weight)
        else:
            ax.text(i, v + 0.14, f"{v:+.1f}", ha="center", va="bottom",
                    fontsize=size*SCALE, color=color, fontweight=weight)

    ax.set_title("Backward Transfer on Large Number of Tasks",
                 fontsize=7, pad=14*SCALE, loc="left", color="#222222")

    plt.tight_layout()
    # fig.savefig(f"{savepath}.pdf", bbox_inches="tight")
    fig.savefig(f"{savepath}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_figure(show_clue=False, savepath="bwt_1_before")
    make_figure(show_clue=True,  savepath="bwt_2_after")
