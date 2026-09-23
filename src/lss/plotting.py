from __future__ import annotations

import matplotlib.pyplot as plt


def plot_latent_coordinates(frame_table, selected_ids, source_labels, source_colors, *, frame_stop, title):
    """Plot raw z0/z1 versus saved frame for explicitly selected trajectories.

    Selection and cadence remain visible in the calling notebook. No alignment,
    scaling, smoothing or implicit trajectory selection is performed here.
    """
    required = {'source', 'trajectory_id', 'frame', 'z0', 'z1'}
    missing = required - set(frame_table.columns)
    if missing:
        raise ValueError(f'Missing latent-table columns: {sorted(missing)}')
    if frame_stop <= 0 or not selected_ids:
        raise ValueError('Require a positive exclusive frame_stop and selected trajectories.')
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    for source, ids in selected_ids.items():
        for index, trajectory_id in enumerate(ids):
            trajectory = frame_table.loc[
                frame_table.source.eq(source) & frame_table.trajectory_id.eq(trajectory_id)
                & frame_table.frame.ge(0) & frame_table.frame.lt(frame_stop)
            ].sort_values('frame')
            if trajectory.empty or trajectory.frame.duplicated().any():
                raise ValueError(f'Missing or duplicate frames: {source}/{trajectory_id}')
            label = source_labels[source] if index == 0 else None
            for ax, coordinate in zip(axes, ('z0', 'z1')):
                ax.plot(trajectory.frame, trajectory[coordinate], color=source_colors[source],
                        lw=1.15, alpha=.72, label=label)
    for ax, coordinate in zip(axes, ('z0', 'z1')):
        ax.set(title=coordinate, xlabel='frame', ylabel=coordinate)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return fig, axes


PAPER_COLORS = {
    # Darkened Okabe-Ito derivatives: color-vision-safe and sufficiently
    # contrasted against the white paper background for lines and markers.
    "ink": "#25282A",
    "blue": "#0072B2",
    "red": "#C84E00",
    "green": "#008A64",
    "orange": "#B87500",
    "gold": "#8A741C",
    "purple": "#A85D91",
    "sky": "#397EA8",
    "slate": "#626C72",
    "light": "#B7C1C6",
    "grid": "#D8DEE1",
}

DATASET_COLORS = {
    "depablo_low_temp": PAPER_COLORS["blue"],
    "depablo": PAPER_COLORS["blue"],
    "De Pablo": PAPER_COLORS["blue"],
    "de Pablo": PAPER_COLORS["blue"],
    "reid": PAPER_COLORS["red"],
    "Reid": PAPER_COLORS["red"],
    "depablo_mixed_temp": PAPER_COLORS["green"],
    "de Pablo mixed-T": PAPER_COLORS["green"],
    "lj_noisy": PAPER_COLORS["purple"],
    "Noisy LJ": PAPER_COLORS["purple"],
    "unknown": PAPER_COLORS["slate"],
}

LATENT_DIMENSION_COLORS = {
    1: PAPER_COLORS["purple"],
    2: PAPER_COLORS["blue"],
    4: PAPER_COLORS["orange"],
    8: PAPER_COLORS["green"],
}

MEASUREMENT_STYLES = {
    "AE ceiling": {"linestyle": "-", "marker": "o"},
    "Latent propagator": {"linestyle": "--", "marker": "s"},
    "Latent rollout": {"linestyle": "--", "marker": "s"},
    "latent simulator": {"linestyle": "--", "marker": "s"},
    "spatial GNN": {"linestyle": "-", "marker": "o"},
}

NETWORK_NODE_COLOR = PAPER_COLORS["red"]
NETWORK_EDGE_COLOR = PAPER_COLORS["slate"]

STATE_COLORS = {
    "folded": PAPER_COLORS["blue"],
    "unfolded": PAPER_COLORS["red"],
}


_LIGHT_COLORS = dict(PAPER_COLORS)
_DARK_COLORS = dict(PAPER_COLORS, ink="#E6EDF3", blue="#56B4E9", red="#F08A62",
                    green="#56C9A5", orange="#E9B65C", gold="#D8C76A",
                    purple="#D99BC7", sky="#83BCDE", slate="#AAB8C2",
                    light="#647581", grid="#3C4856")


def apply_editorial_style(mode: str = "light") -> None:
    """Apply the shared light/dark theme before creating figures.

    Notebook switch: ``PLOT_THEME = "dark"`` followed by
    ``apply_editorial_style(PLOT_THEME)``. Rerun plotting cells after switching.
    """
    palette = {"light": _LIGHT_COLORS, "dark": _DARK_COLORS}[mode]
    replacements = {PAPER_COLORS[key]: palette[key] for key in PAPER_COLORS}
    for mapping in (DATASET_COLORS, LATENT_DIMENSION_COLORS, STATE_COLORS):
        mapping.update({key: replacements[value] for key, value in mapping.items()})
    PAPER_COLORS.update(palette)
    global NETWORK_NODE_COLOR, NETWORK_EDGE_COLOR
    NETWORK_NODE_COLOR = palette["red"]
    NETWORK_EDGE_COLOR = palette["slate"]
    background = {"light": "white", "dark": "#171D26"}[mode]
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": background,
            "savefig.facecolor": background,
            "savefig.edgecolor": background,
            "text.color": PAPER_COLORS["ink"],
            "legend.facecolor": background,
            "legend.edgecolor": PAPER_COLORS["grid"],
            "figure.constrained_layout.use": False,
            "axes.facecolor": background,
            "axes.edgecolor": PAPER_COLORS["ink"],
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PAPER_COLORS["ink"],
            "axes.titlecolor": PAPER_COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": PAPER_COLORS["ink"],
            "ytick.color": PAPER_COLORS["ink"],
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": PAPER_COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": plt.cycler(
                color=[
                    PAPER_COLORS["blue"], PAPER_COLORS["red"],
                    PAPER_COLORS["green"], PAPER_COLORS["purple"],
                    PAPER_COLORS["orange"], PAPER_COLORS["sky"],
                ]
            ),
        }
    )


def paper_caption(text: str) -> None:
    print(text)


def style_axes(ax, *, xlabel=None, ylabel=None, legend: bool = False) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False, loc="best")


def dataset_color(name: str) -> str:
    """Return the canonical paper color for a dataset name or label."""

    return DATASET_COLORS.get(str(name), PAPER_COLORS["slate"])


def latent_dimension_color(latent_dim: int) -> str:
    """Return the canonical paper color for a latent capacity."""

    return LATENT_DIMENSION_COLORS.get(int(latent_dim), PAPER_COLORS["slate"])


def measurement_style(name: str) -> dict:
    """Return a copy of the canonical line/marker style for a measurement."""

    return dict(MEASUREMENT_STYLES.get(str(name), {"linestyle": "-", "marker": "o"}))
