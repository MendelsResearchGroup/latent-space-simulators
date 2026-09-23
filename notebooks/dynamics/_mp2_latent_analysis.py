"""Inline, post-fit views of the shared 2D message-passing autoencoder."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from lss.plotting import apply_editorial_style, DATASET_COLORS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'notebooks/results/mp2_latent_analysis'
SOURCES = ('reid', 'depablo_low_temp', 'lj_noisy')
LABELS = dict(zip(SOURCES, ('Reid', 'dePablo low-T', 'Noisy LJ')))
FRAMES = (5, 10, 25, 50, 100, 150, 199)


def load_tables():
    scores = pd.read_csv(ROOT / 'notebooks/results/compact_lj_response_rollout/response_seed_results.csv')
    scores = scores[scores.variant.eq('mp2_r16_d2') & scores.kind.eq('reconstruction') & scores.strategy.eq('ae')].copy()
    assert len(scores) == 100 and not scores.duplicated(['seed', 'source', 'frame']).any()
    assert scores.valid.eq(20).all() and scores.total.eq(20).all()
    frames = pd.read_csv(RESULTS / 'frame_rows.csv')
    networks = pd.read_csv(RESULTS / 'network_rows.csv')
    metadata = json.loads((RESULTS / 'metadata.json').read_text())
    assert len(frames) == 16000 and len(networks) == 80
    assert frames.groupby(['source', 'simulation_id']).size().eq(200).all()
    assert not frames.duplicated(['source', 'simulation_id', 'frame']).any()
    assert np.isfinite(frames[['z0', 'z1']]).all().all()
    scores = scores[scores.source.isin(SOURCES)].copy()
    early = pd.read_csv(RESULTS / 'early_response/response_seed_results.csv')
    early = early[early.frame.isin((5, 10))].copy()
    assert len(early) == 30 and set(early.source) == set(SOURCES)
    assert early.total.eq(20).all()
    scores = pd.concat([scores, early], ignore_index=True)
    assert not scores.duplicated(['seed', 'source', 'frame']).any()
    assert scores.groupby(['source', 'seed']).frame.apply(lambda x: set(x) == set(FRAMES)).all()
    assert set(scores.seed) == {3456456, 123, 456, 786, 2026}
    frames = frames[frames.source.isin(SOURCES)].copy()
    networks = networks[networks.source.isin(SOURCES)].copy()
    apply_editorial_style()
    return scores, frames, networks, metadata


def reconstruction(scores):
    grouped = scores.groupby(['source', 'frame']).agg(mean=('p_ratio_r2', 'mean'), sd=('p_ratio_r2', 'std'), valid=('valid', 'sum'), total=('total', 'sum')).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    for source in SOURCES:
        g = grouped[grouped.source.eq(source)].sort_values('frame')
        x, mean, sd = g.frame.to_numpy(), g['mean'].to_numpy(), g.sd.to_numpy()
        ax.plot(x, mean, marker='o', lw=2, color=DATASET_COLORS[source], label=LABELS[source])
        ax.fill_between(x, mean-sd, mean+sd, color=DATASET_COLORS[source], alpha=.12)
    ax.axhline(0, color='#626C72', lw=.7)
    ax.axhline(.4, color='#626C72', lw=.8, ls=':', label='0.4 reporting target')
    lower = min(-.3, float((grouped['mean'] - grouped.sd).min()))
    upper = max(1.05, float((grouped['mean'] + grouped.sd).max()))
    padding = .04 * (upper - lower)
    ax.set(xlabel='Trajectory step', ylabel='Reconstruction p-ratio R²', xticks=[5, 25, 50, 100, 150, 199], title='Shared message-passing AE · 2D latent', xlim=(5, 199), ylim=(lower-padding, upper+padding))
    ax.set_yscale('symlog', linthresh=1, linscale=2)
    ax.set_yticks([-5, -1, 0, .4, 1], labels=['−5', '−1', '0', '0.4', '1'])
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    plt.show()
    return grouped.assign(count=lambda d: d.valid.astype(str)+'/'+d.total.astype(str)).pivot(index='frame', columns='source', values='count').rename(columns=LABELS)


def raw_latents(frames):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for source in SOURCES:
        g = frames[frames.source.eq(source)]
        for _, trajectory in g.groupby('simulation_id'):
            trajectory = trajectory.sort_values('frame')
            axes[0].plot(trajectory.z0, trajectory.z1, color=DATASET_COLORS[source], alpha=.28, lw=.8)
        start, end = g[g.frame.eq(0)], g[g.frame.eq(199)]
        axes[0].scatter(start.z0, start.z1, color=DATASET_COLORS[source], s=15, marker='o')
        axes[0].scatter(end.z0, end.z1, color=DATASET_COLORS[source], s=20, marker='^')
        axes[1].scatter(start.z0, start.z1, color=DATASET_COLORS[source], s=28, alpha=.8, label=LABELS[source])
    for ax in axes:
        ax.set(xlabel='Raw z₀', ylabel='Raw z₁')
    axes[0].set_title('All 60 latent trajectories · ○ initial, △ final')
    axes[1].set_title('Initial latent · one point per network')
    axes[1].legend(frameon=False, fontsize=8)
    plt.show()


def response_on_latents(frames):
    data = frames[frames.frame.ge(1)]
    finite = data[np.isfinite(data.p_ratio)]
    vmin, vmax = np.quantile(finite.p_ratio, [.02, .98])
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    for ax, source in zip(axes, SOURCES):
        g = data[data.source.eq(source)]
        for _, trajectory in g.groupby('simulation_id'):
            trajectory = trajectory.sort_values('frame')
            points = trajectory[['z0', 'z1']].to_numpy()
            segments = np.stack([points[:-1], points[1:]], axis=1)
            line = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=.8, alpha=.8)
            line.set_array(trajectory.p_ratio.to_numpy()[1:])
            ax.add_collection(line)
        ax.autoscale_view()
        ax.set(xlabel='Raw z₀', ylabel='Raw z₁', title=LABELS[source])
    fig.colorbar(line, ax=axes, label='Frame p-ratio · colors clipped at pooled 2–98%', extend='both', shrink=.85)
    fig.suptitle('Physical response along raw latent trajectories · steps 1–199')
    plt.show()
    return data.groupby('source').p_ratio.agg(valid=lambda x: int(np.isfinite(x).sum()), total='size').rename(index=LABELS)


def pearson(x, y):
    pair = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan, len(pair)
    return pair.x.corr(pair.y), len(pair)


def initial_response(networks):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained', sharey=True)
    for col, source in enumerate(SOURCES):
        g = networks[networks.source.eq(source)]
        for row, latent in enumerate(('z0_initial', 'z1_initial')):
            ax = axes[row, col]
            r, n = pearson(g[latent], g.p_ratio_final)
            ax.scatter(g[latent], g.p_ratio_final, color=DATASET_COLORS[source], s=27, alpha=.8)
            ax.axhline(0, color='#626C72', lw=.6)
            ax.set(xlabel=f'Initial z{row}', ylabel='Final physical p-ratio' if col == 0 else '', title=f'{LABELS[source]}\nr = {r:+.2f} · n = {n}/{len(g)}')
    fig.suptitle('Does the initial latent correlate with the eventual response? · step 199')
    plt.show()


def frame_correlations(frames):
    rows = []
    for (source, frame), group in frames[frames.frame.between(1, 199)].groupby(['source', 'frame']):
        for latent in ('z0', 'z1'):
            r, n = pearson(group[latent], group.p_ratio)
            rows.append(dict(source=source, frame=frame, latent=latent, r=r, valid=n, total=len(group)))
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained', sharey=True)
    for ax, source in zip(axes, SOURCES):
        for latent, style in [('z0', '-'), ('z1', '--')]:
            g = table[table.source.eq(source) & table.latent.eq(latent)].sort_values('frame')
            ax.plot(g.frame, g.r, style, color=DATASET_COLORS[source], label=latent)
        ax.axhline(0, color='#626C72', lw=.7)
        ax.set(xlabel='Trajectory step', title=LABELS[source], ylim=(-1.05, 1.05), xticks=[1, 50, 100, 150, 199])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Pearson r with frame p-ratio')
    fig.suptitle('Framewise association · compare networks at the same step')
    plt.show()
    return table


def physical_correlations(networks):
    columns = ['p_ratio_final', 'mean_stiffness', 'mean_degree', 'node_count']
    labels = ['Final p-ratio', 'Mean spring k', 'Mean degree', 'Node count']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), layout='constrained', sharey=True)
    rows = []
    for ax, source in zip(axes, SOURCES):
        g = networks[networks.source.eq(source)]
        values = np.full((2, len(columns)), np.nan)
        for i, latent in enumerate(('z0_initial', 'z1_initial')):
            for j, physical in enumerate(columns):
                r, n = pearson(g[latent], g[physical])
                values[i, j] = r
                rows.append(dict(source=source, latent=latent, observable=physical, r=r, valid=n, total=len(g)))
        image = ax.imshow(np.ma.masked_invalid(values), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=[0, 1], yticklabels=['Initial z₀', 'Initial z₁'], title=LABELS[source])
        ax.tick_params(axis='x', rotation=45)
        for i in range(2):
            for j in range(len(columns)):
                label = f'{values[i,j]:+.2f}' if np.isfinite(values[i,j]) else '—'
                ax.text(j, i, label, ha='center', va='center', fontsize=9)
    fig.colorbar(image, ax=axes, label='Pearson r', shrink=.8)
    fig.suptitle('Initial latent versus network properties · constant observables marked —')
    plt.show()
    return pd.DataFrame(rows)


def frame_physics(frames):
    columns, labels = ['p_ratio', 'strain_x', 'strain_y'], ['p-ratio', 'Axial strain', 'Transverse strain']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), layout='constrained', sharey=True)
    for ax, source in zip(axes, SOURCES):
        g = frames[frames.source.eq(source) & frames.frame.eq(100)]
        pairs = [[pearson(g[latent], g[physical]) for physical in columns] for latent in ('z0', 'z1')]
        values = np.array([[r for r, n in row] for row in pairs])
        image = ax.imshow(np.ma.masked_invalid(values), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set(xticks=range(3), xticklabels=labels, yticks=[0, 1], yticklabels=['z₀', 'z₁'], title=LABELS[source])
        ax.tick_params(axis='x', rotation=35)
        for i in range(2):
            for j in range(3):
                r, n = pairs[i][j]
                label = f'{r:+.2f}\n{n}/{len(g)}' if np.isfinite(r) else f'—\n{n}/{len(g)}'
                ax.text(j, i, label, ha='center', va='center', fontsize=8)
    fig.colorbar(image, ax=axes, label='Pearson r', shrink=.8)
    fig.suptitle('Latent coordinates versus physical response · fixed step 100 · valid/total pairs')
    plt.show()


def all_frame_response_scatter(frames):
    """Each point pairs a raw latent coordinate with that same frame's response."""
    norm = Normalize(vmin=0, vmax=199)
    for observable, ylabel in [('p_ratio', 'Physical frame p-ratio'), ('strain_y', 'Transverse strain εᵧ')]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
        for col, source in enumerate(SOURCES):
            g = frames[frames.source.eq(source)]
            for row, latent in enumerate(('z0', 'z1')):
                valid = np.isfinite(g[latent]) & np.isfinite(g[observable])
                ax = axes[row, col]
                points = ax.scatter(g.loc[valid, latent], g.loc[valid, observable],
                                    c=g.loc[valid, 'frame'], norm=norm, cmap='viridis',
                                    s=8, alpha=.35, linewidths=0, rasterized=True)
                ax.axhline(0, color='#626C72', lw=.6, zorder=0)
                ax.set(xlabel=f'Raw z{row}', ylabel=ylabel,
                       title=f'{LABELS[source]} · {int(valid.sum()):,}/{len(g):,} frames')
        fig.colorbar(points, ax=axes, label='Trajectory step', shrink=.8)
        fig.suptitle(f'All-frame latent–response scatter · {ylabel}')
        plt.show()
