#!/usr/bin/env python3
"""Post-fit network-level associations; no fitting or selecting the AE."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "notebooks/results/mp2_latent_analysis"
OUT = RESULTS / "correlation_search"
SOURCES = ("reid", "depablo_low_temp", "lj_noisy")
SEED = 20260909
N_PERM = 20000
LATENTS = ("z0_initial", "z1_initial", "initial_line_coordinate", "z0_step_slope", "z1_step_slope", "latent_path_slope")


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float(xc @ (y-y.mean()) / (xc @ xc))


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def stability(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(x), size=(5000, len(x)))
    xb, yb = x[draws], y[draws]
    xb -= xb.mean(axis=1, keepdims=True)
    yb -= yb.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(xb, axis=1) * np.linalg.norm(yb, axis=1)
    valid = denominator > 0
    bootstrap = (xb[valid] * yb[valid]).sum(axis=1) / denominator[valid]
    loo = [corr(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
    return dict(bootstrap_r_low=float(np.quantile(bootstrap, .025)),
                bootstrap_r_high=float(np.quantile(bootstrap, .975)),
                bootstrap_valid=int(valid.sum()), bootstrap_total=len(draws),
                loo_r_min=float(min(loo)), loo_r_max=float(max(loo)))


def main():
    frames = pd.read_csv(RESULTS / "frame_rows.csv")
    networks = pd.read_csv(RESULTS / "network_rows.csv")
    structure = pd.read_csv(RESULTS / "structure_descriptors/descriptor_rows.csv")
    rows = []
    for (source, ident), traj in frames[frames.source.isin(SOURCES)].groupby(["source", "simulation_id"]):
        traj = traj.sort_values("frame")
        x, y = traj.z0.to_numpy(), traj.z1.to_numpy()
        b = slope(x, y)
        row = dict(source=source, simulation_id=ident,
                   z0_step_slope=slope(traj.frame, x), z1_step_slope=slope(traj.frame, y),
                   latent_path_slope=b, latent_path_fit_r2=corr(x, y)**2,
                   strain_x_final=traj.strain_x.iloc[-1], strain_y_final=traj.strain_y.iloc[-1],
                   p_ratio_strain_fit=-slope(traj.strain_x, traj.strain_y))
        for name, t in (("early", traj[traj.frame.le(99)]), ("late", traj[traj.frame.ge(100)])):
            row[f"latent_path_slope_{name}"] = slope(t.z0, t.z1)
            for latent in ("z0", "z1"):
                row[f"{latent}_step_slope_{name}"] = slope(t.frame, t[latent])
        rows.append(row)
    table = networks[networks.source.isin(SOURCES)].merge(pd.DataFrame(rows), on=["source", "simulation_id"], validate="one_to_one")
    extra = [c for c in structure.columns if c not in table.columns or c in ("source", "simulation_id")]
    table = table.merge(structure[extra], on=["source", "simulation_id"], validate="one_to_one")
    assert len(table) == 60 and table.groupby("source").size().eq(20).all()
    line_info = {}
    for source in SOURCES:
        mask = table.source.eq(source)
        z = table.loc[mask, ["z0_initial", "z1_initial"]].to_numpy()
        center = z.mean(0)
        _, singular, basis = np.linalg.svd(z-center, full_matrices=False)
        direction = basis[0] * np.sign(basis[0, 0])
        table.loc[mask, "initial_line_coordinate"] = (z-center) @ direction
        line_info[source] = dict(center=center.tolist(), direction=direction.tolist(),
                                 variance_fraction=float(singular[0]**2 / (singular**2).sum()))
    descriptors = [c for c in structure.columns if c not in ("source", "simulation_id") and pd.api.types.is_numeric_dtype(structure[c])]
    targets = list(dict.fromkeys(["p_ratio_final", "p_ratio_strain_fit", "strain_x_final", "strain_y_final", "mean_stiffness", "mean_degree", "node_count", *descriptors]))
    rng = np.random.default_rng(SEED)
    search, null_max = [], np.zeros(N_PERM)
    for source in SOURCES:
        g = table[table.source.eq(source)].sort_values("simulation_id")
        # Complete source-wise observables only; constant/undefined columns remain in the audit.
        valid_targets = [c for c in targets if np.isfinite(g[c]).all() and g[c].std() > 1e-12]
        a = g[list(LATENTS)].to_numpy(float)
        b = g[valid_targets].to_numpy(float)
        a = (a-a.mean(0)) / np.linalg.norm(a-a.mean(0), axis=0)
        b = (b-b.mean(0)) / np.linalg.norm(b-b.mean(0), axis=0)
        observed = a.T @ b
        for start in range(0, N_PERM, 500):
            permutations = np.array([rng.permutation(len(g)) for _ in range(500)])
            perm_corr = np.einsum("pni,nj->pij", a[permutations], b)
            null_max[start:start+500] = np.maximum(null_max[start:start+500], np.abs(perm_corr).max(axis=(1, 2)))
        for i, latent in enumerate(LATENTS):
            for target in targets:
                valid = np.isfinite(g[latent]) & np.isfinite(g[target])
                row = dict(source=source, latent=latent, observable=target, valid=int(valid.sum()), total=len(g))
                if target not in valid_targets:
                    row.update(r=np.nan, spearman=np.nan, status="constant_or_incomplete")
                else:
                    j = valid_targets.index(target)
                    row.update(r=float(observed[i, j]), spearman=float(spearmanr(g[latent], g[target]).statistic), status="evaluated")
                search.append(row)
    audit = pd.DataFrame(search)
    audit["familywise_permutation_p"] = [float((1+(null_max >= abs(r)).sum())/(N_PERM+1)) if np.isfinite(r) else np.nan for r in audit.r]
    candidates = []
    initial = audit[audit.source.eq("lj_noisy") & audit.latent.eq("initial_line_coordinate") & audit.status.eq("evaluated")]
    initial = initial.assign(abs_r=lambda x:x.r.abs()).sort_values("abs_r", ascending=False)
    candidates.append(initial.iloc[0].to_dict())
    for source in SOURCES:
        rates = audit[audit.source.eq(source) & audit.latent.isin(["z0_step_slope", "z1_step_slope"]) & audit.observable.eq("p_ratio_final")]
        candidates.append(rates.loc[rates.r.abs().idxmax()].to_dict())
    for row in candidates:
        g = table[table.source.eq(row["source"])].sort_values("simulation_id")
        row.update(stability(g[row["latent"]], g[row["observable"]]))
    pd.DataFrame(candidates).to_csv(OUT / "candidate_relationships.csv", index=False)
    table.to_csv(OUT / "network_analysis.csv", index=False)
    audit.to_csv(OUT / "all_correlations.csv", index=False)
    metadata = dict(model_seed=786, sampling_unit="network, 20 per source", source_names=list(SOURCES),
                    frames_for_slopes=[0, 199], slope_definition="OLS latent coordinate versus trajectory step, with intercept; latent_path_slope is OLS z1 versus z0",
                    physical_response="p_ratio_final: established endpoint physical side estimator at199; p_ratio_strain_fit: minus OLS strain_y versus strain_x over frames0..199, descriptive alternative only",
                    initial_line=line_info, candidate_latents=list(LATENTS), candidate_observables=targets,
                    permutation_seed=SEED, permutations=N_PERM,
                    multiplicity="max absolute Pearson correlation across all tested source/latent/observable pairs; independent within-source network permutations",
                    final_test_used=False, new_training=False, exploratory=True,
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RESULTS/"frame_rows.csv", RESULTS/"network_rows.csv", RESULTS/"structure_descriptors/descriptor_rows.csv")},
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for source in SOURCES:
        print(source)
        g = audit[audit.source.eq(source)]
        print(g[g.latent.isin(["z0_initial", "z1_initial", "initial_line_coordinate"])].assign(abs_r=lambda x:x.r.abs()).sort_values("abs_r", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
