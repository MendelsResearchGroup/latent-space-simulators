#!/usr/bin/env python3
"""Frozen tiny edge-distribution AE pilot; p-ratio is post-fit evaluation only."""
from __future__ import annotations

import argparse, copy, hashlib, importlib.util, json, math, os, random, resource, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import directional_side_indices_from_box
from scipy.stats import pearsonr, spearmanr

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
FROZEN = ROOT / "notebooks/results/all4_coverage_1000/code_v5"
OUT = ROOT / "notebooks/results/tiny_edge_mlp"
SOURCES = ("node_low", "stiffness_low")
TRAIN_IDS = [0]
VAL_IDS = {"node_low": list(range(20, 28)), "stiffness_low": list(range(20, 27))}
HELDOUT_IDS = list(range(40, 120))
HORIZONS = (2, 5, 10, 20, 33, 50, 99, 199)
BATCH_GRAPHS = 32

from lss.dynamics.experiment import resolve_train_val_test, seed_everything
from lss.dynamics.simulation import batch_delta_graphs


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_spec(manifest: dict, train_ids_by_source: dict[str, list[int]], val_ids_by_source: dict[str, list[int]]) -> dict:
    components = []
    for source in SOURCES:
        train_ids, val_ids = train_ids_by_source[source], val_ids_by_source[source]
        components.append(dict(name=source, label=source, path=manifest["datasets"][source]["trainval_path"],
            train_count=len(train_ids), val_count=len(val_ids), split_indices=dict(train=train_ids, val=val_ids, test=[])))
    return dict(name="tiny_edge_joint_low", label="tiny_edge_joint_low", path=components[0]["path"],
        dataset_name="tiny_edge_joint_low", source_name="tiny_edge_joint_low",
        coordinate_normalization="position_normalization", dataset_mixture=components)


def params_for(spec: dict) -> dict:
    # The resolver requires the established normalization/split recipe, not an AE construction.
    base = json.loads((ROOT / "notebooks/results/mini_shared/joint_original/recipe.json").read_text())["config"]
    params = copy.deepcopy(base)
    params.update(name=spec["name"], label=spec["label"], path=spec["path"], dataset_name=spec["dataset_name"],
        source_name=spec["source_name"], dataset_mixture=spec["dataset_mixture"], node_feature_mode="normalized_delta",
        ae_target_mode="normalized_delta", edge_mode="compact_stored", pos_dim=2)
    return params


def rows(sims): return [(i, frame) for i in range(len(sims)) for frame in range(200)]


def stats(sims, frame_rows):
    edge, pos, target = [], [], []
    for start in range(0, len(frame_rows), BATCH_GRAPHS):
        b = batch_delta_graphs(sims, frame_rows[start:start+BATCH_GRAPHS], pos_dim=2, device="cpu",
            node_feature_mode="normalized_delta", edge_mode="compact_stored")
        # Match TinyEdgeAutoencoder.edge_inputs, including the reversed direction.
        edge.append(torch.cat((torch.cat((b["ref_edge_attr"], b["edge_attr"]), 1),
                               torch.cat((b["ref_edge_attr"] * torch.tensor([-1., -1., 1., 1.]),
                                          b["edge_attr"] * torch.tensor([-1., -1., 1., 1.])), 1)), 0))
        pos.append(b["ref_pos"]); target.append(b["normalized_delta"])
    def mean_std(values):
        values = torch.cat(values, 0); return values.mean(0), values.std(0).clamp_min(1e-6)
    return (*mean_std(edge), *mean_std(pos), *mean_std(target))


def pack(model, sims, part, target_mean, target_std):
    b = batch_delta_graphs(sims, part, pos_dim=2, device="cpu", node_feature_mode="normalized_delta", edge_mode="compact_stored")
    prediction, z = model(b["ref_edge_attr"], b["edge_attr"], b["ref_pos"], b["edge_index"], b["batch"])
    target = (b["normalized_delta"] - target_mean) / target_std
    return prediction, target, z, b


def graph_loss(prediction, target, batch, target_mean, target_std):
    # The normalized-displacement target is standardized only with train-only target statistics.
    sq = (prediction - target).square().mean(1)
    n = int(batch.max()) + 1
    totals = sq.new_zeros(n).index_add_(0, batch, sq)
    counts = torch.bincount(batch, minlength=n).to(sq.dtype)
    return (totals / counts).mean()


@torch.no_grad()
def validation_loss(model, sims, frame_rows, target_mean, target_std):
    model.eval(); total = 0.0; count = 0
    for start in range(0, len(frame_rows), BATCH_GRAPHS):
        part = frame_rows[start:start+BATCH_GRAPHS]; p, y, _, b = pack(model, sims, part, target_mean, target_std)
        total += float(graph_loss(p, y, b["batch"], target_mean, target_std)) * len(part); count += len(part)
    return total / count


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2: return float("nan"), float("nan"), int(good.sum())
    return float(pearsonr(x[good], y[good]).statistic), float(spearmanr(x[good], y[good]).statistic), int(good.sum())


def scalar_r2(table):
    good = np.isfinite(table.true_p_ratio) & np.isfinite(table.pred_p_ratio)
    if good.sum() < 2: return float("nan"), int(good.sum())
    y, p = table.loc[good, "true_p_ratio"].to_numpy(), table.loc[good, "pred_p_ratio"].to_numpy()
    sst = ((y-y.mean())**2).sum()
    return (float(1-((p-y)**2).sum()/sst) if sst > 0 else float("nan")), int(good.sum())


def json_state(model):
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True, choices=(123,456,786)); args = parser.parse_args()
    started = time.monotonic(); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8"))); seed_everything(args.seed)
    folder = OUT / f"seed{args.seed}"; folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((FROZEN / "data_manifest.json").read_text())
    for source in SOURCES: assert sha(Path(manifest["datasets"][source]["trainval_path"])) == manifest["datasets"][source]["trainval_sha256"]
    train_spec = source_spec(manifest, {s: TRAIN_IDS for s in SOURCES}, VAL_IDS); params = params_for(train_spec)
    dump(folder / "recipe.json", dict(study="tiny_edge_mlp", seed=args.seed, job_id=os.environ.get("PBS_JOBID"), train_ids=TRAIN_IDS, validation_ids=VAL_IDS, heldout_ids=HELDOUT_IDS, sources=list(SOURCES), frames="all 200 frames at 1000 MD steps", edge_mode="compact_stored", target="normalized_delta standardized with train-only target statistics", objective="graph-equal standardized normalized displacement MSE", optimizer="Adam lr=1e-3 weight_decay=1e-5", max_epochs=300, patience=30, min_delta=1e-8, checkpoint_selection="validation position/displacement loss only", p_ratio="post-fit physical endpoint estimator only", code_v5_sha256=sha(FROZEN / "manifest.json"), data_manifest_sha256=sha(FROZEN / "data_manifest.json"), tiny_model_sha256=sha(Path(os.environ.get("LSS_TINY_MODEL_PATH", ROOT / "src/lss/dynamics/tiny_edge_ae.py")))))
    train, validation, _, split_info = resolve_train_val_test(train_spec, params, split_seed=123)
    assert len(train) == 2 and len(validation) == 15 and all(len(s) == 200 for s in train + validation)
    train_rows, val_rows = rows(train), rows(validation)
    feature_mean, feature_std, position_mean, position_std, target_mean, target_std = stats(train, train_rows)
    tiny_path = Path(os.environ.get("LSS_TINY_MODEL_PATH", ROOT / "src/lss/dynamics/tiny_edge_ae.py"))
    tiny = load_module("tiny_edge_ae_frozen", tiny_path)
    model = tiny.TinyEdgeAutoencoder(); assert sum(p.numel() for p in model.parameters()) == 268
    model.set_normalization(feature_mean, feature_std, position_mean, position_std)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = validation_loss(model, validation, val_rows, target_mean, target_std); best_state = copy.deepcopy(model.state_dict()); stale = 0
    history = [dict(epoch=0, validation_normalized_displacement_mse=best)]
    for epoch in range(1, 301):
        random.shuffle(train_rows); model.train(); losses = []
        for start in range(0, len(train_rows), BATCH_GRAPHS):
            prediction, target, _, batch = pack(model, train, train_rows[start:start+BATCH_GRAPHS], target_mean, target_std)
            loss = graph_loss(prediction, target, batch["batch"], target_mean, target_std)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach()))
        score = validation_loss(model, validation, val_rows, target_mean, target_std)
        history.append(dict(epoch=epoch, train_graph_equal_normalized_displacement_mse=float(np.mean(losses)), validation_normalized_displacement_mse=score))
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False); print(json.dumps(history[-1]), flush=True)
        if score < best - 1e-8: best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= 30: break
    model.load_state_dict(best_state); model.eval()
    torch.set_grad_enabled(False)
    torch.save(dict(seed=args.seed, state_dict=best_state, target_mean=target_mean, target_std=target_std), folder / "model.pt")
    dump(folder / "model_exact_map.json", dict(architecture="TinyEdgeAutoencoder edge MLP 8->8->2; mean over orientation-symmetrized edges; decoder node MLP 8->16->2", feature_order=["ref_dx", "ref_dy", "ref_length", "ref_stiffness", "delta_dx", "delta_dy", "delta_length", "relative_delta_length"], reversed_orientation="negate dx/dy in reference and delta blocks only", position_features=["reference_x", "reference_y"], state_dict=json_state(model), target_mean=target_mean.tolist(), target_std=target_std.tolist(), parameter_count=268))
    dump(folder / "split_info.json", split_info)
    evaluator = load_module("frozen_v5_evaluate", FROZEN / "evaluate.py")
    test_spec = source_spec(manifest, {s: [] for s in SOURCES}, {s: HELDOUT_IDS for s in SOURCES}); _, heldout, _, _ = resolve_train_val_test(test_spec, params, split_seed=123)
    for source in SOURCES:
        subset = [sim for sim in heldout if str(sim[0].source_name) == source]; assert len(subset) == 80
        for sim, identity in zip(subset, HELDOUT_IDS):
            for graph in sim: graph.source_sim_id = identity
    # Untrained control uses exactly the fitted training normalizers; no response values guide either coordinate.
    random_model = tiny.TinyEdgeAutoencoder(); random_model.load_state_dict(initial_state); random_model.eval()
    latent_rows, reconstruction_rows = [], []
    for source in SOURCES:
        sims = [sim for sim in heldout if str(sim[0].source_name) == source]
        for trajectory_id, sim in zip(HELDOUT_IDS, sims):
            reference, _ = evaluator._pref(sim[0]); sides = directional_side_indices_from_box(reference, quantile=.10)
            final_truth = evaluator.physical(sim[199], sim[0]); final_p = evaluator._p_ratio(reference, final_truth, sides)
            for label, ae in (("trained", model), ("untrained", random_model)):
                b = batch_delta_graphs([sim], [(0,0)], pos_dim=2, device="cpu", node_feature_mode="normalized_delta", edge_mode="compact_stored")
                z = ae.encode(b["ref_edge_attr"], b["edge_attr"], b["edge_index"], b["batch"]).reshape(-1)
                latent_rows.append(dict(seed=args.seed, model=label, source=source, trajectory_id=trajectory_id, z0=float(z[0]), z1=float(z[1]), final_endpoint_p_ratio=final_p))
            for frame in HORIZONS:
                p, y, _, b = pack(model, [sim], [(0,frame)], target_mean, target_std)
                predicted = sim[0].clone(); predicted.x = predicted.x.clone()
                predicted.x[:, :2] = sim[0].x[:, :2] + (p * target_std + target_mean) * b["position_scale"]
                reconstruction_rows.append(dict(seed=args.seed, source=source, trajectory_id=trajectory_id, frame=frame,
                    true_p_ratio=evaluator._p_ratio(reference, evaluator.physical(sim[frame], sim[0]), sides),
                    pred_p_ratio=evaluator._p_ratio(reference, evaluator.physical(predicted, sim[0]), sides),
                    normalized_displacement_mse=float(((p * target_std + target_mean - (y * target_std + target_mean)).square().mean()))))
    latent, reconstruction = pd.DataFrame(latent_rows), pd.DataFrame(reconstruction_rows)
    latent.to_csv(folder / "initial_latents.csv", index=False); reconstruction.to_csv(folder / "reconstruction_endpoint_rows.csv", index=False)
    corr_rows = []
    for label in ("trained", "untrained"):
        for source in (*SOURCES, "pooled"):
            t = latent.loc[(latent.model == label) & (True if source == "pooled" else latent.source == source)]
            for coordinate in ("z0", "z1"):
                pearson, spearman, valid = corr(t[coordinate], t.final_endpoint_p_ratio)
                corr_rows.append(dict(seed=args.seed, model=label, source=source, coordinate=coordinate, pearson=pearson, spearman=spearman, valid=valid, total=len(t)))
    pd.DataFrame(corr_rows).to_csv(folder / "initial_latent_correlations.csv", index=False)
    summary = []
    for (source, frame), t in reconstruction.groupby(["source", "frame"]):
        r2, valid = scalar_r2(t); summary.append(dict(seed=args.seed, source=source, frame=frame, ae_reconstruction_p_ratio_r2=r2, p_ratio_valid=valid, p_ratio_total=len(t), normalized_displacement_mse=t.normalized_displacement_mse.mean()))
    for frame, t in reconstruction.groupby("frame"):
        r2, valid = scalar_r2(t); summary.append(dict(seed=args.seed, source="pooled", frame=frame, ae_reconstruction_p_ratio_r2=r2, p_ratio_valid=valid, p_ratio_total=len(t), normalized_displacement_mse=t.normalized_displacement_mse.mean()))
    pd.DataFrame(summary).to_csv(folder / "reconstruction_pratio_summary.csv", index=False)
    dump(folder / "completed.json", dict(status="completed", seed=args.seed, best_validation_normalized_displacement_mse=best, epochs=len(history)-1, checkpoint_sha256=sha(folder / "model.pt"), seconds=time.monotonic()-started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, p_ratio_used_for_training_or_selection=False))

if __name__ == "__main__": main()
