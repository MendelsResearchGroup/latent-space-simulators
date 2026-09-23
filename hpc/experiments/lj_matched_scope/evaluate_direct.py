#!/usr/bin/env python3
"""Post-fit evaluation of direct frame-zero AE perturbation edges."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "notebooks/results").is_dir() and (parent / "src").is_dir())
OUT = ROOT / "notebooks/results/lj_matched_scope"
CODE = Path(__file__).resolve().parent
os.environ["LSS_PROJECT_ROOT"] = str(ROOT)
os.environ["LSS_CODE_ROOT"] = str(CODE)
runner_spec = importlib.util.spec_from_file_location("direct_edge_runner", CODE / "run_direct.py")
assert runner_spec is not None and runner_spec.loader is not None
runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
pilot = runner.pilot
HORIZONS = tuple(range(10, 101, 10))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def fitted_response(path: torch.Tensor, sides: dict, horizon: int) -> dict:
    frames = np.arange(horizon + 1, dtype=float)
    values = path[:horizon + 1].detach().cpu().double()
    width = values[:, sides["right"], 0].mean(1) - values[:, sides["left"], 0].mean(1)
    height = values[:, sides["top"], 1].mean(1) - values[:, sides["bottom"], 1].mean(1)
    strain_x = ((width - width[0]) / width[0]).numpy()
    strain_y = ((height - height[0]) / height[0]).numpy()
    design = np.column_stack((np.ones(len(frames)), frames))
    coef_x = np.linalg.lstsq(design, strain_x, rcond=None)[0]
    coef_y = np.linalg.lstsq(design, strain_y, rcond=None)[0]
    fit_x, fit_y = design @ coef_x, design @ coef_y
    axial = float(coef_x[1] * horizon)
    valid = np.isfinite(coef_x).all() and np.isfinite(coef_y).all() and abs(axial) >= 1e-3
    def r2(observed, fitted):
        denominator = np.square(observed - observed.mean()).sum()
        return float(1 - np.square(observed - fitted).sum() / denominator) if denominator > 0 else np.nan
    return {"p_ratio": float(-coef_y[1] / coef_x[1]) if valid else np.nan,
            "valid": valid, "fit_r2_x": r2(strain_x, fit_x), "fit_r2_y": r2(strain_y, fit_y)}


def response_r2(truth, prediction):
    denominator = np.square(truth - truth.mean()).sum()
    return float(1 - np.square(truth - prediction).sum() / denominator) if denominator > 0 else np.nan


def validation_records(data: dict, ae, latent_model, arm: str) -> list[dict]:
    """Build only held-out records; training records were already frozen by each GNN job."""
    records = []
    for row in data["validation"]:
        positions = pilot.normalized_positions(row)
        base, base_stiffness = pilot.base_relation(row)
        if arm == "latent":
            score = pilot.latent_score(ae, latent_model, row, data)
            extra = pilot.select_topk_nonphysical_edges(score, base, 5)
            latent_score_mean = float(score[extra[0], extra[1]].mean())
        else:
            score = None
            extra = base[:, :0]
            latent_score_mean = np.nan
        index = torch.cat((base, extra), dim=1)
        records.append({
            "original_index": int(row["original_index"]),
            "positions": positions,
            "row": row,
            "edge_index": index,
            "stiffness": torch.cat((base_stiffness, torch.zeros(extra.shape[1]))),
            "is_base": torch.cat((torch.ones(base.shape[1]), torch.zeros(extra.shape[1]))),
            "is_added": torch.cat((torch.zeros(base.shape[1]), torch.ones(extra.shape[1]))),
            "base_edge_count": int(base.shape[1]),
            "added_edge_count": int(extra.shape[1]),
            "latent_edge_count": int(extra.shape[1]),
            "latent_score_mean": latent_score_mean,
            "_score": score,
        })
    return records


def main() -> None:
    manifest = json.loads((CODE / "manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    rows, added_rows, graph_rows, node_rows = [], [], [], []
    for scope in ("ljonly4", "shared4"):
        for seed in (123, 456, 786):
            latent_dir = OUT / "latent" / f"{scope}_s{seed}" / "full"
            data = torch.load(latent_dir / "cache.pt", map_location="cpu", weights_only=False)
            ae = runner.load_ae(data)
            pilot.latent_score = runner.direct_score
            validation = validation_records(data, ae, None, "latent")
            for record in validation:
                score = record["_score"]
                start = record["base_edge_count"]
                extra = record["edge_index"][:, start:]
                position = record["positions"][0]
                for source, target in extra.t().tolist():
                    added_rows.append({"scope": scope, "seed": seed,
                        "original_index": record["original_index"],
                        "source_node": source, "target_node": target,
                        "influence_score": float(score[source, target]),
                        "reference_distance": float(torch.linalg.vector_norm(position[target] - position[source]))})
            if seed == 123:
                visual = validation[0]
                position = visual["positions"][0]
                for node, xy in enumerate(position.tolist()):
                    node_rows.append({"scope": scope, "original_index": visual["original_index"],
                                      "node": node, "x": xy[0], "y": xy[1]})
                for relation, edge_type in ((visual["edge_index"][:, :visual["base_edge_count"]], "physical"),
                                            (visual["edge_index"][:, visual["base_edge_count"]:], "latent")):
                    for source, target in relation.t().tolist():
                        graph_rows.append({"scope": scope, "original_index": visual["original_index"],
                                           "edge_type": edge_type, "source_node": source, "target_node": target})
            evaluations = [(f"{scope}_direct", validation, OUT / "runs" / f"{scope}_direct_s{seed}")]
            if scope == "ljonly4":
                base_validation = validation_records(data, ae, None, "base")
                evaluations.append(("base", base_validation,
                    ROOT / "notebooks/results/lj_latent_influence_edges/runs" / f"base_s{seed}"))
            for arm, records, run_dir in evaluations:
                assert json.loads((run_dir / "completed.json").read_text())["status"] == "completed"
                checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
                model = pilot.NormalizedAccelerationGNN(hidden_dim=64, steps=4)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                predictions = pilot.batched_rollout(model, records, checkpoint["normalizers"])
                for record, prediction in zip(records, predictions):
                    reference = record["row"]["physical_reference"]
                    sides = pilot.directional_side_indices_from_box(reference, quantile=.10)
                    truth_physical = record["row"]["physical_positions"]
                    prediction_physical = pilot.physical_positions(record, prediction)
                    for horizon in HORIZONS:
                        truth = fitted_response(truth_physical, sides, horizon)
                        estimate = fitted_response(prediction_physical, sides, horizon)
                        rows.append({"arm": arm, "seed": seed, "split": "validation",
                            "original_index": record["original_index"], "frame": horizon,
                            "true_p_ratio": truth["p_ratio"], "pred_p_ratio": estimate["p_ratio"],
                            "finite_pair": truth["valid"] and estimate["valid"],
                            "true_fit_r2_x": truth["fit_r2_x"], "true_fit_r2_y": truth["fit_r2_y"],
                            "pred_fit_r2_x": estimate["fit_r2_x"], "pred_fit_r2_y": estimate["fit_r2_y"],
                            "coordinate_mse": float((prediction_physical[horizon] - truth_physical[horizon]).square().mean())})
    table = pd.DataFrame(rows); table.to_csv(OUT / "direct_postfit_per_network.csv", index=False)
    pd.DataFrame(added_rows).to_csv(OUT / "direct_validation_edges.csv", index=False)
    pd.DataFrame(node_rows).to_csv(OUT / "direct_visualization_nodes.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(OUT / "direct_visualization_edges.csv", index=False)
    summary = []
    for (arm, seed, frame), group in table.groupby(["arm", "seed", "frame"]):
        valid = group[group.finite_pair]
        summary.append({"arm": arm, "seed": seed, "frame": frame,
            "postfit_p_ratio_r2": response_r2(valid.true_p_ratio.to_numpy(), valid.pred_p_ratio.to_numpy()),
            "p_ratio_valid": len(valid), "p_ratio_total": len(group),
            "coordinate_mse": float(group.coordinate_mse.mean())})
    summary = pd.DataFrame(summary); summary.to_csv(OUT / "direct_postfit_summary.csv", index=False)
    aggregate = summary.groupby(["arm", "frame"], as_index=False).agg(
        postfit_p_ratio_r2_mean=("postfit_p_ratio_r2", "mean"),
        postfit_p_ratio_r2_sd=("postfit_p_ratio_r2", "std"),
        coordinate_mse_mean=("coordinate_mse", "mean"),
        coordinate_mse_sd=("coordinate_mse", "std"),
        p_ratio_valid=("p_ratio_valid", "sum"), p_ratio_total=("p_ratio_total", "sum"))
    aggregate.to_csv(OUT / "direct_postfit_seed_summary.csv", index=False)
    metadata = {"status": "completed", "script_sha256": sha(Path(__file__)),
                "final_test_used": False, "response_used_for_training_or_selection": False,
                "visualization_network": int(pd.DataFrame(node_rows).original_index.iloc[0]),
                "latent_edges_are_unseen_graph_frame0_only": True,
                "base_reuse": "unchanged base rows/checkpoints from lj_latent_influence_edges"}
    (OUT / "direct_evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(aggregate.to_string(index=False))


if __name__ == "__main__": main()
