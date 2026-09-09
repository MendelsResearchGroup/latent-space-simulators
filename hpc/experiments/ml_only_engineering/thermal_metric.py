"""Frozen mixed-T raw baseline for the predeclared frame-time OLS p-ratio."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DONOR = Path("/rg/mendels_prj/alexander.z/DL-course-project")
sys.path[:0] = [str(DONOR.parent / "MetaForge/src")]
from legacy_graph_utils import directional_side_indices_from_box

OUT = ROOT / "notebooks/results/thermal_engineering_audit"
DATA = DONOR / "data/depablo-10k-mix-temp.pt"
SELECTED = {47: 0.1, 68: 1.0, 140: 5.0, 161: 10.0, 242: 20.0, 268: 30.0}
EPS = 1e-12


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ols_numpy(time: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Return intercept, slope and ordinary fitted-line R2."""
    design = np.column_stack([np.ones(len(time)), time])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coef
    total = float(np.square(values - values.mean()).sum())
    r2 = float(1.0 - np.square(values - fitted).sum() / total) if total > EPS else float("nan")
    return float(coef[0]), float(coef[1]), r2


def frame_time_ols_p_ratio(
    physical_positions: np.ndarray,
    sides: dict[str, np.ndarray],
    frame_index: np.ndarray | None = None,
    *,
    min_frames: int = 8,
    min_axial_total_strain: float = 1e-3,
) -> dict:
    """Known-x-driven, two-intercept OLS estimator from physical positions."""
    pos = np.asarray(physical_positions, dtype=float)
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError("physical_positions must be [frames,nodes,2]")
    time = np.arange(len(pos), dtype=float) if frame_index is None else np.asarray(frame_index, dtype=float)
    if time.shape != (len(pos),):
        raise ValueError("frame_index must have one value per frame")
    width = pos[:, sides["right"], 0].mean(1) - pos[:, sides["left"], 0].mean(1)
    height = pos[:, sides["top"], 1].mean(1) - pos[:, sides["bottom"], 1].mean(1)
    finite = np.isfinite(time) & np.isfinite(width) & np.isfinite(height)
    valid = int(finite.sum())
    result = dict(
        p_ratio=float("nan"), total_frames=int(len(pos)), valid_frames=valid,
        valid=False, width=width, height=height,
        strain_x=np.full(len(pos), np.nan), strain_y=np.full(len(pos), np.nan),
        intercept_x=float("nan"), slope_x=float("nan"), fit_r2_x=float("nan"),
        intercept_y=float("nan"), slope_y=float("nan"), fit_r2_y=float("nan"),
        fitted_axial_total_strain=float("nan"),
    )
    if valid < min_frames:
        return result
    first = np.flatnonzero(finite)[0]
    if abs(width[first]) <= EPS or abs(height[first]) <= EPS:
        return result
    strain_x = (width - width[first]) / width[first]
    strain_y = (height - height[first]) / height[first]
    finite = finite & np.isfinite(strain_x) & np.isfinite(strain_y)
    valid = int(finite.sum())
    result.update(valid_frames=valid, valid=False, strain_x=strain_x, strain_y=strain_y)
    if valid < min_frames:
        return result
    ax, bx, r2x = _ols_numpy(time[finite], strain_x[finite])
    ay, by, r2y = _ols_numpy(time[finite], strain_y[finite])
    total_axial = bx * (float(time[finite].max()) - float(time[finite].min()))
    result.update(intercept_x=ax, slope_x=bx, fit_r2_x=r2x, intercept_y=ay, slope_y=by,
                  fit_r2_y=r2y, fitted_axial_total_strain=total_axial)
    if abs(total_axial) >= min_axial_total_strain and abs(bx) > EPS:
        result["p_ratio"] = float(-by / bx)
        result["valid"] = bool(np.isfinite(result["p_ratio"]))
    return result


def frame_time_ols_p_ratio_torch(
    physical_positions: torch.Tensor,
    sides: dict[str, np.ndarray | torch.Tensor],
    frame_index: torch.Tensor | None = None,
    *,
    min_frames: int = 8,
    min_axial_total_strain: float = 1e-3,
) -> dict:
    """Differentiable counterpart for finite physical tensors; returns tensors."""
    pos = physical_positions
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError("physical_positions must be [frames,nodes,2]")
    dtype, device = pos.dtype, pos.device
    time = torch.arange(len(pos), dtype=dtype, device=device) if frame_index is None else frame_index.to(dtype=dtype, device=device)
    left = torch.as_tensor(sides["left"], device=device, dtype=torch.long)
    right = torch.as_tensor(sides["right"], device=device, dtype=torch.long)
    bottom = torch.as_tensor(sides["bottom"], device=device, dtype=torch.long)
    top = torch.as_tensor(sides["top"], device=device, dtype=torch.long)
    width = pos[:, right, 0].mean(1) - pos[:, left, 0].mean(1)
    height = pos[:, top, 1].mean(1) - pos[:, bottom, 1].mean(1)
    finite = torch.isfinite(time) & torch.isfinite(width) & torch.isfinite(height)
    if int(finite.sum()) < min_frames:
        nan = torch.full((), float("nan"), dtype=dtype, device=device)
        return dict(p_ratio=nan, valid=False, valid_frames=int(finite.sum()), width=width, height=height)
    first = int(torch.nonzero(finite, as_tuple=False)[0])
    strain_x = (width - width[first]) / width[first]
    strain_y = (height - height[first]) / height[first]
    finite = finite & torch.isfinite(strain_x) & torch.isfinite(strain_y)
    if int(finite.sum()) < min_frames:
        nan = torch.full((), float("nan"), dtype=dtype, device=device)
        return dict(p_ratio=nan, valid=False, valid_frames=int(finite.sum()), width=width, height=height,
                    strain_x=strain_x, strain_y=strain_y)
    x = time[finite]
    design = torch.stack([torch.ones_like(x), x], 1)
    coef_x = torch.linalg.lstsq(design, strain_x[finite, None]).solution[:, 0]
    coef_y = torch.linalg.lstsq(design, strain_y[finite, None]).solution[:, 0]
    total_axial = coef_x[1] * (x.max() - x.min())
    p = -coef_y[1] / coef_x[1]
    if float(total_axial.detach().abs()) < min_axial_total_strain or float(coef_x[1].detach().abs()) <= EPS:
        p = torch.full((), float("nan"), dtype=dtype, device=device)
    return dict(p_ratio=p, valid=bool(torch.isfinite(p).detach()), valid_frames=int(finite.sum()), width=width, height=height,
                strain_x=strain_x, strain_y=strain_y, intercept_x=coef_x[0], slope_x=coef_x[1],
                intercept_y=coef_y[0], slope_y=coef_y[1], fitted_axial_total_strain=total_axial)


def unwrap_positions(sim: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-image fractional increments, mapped through each evolving box."""
    raw = np.stack([frame.x[:, :2].detach().cpu().numpy().astype(float) for frame in sim])
    lower = np.asarray([[g.box.x1, g.box.y1] for g in sim], dtype=float)
    length = np.asarray([[g.box.x2 - g.box.x1, g.box.y2 - g.box.y1] for g in sim], dtype=float)
    frac = (raw - lower[:, None, :]) / length[:, None, :]
    delta = np.diff(frac, axis=0)
    shifts = np.rint(delta)
    unwrapped_frac = np.concatenate([frac[:1], frac[:1] + np.cumsum(delta - shifts, axis=0)])
    return lower[:, None, :] + unwrapped_frac * length[:, None, :], length, shifts


def synthetic_checks() -> dict:
    sides = {key: np.array([0]) for key in ("left", "right", "bottom", "top")}
    # Four nodes define independent fixed side groups.
    sides = {"left": np.array([0]), "right": np.array([1]), "bottom": np.array([2]), "top": np.array([3])}
    time = np.arange(20, dtype=float)
    pos = np.zeros((20, 4, 2), dtype=float)
    pos[:, 1, 0] = 10.0 * (1 - .01 * time)
    pos[:, 3, 1] = 8.0 * (1 + .003 * time)
    affine = frame_time_ols_p_ratio(pos, sides, time)
    negative_pos=pos.copy();negative_pos[:,3,1]=8.0*(1-.003*time)
    negative=frame_time_ols_p_ratio(negative_pos,sides,time)
    rng = np.random.default_rng(17)
    noisy = pos + rng.normal(0, 1e-4, pos.shape)
    noisy_np = frame_time_ols_p_ratio(noisy, sides, time)
    torch_value = frame_time_ols_p_ratio_torch(torch.tensor(noisy, dtype=torch.float64), sides, torch.tensor(time, dtype=torch.float64))
    degenerate = frame_time_ols_p_ratio(np.repeat(pos[:1], 20, axis=0), sides, time)
    checks = dict(
        negative_affine_expected=-.3, negative_affine_observed=negative["p_ratio"], negative_affine_pass=abs(negative["p_ratio"]+.3)<1e-12,
        degenerate_valid_flag=degenerate["valid"],
        affine_expected=0.3, affine_observed=affine["p_ratio"], affine_pass=abs(affine["p_ratio"] - .3) < 1e-12,
        noisy_numpy=noisy_np["p_ratio"], noisy_torch=float(torch_value["p_ratio"]), numpy_torch_abs_error=abs(noisy_np["p_ratio"] - float(torch_value["p_ratio"])),
        numpy_torch_pass=abs(noisy_np["p_ratio"] - float(torch_value["p_ratio"])) < 1e-10,
        degenerate_is_nan=bool(np.isnan(degenerate["p_ratio"])), degenerate_valid_frames=degenerate["valid_frames"],
    )
    if checks["degenerate_valid_flag"] or not checks["negative_affine_pass"] or not checks["affine_pass"] or not checks["numpy_torch_pass"] or not checks["degenerate_is_nan"]:
        raise AssertionError(checks)
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = synthetic_checks()
    raw = torch.load(DATA, weights_only=False, map_location="cpu", mmap=True)
    rows = []
    for index, temperature in SELECTED.items():
        sim = raw[index]
        if len(sim) != 200 or float(sim[0].temperature) != temperature:
            raise ValueError(f"Unexpected selected mixed-T trajectory {index}")
        physical, box_lengths, shifts = unwrap_positions(sim)
        sides = directional_side_indices_from_box(copy.deepcopy(sim[0]), quantile=.10)
        metric = frame_time_ols_p_ratio(physical, sides)
        times = [getattr(frame, "time", None) for frame in sim]
        finite_times = [float(x) for x in times if x is not None and np.isfinite(float(x))]
        time_steps = np.diff(np.asarray(finite_times, dtype=float)) if len(finite_times) > 1 else np.asarray([])
        rows.append(dict(
            source="depablo_mixed_temp", original_index=index, temperature=temperature,
            p_ratio=metric["p_ratio"], total_frames=metric["total_frames"], valid_frames=metric["valid_frames"],
            estimator_valid=metric["valid"], fitted_axial_total_strain=metric["fitted_axial_total_strain"],
            intercept_x=metric["intercept_x"], slope_x=metric["slope_x"], temporal_fit_r2_x=metric["fit_r2_x"],
            intercept_y=metric["intercept_y"], slope_y=metric["slope_y"], temporal_fit_r2_y=metric["fit_r2_y"],
            initial_box_x=box_lengths[0, 0], final_box_x=box_lengths[-1, 0],
            box_x_ratio=box_lengths[-1, 0] / box_lengths[0, 0], initial_box_y=box_lengths[0, 1],
            final_box_y=box_lengths[-1, 1], box_y_ratio=box_lengths[-1, 1] / box_lengths[0, 1],
            wrap_corrections=int(np.count_nonzero(shifts)), time_metadata_count=len(finite_times),
            time_metadata_first=finite_times[0] if finite_times else np.nan,
            time_metadata_last=finite_times[-1] if finite_times else np.nan,
            time_metadata_step_min=float(time_steps.min()) if len(time_steps) else np.nan,
            time_metadata_step_max=float(time_steps.max()) if len(time_steps) else np.nan,
            time_metadata_uniform=bool(len(time_steps) and np.allclose(time_steps, time_steps[0])),
            frame_index_start=0, frame_index_last=199,
        ))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "raw_linear_fit.csv", index=False)
    recipe = dict(
        source="depablo_mixed_temp", selected_validation_ids=SELECTED,
        split="existing validation IDs only; no final-test rows", frames="all 200 saved frames, index 0..199",
        estimator="two OLS fits with intercept: eps_x=a_x+b_x*t; eps_y=a_y+b_y*t; p=-b_y/b_x; known x-driven",
        side_groups="fixed original physical frame0 directional quantile=0.10",
        coordinates="minimum-image fractional increments then map through each evolving physical box",
        validity="all finite frames; >=8 frames; abs(fitted axial total strain)>=1e-3; otherwise NaN",
        time="saved frame index; graph.time metadata retained separately but not used as the fit coordinate",
        dataset=str(DATA), dataset_sha256=sha(DATA), script_sha256=sha(Path(__file__)),
        selected_from="notebooks/results/temporal_linearity_check/recipe.json",
    )
    (OUT / "raw_linear_fit_recipe.json").write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
    (OUT / "metric_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
