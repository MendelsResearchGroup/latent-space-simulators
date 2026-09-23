"""Physical post-fit response evaluation; never called during checkpoint selection."""
import math
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.graph import clone_graph


def decode(ae, row, normalized_z, stats):
    batch = torch.zeros(len(row['h_ref']), dtype=torch.long)
    z = normalized_z.reshape(1, -1) * stats['z_std'] + stats['z_mean']
    return ae.decode(z, row['h_ref'], batch)


@torch.no_grad()
def evaluate(ae, model, data, seed, family, context_mode):
    rows = []
    model.eval()
    for row in data['validation']:
        ref = row['physical_reference']
        sides = directional_side_indices_from_box(ref, quantile=.10)
        path = model(row['latent_norm'][0], row['reference_tokens'])
        prediction_kind = 'direct_prediction' if family in ('direct_mlp', 'cubic') else 'autonomous'
        for frame in range(1, 101):
            truth = clone_graph(ref)
            truth.x = truth.x.clone()
            truth.x[:, :2] = row['physical_positions'][frame]
            true_p = float(calc_p_ratio_rollout_sides([ref, truth], -1, side_idx=sides))
            for kind, z in [('reconstruction', row['latent_norm'][frame]), (prediction_kind, path[frame])]:
                output = decode(ae, row, z, data['latent_stats'])
                norm = data['normalizers']
                positions = row['ref_pos'] + (output * norm['target_std'] + norm['target_mean']) * row['span']
                physical = positions.double() * row['half'] + row['center']
                pred = clone_graph(ref)
                pred.x = pred.x.clone()
                pred.x[:, :2] = physical
                pred_p = float(calc_p_ratio_rollout_sides([ref, pred], -1, side_idx=sides))
                dt = truth.x[:, :2] - ref.x[:, :2]
                dp = physical - ref.x[:, :2]
                rows.append(dict(seed=seed, family=family, context_mode=context_mode,
                    source=row['source'], split='validation', original_index=row['original_index'],
                    frame=frame, kind=kind, true_p_ratio=true_p, pred_p_ratio=pred_p,
                    finite_pair=math.isfinite(true_p) and math.isfinite(pred_p),
                    displacement_sse=float((dp-dt).square().sum()), displacement_count=len(dt),
                    displacement_sum_x=float(dt[:,0].sum()), displacement_sum_y=float(dt[:,1].sum()),
                    displacement_square_sum_x=float(dt[:,0].square().sum()),
                    displacement_square_sum_y=float(dt[:,1].square().sum()),
                    p_ratio_method='physical_endpoint_sides_quantile_0.10', initial_frame_only=True))
    return pd.DataFrame(rows)
