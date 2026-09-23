"""Physical response on forecast frames only, after the declared observed prefix."""
import math
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.graph import clone_graph
from lss.dynamics.observed_history_propagator import linear_history_path


def decode(ae, row, normalized_z, stats):
    batch = torch.zeros(len(row['h_ref']), dtype=torch.long)
    z = normalized_z.reshape(1,-1)*stats['z_std']+stats['z_mean']
    return ae.decode(z,row['h_ref'],batch)


@torch.no_grad()
def evaluate(ae,model,data,seed,family,observed_frames):
    rows=[]
    model.eval()
    for row in data['validation']:
        reference=row['physical_reference']
        sides=directional_side_indices_from_box(reference,quantile=.10)
        path=model(row['latent_norm'][:observed_frames],row['reference_tokens'])
        true_z = row['latent_norm']
        teacher = model.teacher_path(true_z,row['reference_tokens'])
        persistence = torch.cat((true_z[:3],true_z[2:100]))
        velocity = torch.cat((true_z[:3],true_z[2:100]+(true_z[2:100]-true_z[:98])/2))
        trajectories=[('reconstruction',true_z),('prefix_prediction',path),
                      ('teacher_one_step',teacher),('persistence_one_step',persistence),
                      ('velocity_one_step',velocity)]
        for frame in range(3,101):
            truth=clone_graph(reference)
            truth.x=truth.x.clone()
            truth.x[:,:2]=row['physical_positions'][frame]
            true_p=float(calc_p_ratio_rollout_sides([reference,truth],-1,side_idx=sides))
            for kind,trajectory in trajectories:
                output=decode(ae,row,trajectory[frame],data['latent_stats'])
                norm=data['normalizers']
                positions=row['ref_pos']+(output*norm['target_std']+norm['target_mean'])*row['span']
                physical=positions.double()*row['half']+row['center']
                pred=clone_graph(reference)
                pred.x=pred.x.clone()
                pred.x[:,:2]=physical
                pred_p=float(calc_p_ratio_rollout_sides([reference,pred],-1,side_idx=sides))
                true_disp=truth.x[:,:2]-reference.x[:,:2]
                pred_disp=physical-reference.x[:,:2]
                increment = true_z[frame]-true_z[frame-1]
                # For free rollout this is the generated increment, not a teacher step.
                predicted_increment = trajectory[frame]-(trajectory[frame-1] if kind=='prefix_prediction' else true_z[frame-1])
                rows.append(dict(latent_increment_sse=float((predicted_increment-increment).square().sum()),
                    latent_increment_sum_x=float(increment[0]),latent_increment_sum_y=float(increment[1]),
                    latent_increment_square_sum_x=float(increment[0].square()),
                    latent_increment_square_sum_y=float(increment[1].square()),seed=seed,family=family,observed_frames=observed_frames,
                    source=row['source'],split='validation',original_index=row['original_index'],
                    frame=frame,forecast_steps=frame-(observed_frames-1),kind=kind,
                    true_p_ratio=true_p,pred_p_ratio=pred_p,
                    finite_pair=math.isfinite(true_p) and math.isfinite(pred_p),
                    displacement_sse=float((pred_disp-true_disp).square().sum()),displacement_count=len(true_disp),
                    displacement_sum_x=float(true_disp[:,0].sum()),displacement_sum_y=float(true_disp[:,1].sum()),
                    displacement_square_sum_x=float(true_disp[:,0].square().sum()),
                    displacement_square_sum_y=float(true_disp[:,1].square().sum()),
                    p_ratio_method='physical_endpoint_sides_quantile_0.10'))
    return pd.DataFrame(rows)
