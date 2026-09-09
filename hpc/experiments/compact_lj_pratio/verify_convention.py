"""Numerical convention checks using the actual evaluator and raw LJ references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from graph_utils.box import Box
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.data import load_dataset, normalize_trajectory_to_reference_box
from lss.graph import clone_graph
import run_pratio as runner

torch.set_num_threads(1)
x=torch.tensor([[-4.,-1.],[-4.,1.],[4.,-1.],[4.,1.]],dtype=torch.float64)
sim=[Data(x=v,edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,5)),box=Box(-4,4,-1,1,-.1,.1)) for v in [x,x*torch.tensor([1.01,.988],dtype=torch.float64)]]
y=calc_p_ratio_rollout_sides(sim,-1)
normalize_trajectory_to_reference_box(sim,pos_dim=2)
pred=clone_graph(sim[1]);assert not hasattr(pred,'reference_box_half_extent')
restored=[runner.physical(sim[0],sim[0]),runner.physical(pred,sim[0])]
assert abs(calc_p_ratio_rollout_sides(restored,-1)-y)<1e-10
print('Actual-runner synthetic inverse and minimal decoded graph: PASS',flush=True)
r=json.loads((runner.ROOT/'notebooks/results/compact_lj_reconstruction/r16_d4_s123_code_v2/recipe.json').read_text())
s=next(s for s in r['source']['dataset_mixture'] if s['name']=='lj_noisy')
data=load_dataset(runner.ROOT/'data'/Path(s['path']).name,coordinate_normalization=None,pos_dim=2)
records=[]
for i in s['split_indices']['val']:
    raw=[data[i][f].clone() for f in [0,*runner.FRAMES]]
    for g in raw:g.x=g.x.double()
    expected=[calc_p_ratio_rollout_sides(raw,j) for j in range(1,6)]
    normalized=[data[i][f].clone() for f in [0,*runner.FRAMES]]
    normalize_trajectory_to_reference_box(normalized,pos_dim=2)
    ref=normalized[0];pref=runner.physical(ref,ref)
    pref.x[:,:2]=ref.reference_context_positions.double()[:,:2]
    sides=directional_side_indices_from_box(pref)
    for j,frame in enumerate(runner.FRAMES,1):
        actual=calc_p_ratio_rollout_sides([pref,runner.physical(normalized[j],ref)],-1,side_idx=sides)
        records.append({'original_index':i,'frame':frame,'raw':expected[j-1],'inverse':actual,'difference':actual-expected[j-1]})
output=runner.ROOT/'notebooks/results/compact_lj_pratio/raw_target_audit'
output.mkdir(parents=True,exist_ok=True)
table=pd.DataFrame(records);table.to_csv(output/'lj_inverse_parity.csv',index=False)
maximum=float(table.difference.abs().max())
print('LJ exact-reference side groups, maximum raw/inverse p-ratio difference:',maximum,flush=True)
assert maximum < 1e-4
(output/'convention_checks.json').write_text(json.dumps({'synthetic_pass':True,'minimal_decoded_graph_pass':True,
    'lj_max_abs_raw_inverse_pratio_difference':maximum,'lj_rows':len(table),
    'script_sha256':runner.sha(Path(__file__)),'runner_sha256':runner.sha(Path(runner.__file__))},indent=2)+'\n')
