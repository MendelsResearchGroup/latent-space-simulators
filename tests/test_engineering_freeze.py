"""Final design freezing must preserve the explicitly declared physical protocol."""
import json
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from lss.engineering.evaluation import freeze_final_designs, verify_frozen_designs
from lss.engineering.optimization import OptimizationConfig


def result(tmp_path):
    bundle=tmp_path/'bundle.pt';bundle.write_bytes(b'frozen-model')
    return SimpleNamespace(original=torch.zeros(2),optimized=torch.ones(2),
        history=pd.DataFrame([{'step':1,'best_score':-.1}]),log_stiffness=torch.zeros(2),position_q=torch.zeros(2,2),
        network=SimpleNamespace(source='depablo_mixed_temp',index=47),
        model=SimpleNamespace(bundle_path=bundle,bundle_sha256='test-model-identity'),
        config=OptimizationConfig(),initial_score=.2,best_score=-.1,steps=1,stop_reason='test',
        geometry_summary={'periodic':True,'continuous_linear_step_check':True})


def test_freeze_keeps_thermal_protocol_explicit(tmp_path):
    dataset=tmp_path/'data.pt';dataset.write_bytes(b'pristine-data')
    protocol={'protocol':'thermal','frames':200,'thermal_seeds':[786,123,456],'feedback_to_optimizer':False}
    folder=tmp_path/'frozen'
    freeze_final_designs(result(tmp_path),folder,dataset_path=dataset,known_case_refactor_verification=False,
                         verification_protocol=protocol)
    recipe=json.loads((folder/'frozen_recipe.json').read_text())
    assert recipe['verification']==protocol
    assert recipe['geometry_constraints']['periodic']
    assert any(path.endswith('/planarity.py') for path in recipe['source_hashes'])
    assert len(recipe['designs'])==2


def test_reid_verifier_refuses_thermal_recipe_before_physics(tmp_path):
    dataset=tmp_path/'data.pt';dataset.write_bytes(b'pristine-data')
    folder=tmp_path/'frozen'
    freeze_final_designs(result(tmp_path),folder,dataset_path=dataset,known_case_refactor_verification=False,
        verification_protocol={'protocol':'thermal','frames':200,'feedback_to_optimizer':False})
    with pytest.raises(ValueError,match='Reid athermal protocol'):
        verify_frozen_designs(folder,tmp_path/'must-not-run-lammps')
