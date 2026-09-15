import importlib.util
import os
from pathlib import Path

import torch
import lss.data


MODULE = Path(__file__).parents[1] / "hpc/experiments/mini_temperature200_frames500_size_weighted/train.py"
os.environ.setdefault("LSS_PROJECT_ROOT", str(Path(__file__).parents[1]))
SPEC = importlib.util.spec_from_file_location("mini_size_weighted_train", MODULE)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def test_inverse_energy_is_training_size_only_capped_and_mean_normalized(monkeypatch):
    # Avoid graph construction: this verifies the declared E -> weight math.
    energies = iter([torch.tensor(0.01), torch.tensor(10.0), torch.tensor(1000.0)])
    monkeypatch.setattr(STUDY, "FRAMES", 4)
    monkeypatch.setattr(STUDY, "batch_delta_graphs", lambda *args, **kwargs: {})
    monkeypatch.setattr(STUDY, "ae_target_tensor", lambda *args, **kwargs: torch.tensor([[next(energies).sqrt()]]))
    params = {"node_feature_mode": "normalized_delta", "edge_mode": "compact_stored", "ae_target_mode": "normalized_delta"}
    # A nonzero target mean verifies that E uses displacement/target_std, not centered target.
    norm = {"target_mean": torch.tensor([[99.0]]), "target_std": torch.tensor([[1.0]])}
    energy, weights, epsilon = STUDY.frame_energy_and_weights([[object()]], params, norm, "inverse_energy_cap100")
    torch.testing.assert_close(energy, torch.tensor([0.01, 10.0, 1000.0]))
    assert epsilon == 0.1  # median positive E / 100
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    # E_1 is floored at epsilon, so its raw inverse weight is capped at 100x E_2.
    assert torch.isclose(weights[0] / weights[1], torch.tensor(100.0))


def test_uniform_weights_exclude_reference_frame(monkeypatch):
    energies = iter([torch.tensor(2.0), torch.tensor(3.0)])
    monkeypatch.setattr(STUDY, "FRAMES", 3)
    monkeypatch.setattr(STUDY, "batch_delta_graphs", lambda *args, **kwargs: {})
    monkeypatch.setattr(STUDY, "ae_target_tensor", lambda *args, **kwargs: torch.tensor([[next(energies).sqrt()]]))
    params = {"node_feature_mode": "normalized_delta", "edge_mode": "compact_stored", "ae_target_mode": "normalized_delta"}
    norm = {"target_mean": torch.tensor([[0.0]]), "target_std": torch.tensor([[1.0]])}
    energy, weights, _ = STUDY.frame_energy_and_weights([[object()]], params, norm, "uniform")
    assert len(energy) == len(weights) == 2
    torch.testing.assert_close(weights, torch.ones(2))


def test_source_spec_uses_explicit_mixture_indices_required_by_loader():
    manifest = {"datasets": {"stiffness_low": {"cache_path": "/tmp/example.pt"}}}
    spec = STUDY.source_spec(manifest, smoke=False)
    member = spec["dataset_mixture"]
    assert spec["train_count"] == member[0]["train_count"] == 20
    assert spec["val_count"] == member[0]["val_count"] == 20
    assert member[0]["split_indices"] == {"train": list(range(20)), "val": list(range(20, 40)), "test": []}


def test_actual_split_resolver_honors_source_mixture_indices(monkeypatch):
    # This invokes the production resolve_train_val_test -> resolve_dataset_splits path.
    simulations = [[type("Graph", (), {"source_sim_id": index})()] for index in range(120)]
    monkeypatch.setattr(lss.data, "load_dataset", lambda *args, **kwargs: simulations)
    manifest = {"datasets": {"stiffness_low": {"cache_path": "/tmp/example.pt"}}}
    spec = STUDY.source_spec(manifest, smoke=False)
    cfg = STUDY.config(Path("/tmp/size-weighted-test"), 123)
    train, validation, heldout, info = STUDY.resolve_train_val_test(spec, {**cfg, **spec}, split_seed=123)
    assert [sim[0].source_sim_id for sim in train] == list(range(20))
    assert [sim[0].source_sim_id for sim in validation] == list(range(20, 40))
    assert heldout == []
    assert info[0]["excluded"] == 80
