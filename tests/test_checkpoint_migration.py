"""One-time migration preserves payloads without runtime import aliases."""
import importlib.util
from pathlib import Path
import pickletools
import zipfile

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/migrations/checkpoint_imports.py'
spec = importlib.util.spec_from_file_location('checkpoint_imports', MODULE_PATH)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def test_checkpoint_migration_preserves_source_and_tensor_storage(tmp_path):
    source = tmp_path / 'historical.pt'
    destination = tmp_path / 'current.pt'
    payload = b'\x80\x02clss.latent.models\nNodeDeltaAttentionAutoEncoder\n.'
    storage = bytes(range(256)) * 8
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('checkpoint/data.pkl', payload)
        archive.writestr('checkpoint/data/0', storage)
        archive.writestr('checkpoint/version', b'3\n')
    original = source.read_bytes()
    record = migration.migrate(source, destination)
    assert source.read_bytes() == original
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('checkpoint/data/0') == storage
        globals_ = [arg for op, arg, _ in pickletools.genops(archive.read('checkpoint/data.pkl')) if op.name == 'GLOBAL']
    assert globals_ == ['lss.dynamics.autoencoder NodeDeltaAttentionAutoEncoder']
    assert record['all_other_zip_members_unchanged']
    assert destination.with_suffix('.pt.migration.json').exists()
    with pytest.raises(FileExistsError):
        migration.migrate(source, destination)


def test_propagator_and_retired_imports_are_explicit():
    assert migration.canonical_global('lss.dynamics.models', 'LatentDynamicsMLP') == ('lss.dynamics.propagator', 'LatentDynamicsMLP')
    assert migration.canonical_global('lss.models.hybrid_simulator', 'HybridSimulator') == ('lss.past_experiments.models.hybrid_simulator', 'HybridSimulator')
    assert migration.canonical_global('torch.nn.modules.linear', 'Linear') == ('torch.nn.modules.linear', 'Linear')


def test_framed_pickle_is_rejected_instead_of_corrupted():
    with pytest.raises(ValueError, match='unsupported encoding'):
        migration.rewrite_pickle(b'\x80\x04\x95\x02\x00\x00\x00\x00\x00\x00\x00N.')
