#!/usr/bin/env python3
"""One bounded ML-only optimization; final physics is intentionally absent here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from lss.engineering.model import load_model, load_network
from lss.engineering.geometry import boundary_constraints, valid_geometry
from lss.engineering.planarity import PlanarConstraints
from lss.engineering.optimization import OptimizationConfig, optimize_network
from lss.engineering.evaluation import freeze_final_designs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(output: Path, cohort: str, index: int) -> None:
    torch.set_num_threads(1)
    output = Path(output)
    recipe = json.loads((output / 'cohort_and_recipe.json').read_text())
    rows = [row for row in recipe['rows'] if row['cohort'] == cohort and row['index'] == index]
    if len(rows) != 1:
        raise ValueError(f'unknown cohort/index: {cohort}/{index}')
    row = rows[0]
    config_key = 'reid' if cohort == 'same10_reid' else 'thermal_arm_a'
    config = recipe[config_key]
    case = output / 'cases' / f'{cohort}_{index}'
    if case.exists():
        raise RuntimeError(f'refuse overwrite: {case}')
    case.mkdir(parents=True)
    print(json.dumps({'event': 'case_started', 'cohort': cohort, 'index': index,
                      'temperature': row['temperature'], 'pbs_job_id': os.environ.get('PBS_JOBID')}), flush=True)
    model = load_model(Path(config['bundle']))
    network = load_network(Path(config['dataset']), index, row['source'])
    options = OptimizationConfig(**config['optimization'])
    zero_stiffness = torch.zeros_like(network.reference.edge_attr[:, 3])
    zero_positions = torch.zeros_like(network.reference.reference_context_positions[:, :2])
    sides, _, _, _ = boundary_constraints(network.reference, network.physical)
    position0 = network.reference.reference_context_positions[:, :2].detach()
    planar = PlanarConstraints(position0.numpy(), network.reference.edge_index.numpy(),
                               [network.physical.box.x, network.physical.box.y], options.spring_clearance_fraction)
    pristine_pairs = planar.violating_pairs(position0.numpy()).tolist()
    if not valid_geometry(network.reference, network.physical, zero_stiffness, zero_positions, sides, planar, position0):
        # A crossing or contact already present in an original is a declared result,
        # never a reason to alter the cohort or feed physics back into optimization.
        torch.save(network.physical, case / 'original.pt')
        status = {'cohort': cohort, 'index': index, 'temperature': row['temperature'], 'status': 'invalid_original',
                  'error': 'pristine geometry violates the frozen planar constraints',
                  'periodic_pristine_violating_pairs': pristine_pairs, 'planar_constraints': planar.summary(),
                  'original_graph': str((case / 'original.pt').resolve()),
                  'original_graph_sha256': sha256(case / 'original.pt'), 'pbs_job_id': os.environ.get('PBS_JOBID'),
                  'physics_used': False}
        (case / 'outcome.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps({'event': 'case_finished', **status}), flush=True)
        return
    result = optimize_network(model, network, options)
    if cohort == 'same10_reid':
        verification_protocol = {'protocol': 'Reid athermal x compression with y box relaxation',
                                 'scale_x': 0.9700106999999999, 'increments': 120,
                                 'feedback_to_optimizer': False}
    else:
        verification_protocol = json.loads((output / 'thermal_protocol.json').read_text())
        verification_protocol.update(protocol_name='thermal_armA_200frame_ols_v1', expected_designs=2,
                                     expected_evaluations=6, feedback_to_optimizer=False)
    manifest = freeze_final_designs(result, case / 'frozen', dataset_path=Path(config['dataset']),
                                   known_case_refactor_verification=False, verification_protocol=verification_protocol)
    status = {'cohort': cohort, 'index': index, 'temperature': row['temperature'], 'status': 'frozen',
              'periodic_pristine_violating_pairs': pristine_pairs, 'planar_constraints': result.geometry_summary,
              'frozen_manifest': str((case / 'frozen/frozen_manifest.json').resolve()),
              'frozen_manifest_sha256': sha256(case / 'frozen/frozen_manifest.json'), 'designs': manifest['designs'],
              'pbs_job_id': os.environ.get('PBS_JOBID'), 'physics_used': False}
    (case / 'outcome.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({'event': 'case_finished', 'cohort': cohort, 'index': index, 'status': 'frozen'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    main(args.output, args.cohort, args.index)
