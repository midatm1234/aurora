"""Reusable experiment helpers: output isolation and bounded pilot arguments."""
from pathlib import Path

import pytest
import yaml

from finetune.generate_spatiotemporal_configs import HEADS, VARIANTS, build_variant, main
from finetune.pilot_spatiotemporal import run_pilot


def test_generated_variants_isolate_artifacts_and_reuse_original_data():
    base = {'case_name': 'O3_reference', 'paths': {'data_dir': '../data'},
            'model': {'refinement': {}}}
    variants = [build_variant(base, head=head, name=name, stage=stage,
                              backend=backend, bottleneck_attention=attention)
                for head in HEADS for name, stage, backend, attention in VARIANTS]
    assert len({v['case_name'] for v in variants}) == len(HEADS) * len(VARIANTS)
    assert all(v['paths']['data_case_name'] == 'O3_reference' for v in variants)
    assert all(v['model']['mamba_temporal']['enabled'] is False for v in variants)
    assert all(v['model']['mamba_temporal_enabled'] is False for v in variants)
    assert base == {'case_name': 'O3_reference', 'paths': {'data_dir': '../data'},
                    'model': {'refinement': {}}}
    base['paths']['data_case_name'] = 'shared_prepared_O3'
    variant = build_variant(base, head=HEADS[0], name='control', stage='spatial_only',
                            backend='none', bottleneck_attention=False)
    assert variant['paths']['data_case_name'] == 'shared_prepared_O3'


def test_generator_keeps_base_project_root_when_writing_another_directory(tmp_path):
    source = tmp_path / 'configs'
    source.mkdir()
    base_path = source / 'custom.yaml'
    base_path.write_text(yaml.safe_dump({'case_name': 'base', 'paths': {'project_root': '..'}}))
    output = tmp_path / 'experiments' / 'ablations'
    assert main(['--base', str(base_path), '--out', str(output), '--heads', HEADS[0]]) == 0
    generated = list(output.glob('*.yaml'))
    assert len(generated) == len(VARIANTS)
    for path in generated:
        config = yaml.safe_load(path.read_text())
        assert (path.parent / config['paths']['project_root']).resolve() == tmp_path
        assert config['_ablation']['derived_from'] == 'custom.yaml'
        assert config['paths']['data_case_name'] == 'base'


@pytest.mark.parametrize('overrides', [
    {'steps': 0}, {'steps': -1}, {'batch': 0}, {'batch': 5, 'train_trajectories': 4},
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'seed': -1},
])
def test_invalid_pilot_bounds_fail_before_model_or_data_allocation(overrides):
    arguments = dict(steps=1, batch=1, seed=4, learning_rate=.001, train_trajectories=4)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        run_pilot('diffusion_unet', 'spatial_only', **arguments)


def test_pilot_explicitly_disables_legacy_temporal_adapter(monkeypatch):
    from finetune import pilot_spatiotemporal as pilot

    captured = {}

    class Model:
        def conditioning_channels(self):
            return 3

        def initialize_refiner(self, channels):
            assert channels == 3

    def build(backbone, packing, config):
        captured.update(config['model'])
        return Model()

    monkeypatch.setattr(pilot, 'build_two_phase_refiner', build)
    pilot._build_model('diffusion_unet', 'temporal_causal_conv', object())
    assert captured['mamba_temporal'] == {'enabled': False}
    assert captured['mamba_temporal_enabled'] is False
    assert captured['refinement']['temporal']['backend'] == 'causal_conv'
