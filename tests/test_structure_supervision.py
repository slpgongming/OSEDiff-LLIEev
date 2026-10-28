"""区域监督、固定评价口径、best 权重及续训历史的实际回归。"""

from dataclasses import asdict
import json

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from models.structure_supervision import (StructureSupervisionConfig, StructureValidationAccumulator,
                                          structure_supervision_loss)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_h0_matches_original_loss_and_gradient_exactly():
    torch.manual_seed(3)
    prediction = torch.rand(2, 2, 7, 9, requires_grad=True)
    target = torch.rand_like(prediction)
    old = F.smooth_l1_loss(prediction, target, beta=.1)
    actual = structure_supervision_loss(prediction, target)
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0],
                               torch.autograd.grad(old, prediction)[0], rtol=0, atol=0)


def test_balanced_loss_does_not_let_flat_pixel_count_dominate():
    config = StructureSupervisionConfig(mode='region_balanced')
    # 两张图各包含强边缘和不同数量的平坦像素；区域误差相同时，两张图的 loss 应一致。
    target = torch.tensor([[[[0., 0., 0., .8]]]])
    prediction = torch.tensor([[[[.1, .1, .1, .4]]]], requires_grad=True)
    wider_target = torch.cat((target[..., :1].expand(1, 1, 1, 30), target[..., -1:]), dim=-1)
    wider_prediction = torch.cat((prediction[..., :1].expand(1, 1, 1, 30), prediction[..., -1:]), dim=-1)
    expected = (F.smooth_l1_loss(torch.tensor(.1), torch.tensor(0.), beta=.1)
                + F.smooth_l1_loss(torch.tensor(.4), torch.tensor(.8), beta=.1)) / 2
    loss = structure_supervision_loss(prediction, target, config)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(structure_supervision_loss(wider_prediction, wider_target, config), expected)
    gradient, = torch.autograd.grad(loss, prediction)
    assert torch.isfinite(gradient).all() and gradient[..., -1].abs().sum() > 0


def test_empty_regions_are_skipped_per_sample_and_scale():
    config = StructureSupervisionConfig(mode='region_balanced')
    target = torch.tensor([[[[0.]], [[.8]]], [[[.1]], [[0.]]]])
    prediction = (target + .03).requires_grad_(True)
    expected = F.smooth_l1_loss(prediction, target, beta=.1)
    torch.testing.assert_close(structure_supervision_loss(prediction, target, config), expected)


@pytest.mark.parametrize('kwargs', [dict(thresholds=(.2,.1)), dict(thresholds=(0.,.2)),
                                   dict(thresholds=(.05,1.)), dict(region_weights=(0.,1.,1.)),
                                   dict(region_weights=(1.,float('nan'),1.)), dict(version=2)])
def test_invalid_region_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        StructureSupervisionConfig(**kwargs)


def test_flat_only_validation_is_finite_and_reports_missing_edge_regions():
    metric = StructureValidationAccumulator(2, scenes=('a',), border=0)
    zeros = torch.zeros(2, 2, 8, 8)
    before = torch.get_rng_state()
    metric.update(zeros, zeros, ['a','a'])
    result = metric.compute()
    assert torch.equal(before, torch.get_rng_state())
    assert result['overall']['samples'] == 2
    assert result['overall']['balanced_mae'] == 0
    assert result['overall']['strong_mae'] is None
    assert result['overall']['spatial_correlation'] is None
    assert result['overall']['edge_f1'] == 1
    assert result['overall']['metric_counts']['flat_mae'] == 4
    json.dumps(result, allow_nan=False)


def test_edge_matching_tolerance_and_scene_aggregation():
    target = torch.zeros(2, 1, 10, 10)
    prediction = torch.zeros_like(target)
    target[..., 3] = 1
    prediction[0, ..., 5] = 1  # 两像素位移，在容差内。
    prediction[1, ..., 7] = 1  # 四像素位移，在容差外。
    metric = StructureValidationAccumulator(1, scenes=('a','b'), border=0, tolerance=2)
    metric.update(prediction, target, ['a','b'])
    report = metric.compute()
    assert report['by_scene']['a']['edge_f1'] == 1
    assert report['by_scene']['b']['edge_f1'] == 0
    assert report['overall']['edge_f1'] == .5
    parts = []
    for index, name in enumerate(('a','b')):
        part = StructureValidationAccumulator(1, scenes=('a','b'), border=0, tolerance=2)
        part.update(prediction[index:index+1], target[index:index+1], [name])
        parts.append(part)
    parts[0].totals += parts[1].totals
    assert parts[0].compute() == report


def common_args(tmp_path):
    return ['--data_root', str(tmp_path / 'data'), '--scene_split', str(tmp_path / 'split.json'),
            '--output_dir', str(tmp_path / 'run'), '--device', 'cpu', '--base_channels', '4', '--batch_size', '2']


def test_resume_inherits_configs_and_rejects_silent_objective_changes(tmp_path):
    import train_event_structure_head as training
    supervision = StructureSupervisionConfig(mode='region_balanced', thresholds=(.04,.25))
    checkpoint = dict(phase='structure', supervision_config=asdict(supervision), selection_config=asdict(training.SelectionConfig()))
    args = training.parse_args(common_args(tmp_path) + ['--resume_from_checkpoint','last.pt'])
    assert training.resolve_training_configs(args, checkpoint)[0] == supervision
    for options in (['--structure_loss','smooth_l1'], ['--region_thresholds','.05','.2'],
                    ['--validation_border','3'], ['--best_metric','validation_loss']):
        args = training.parse_args(common_args(tmp_path) + ['--resume_from_checkpoint','last.pt'] + options)
        with pytest.raises(ValueError, match='Resume'):
            training.resolve_training_configs(args, checkpoint)
    args = training.parse_args(common_args(tmp_path) + ['--phase','confidence','--init_structure_checkpoint','best.pt'])
    resolved, selection = training.resolve_training_configs(args, checkpoint)
    assert resolved == supervision and selection.metric == 'validation_loss'


def _dataset(root):
    for scene in ('a','b','c'):
        for folder in ('low','normal'):
            (root / 'data' / scene / folder).mkdir(parents=True)
        for index in range(2):
            tick = 10**15 + index * 1000
            image = np.zeros((264,264,3),dtype=np.uint8)
            image[:,130:] = 40 + index * 10
            Image.fromarray(image).save(root / 'data' / scene / 'low' / f'{tick}.png')
            Image.fromarray(image * 3).save(root / 'data' / scene / 'normal' / f'{tick+1}.png')
            events = np.array([[tick-40,130,20,1],[tick-10,130,100,-1],
                               [tick+10,130,150,1],[tick+40,130,220,-1]],dtype=np.int64)
            np.savez(root / 'data' / scene / 'low' / f'{tick}.npz', events)
    (root / 'split.json').write_text(json.dumps(dict(train=['a'],calibration=['b'],validation=['c'])),encoding='utf-8')


def test_best_is_earlier_checkpoint_and_survives_resume_to_new_directory(tmp_path, monkeypatch):
    import train_event_structure_head as training
    _dataset(tmp_path)
    original = training.phase_loss
    validation_calls = []
    def worsening_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('return_details'):
            validation_calls.append(1)
            loss, edge_error, confidence_error, details = result
            details['prediction'] = details['target'] * (0.9 if len(validation_calls) == 1 else .5)
            return loss, edge_error, confidence_error, details
        return result
    monkeypatch.setattr(training, 'phase_loss', worsening_validation)
    args = training.parse_args(common_args(tmp_path) + ['--structure_loss','region_balanced','--epochs','2'])
    training.main(args)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    best = load(tmp_path/'run/checkpoint_best.pt')
    last = load(tmp_path/'run/checkpoint_last.pt')
    assert best['epoch'] == 0 and last['epoch'] == 1
    assert last['selection_state']['best_epoch'] == 0
    assert last['supervision_config']['mode'] == 'region_balanced'
    assert last['global_step'] == 2
    assert any(not torch.equal(best['state_dict'][key], last['state_dict'][key]) for key in best['state_dict'])
    args = training.parse_args(common_args(tmp_path) + ['--output_dir',str(tmp_path/'resumed'), '--epochs','3',
                                                      '--resume_from_checkpoint',str(tmp_path/'run/checkpoint_last.pt')])
    training.main(args)
    resumed_best = load(tmp_path/'resumed/checkpoint_best.pt')
    assert resumed_best['epoch'] == 0
    for key in best['state_dict']:
        torch.testing.assert_close(resumed_best['state_dict'][key], best['state_dict'][key], rtol=0, atol=0)
    assert load(tmp_path/'resumed/checkpoint_last.pt')['global_step'] == 3


def test_legacy_resume_starts_explicit_new_selection_window(tmp_path):
    import train_event_structure_head as training
    state, best = training.resume_selection_state(dict(epoch=9), 'legacy.pt', tmp_path, training.SelectionConfig())
    assert best is None and state['best_score'] is None and state['history_start_epoch'] == 10
    checkpoint = dict(epoch=2, phase='structure', scene_split={},
                      selection_state=dict(metric='balanced_mae',best_epoch=0,best_score=.1,history_start_epoch=0))
    with pytest.raises(ValueError, match='Historical checkpoint_best'):
        training.resume_selection_state(checkpoint, str(tmp_path/'last.pt'), tmp_path, training.SelectionConfig())
