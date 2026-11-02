"""用真实指标与合成偏色图验证 Y/RGB 区分，并验证两级汇总不丢字段。"""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from my_utils.quality_metrics import (
    DISTORTION_NAMES, METRIC_NAMES, METRIC_PROTOCOL,
    build_quality_metrics, compute_quality_metrics,
)


class DualChannelMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = build_quality_metrics('cpu', include_lpips=False)

    def test_rgb_psnr_uses_all_channels_and_y_matches_previous_protocol(self):
        import pyiqa
        torch.manual_seed(12)
        target = torch.rand(1,3,32,32)*.4+.3
        prediction = target.clone()
        # 红/绿误差在 Y 中抵消，但 RGB 仍有可测的色彩误差。
        prediction[:,0] += .05
        prediction[:,1] -= .05*65.481/128.553
        values = compute_quality_metrics(prediction*2-1,target*2-1,self.metrics)
        self.assertEqual(set(values),set(DISTORTION_NAMES))
        mse = ((prediction-target)**2).mean()
        expected = float(10*torch.log10(1/(mse+1e-8)))
        self.assertAlmostEqual(values['psnr_rgb'],expected,places=4)
        self.assertGreater(values['psnr_y'],values['psnr_rgb']+20)
        self.assertGreater(values['ssim_y'],values['ssim_rgb'])
        for metric in ('psnr','ssim'):
            old = pyiqa.create_metric(metric,test_y_channel=True,color_space='ycbcr',device='cpu')
            self.assertAlmostEqual(values[metric+'_y'],float(old(prediction,target)),places=4)

    def test_identity_and_shared_clamp(self):
        torch.manual_seed(7)
        image = torch.rand(1,3,32,32)
        values = compute_quality_metrics(image*2-1,image*2-1,self.metrics)
        self.assertAlmostEqual(values['ssim_y'],1.,places=6)
        self.assertAlmostEqual(values['ssim_rgb'],1.,places=6)
        x = image*4-2
        actual = compute_quality_metrics(x,x.clamp(-1,1),self.metrics)
        self.assertAlmostEqual(actual['ssim_rgb'],1.,places=6)

    def test_training_rgb_metrics_match_test_rgb(self):
        metrics = build_quality_metrics('cpu', include_lpips=False, channels=('rgb',))
        self.assertEqual(set(metrics), {'psnr_rgb', 'ssim_rgb'})
        torch.manual_seed(19)
        target = torch.rand(1,3,32,32)
        prediction = target * .9
        for name, metric in metrics.items():
            self.assertEqual(float(metric(prediction,target)), float(self.metrics[name](prediction,target)))

    def test_resume_selection_does_not_mix_y_and_rgb(self):
        from train_esfr import restore_best_psnr
        # 旧 Y 分数即使更高，也不能阻止新的 RGB 最佳权重保存。
        self.assertEqual(restore_best_psnr({'best_psnr':40}), (float('-inf'),True))
        self.assertEqual(restore_best_psnr({'selection_metric':'psnr_y','best_psnr':40}), (float('-inf'),True))
        self.assertEqual(restore_best_psnr({'selection_metric':'psnr_rgb','best_psnr_rgb':25}), (25,False))
        with self.assertRaises(ValueError):
            restore_best_psnr({'selection_metric':'unknown'})

    def test_accumulator_keeps_distinct_channel_averages(self):
        from test_event_film_lora_osediff import MetricAccumulator
        accumulator = MetricAccumulator()
        a = dict(psnr_y=30,ssim_y=.9,psnr_rgb=20,ssim_rgb=.7,lpips=.2)
        b = dict(psnr_y=40,ssim_y=1,psnr_rgb=26,ssim_rgb=.9,lpips=.1)
        accumulator.update(a);accumulator.update(b)
        self.assertEqual(accumulator.count,2)
        for name in METRIC_NAMES:
            self.assertAlmostEqual(accumulator.averages()[name],(a[name]+b[name])/2)

    def test_shard_aggregation_exports_all_metrics(self):
        from my_utils import evaluation
        rows = [dict(psnr_y=30,ssim_y=.9,psnr_rgb=20,ssim_rgb=.7,lpips=.2),
                dict(psnr_y=40,ssim_y=1,psnr_rgb=26,ssim_rgb=.9,lpips=.1)]
        documents = {'manifest_shard0.json':dict(metric_protocol=METRIC_PROTOCOL,limit=0,dataset_count=2,settings={'on':{}}),
                     'done_shard0.json':dict(indices=[0,1])}
        for index,values in enumerate(rows):
            documents[f'{index:06d}.json']=dict(index=index,scene='room',frame_id=str(index),
                gt_png=f'/fake/gt{index}.png',results={'on':dict(**values,result_path=f'/fake/out{index}.png')})
        written = {}
        def save_csv(path, values, fields):
            self.assertTrue(set(METRIC_NAMES).issubset(fields))
            written[Path(path).name]=values
        with patch.object(Path,'read_text',lambda p,*a,**k:json.dumps(documents[p.name])), \
             patch.object(Path,'exists',return_value=False), patch.object(Path,'mkdir'), \
             patch.object(Path,'glob',return_value=[Path('a.png'),Path('b.png')]), \
             patch.object(evaluation.os,'link'), patch.object(evaluation,'atomic_json'), \
             patch('pyiqa.create_metric',return_value=lambda *a:torch.tensor(12.)), \
             patch('test_idea3_decoder_refinement.write_csv',side_effect=save_csv):
            evaluation.aggregate(SimpleNamespace(output_dir='/fake/results',shards=1))
        self.assertEqual(written['per_image.csv'][0]['psnr_rgb'],20)
        for filename in ('per_scene.csv','summary.csv'):
            for key in METRIC_NAMES:
                self.assertAlmostEqual(written[filename][0][key],(rows[0][key]+rows[1][key])/2)

    def test_old_summary_protocol_rejected(self):
        from my_utils.evaluation import aggregate
        with patch.object(Path,'read_text',return_value=json.dumps(dict(metric_protocol='old-y-only'))):
            with self.assertRaisesRegex(ValueError,'Old metric protocol'):
                aggregate(SimpleNamespace(output_dir='/fake/results',shards=1))
