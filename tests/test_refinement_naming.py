"""REDF 改名只改变配置文本，旧权重/共同初始化仍可严格恢复。"""
import copy
import io
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from models.decoder_refinement import normalize_refinement_variant
from models.decoder_refinement_initialization import load_refinement_initialization
from test_decoder_refinement import build_small_vae


class RefinementNamingTest(unittest.TestCase):
    def test_alias_and_new_serialization(self):
        self.assertEqual(normalize_refinement_variant('legacy_fused'), 'rgb_event_decoder_fusion')
        vae = build_small_vae()
        vae.install_feature_aware_decoder(refinement_variant='legacy_fused')
        self.assertEqual(vae.decoder.refinement_config()['variant'], 'rgb_event_decoder_fusion')
        with self.assertRaises(ValueError):
            normalize_refinement_variant('cgdrs')

    def test_old_stage4_config_loads_without_mutating_input(self):
        from osediff_event_film import OSEDiffEventFiLMGen
        gen = OSEDiffEventFiLMGen.__new__(OSEDiffEventFiLMGen)
        torch.nn.Module.__init__(gen)
        gen.vae = build_small_vae()
        gen.vae.install_feature_aware_decoder()
        gen.args = SimpleNamespace()
        gen.decoder_refinement_enabled = True
        gen.rgb_event_encoder = torch.nn.Module()
        gen.rgb_event_encoder.pretrained_manifest = None
        config = gen.vae.decoder.refinement_config()
        config['variant'] = 'legacy_fused'
        original = copy.deepcopy(config)
        state = {k: v.clone() for k, v in gen.vae.decoder.feature_fusion_blocks.state_dict().items()}
        loaded, skipped = gen.load_decoder_refinement_checkpoint(dict(
            decoder_refinement_config=config, state_dict_decoder_refinement=state,
            state_dict_rgb_event_encoder={},
        ))
        self.assertEqual((loaded, skipped), (len(state), 0))
        self.assertEqual(config, original)
        self.assertEqual(gen.vae.decoder.refinement_config()['variant'], 'rgb_event_decoder_fusion')

    def test_old_paired_initialization_loads(self):
        model = SimpleNamespace(vae=SimpleNamespace(decoder=torch.nn.Module()))
        model.vae.decoder.feature_fusion_blocks = torch.nn.Linear(3, 3)
        model.vae.decoder.refinement_config = lambda: {'variant': 'rgb_event_decoder_fusion'}
        model.rgb_event_encoder = SimpleNamespace(pretrained_manifest={'sha256': {'rgb': 'fixture'}})
        state = {k:v.clone()+1 for k,v in model.vae.decoder.feature_fusion_blocks.state_dict().items()}
        saved = dict(format='paired-stage4-refinement-init-v1', seed=42,
            stage3_sha256='fixture', decoder_refinement_config={'variant':'legacy_fused'},
            rgb_event_manifest=model.rgb_event_encoder.pretrained_manifest, state_dict=state,
            state_sha256='fixture', torch_rng_cpu=torch.get_rng_state(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state())
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        with patch('torch.cuda.is_available', return_value=False), patch(
            'models.decoder_refinement_initialization.sha256_file', return_value='fixture'
        ):
            load_refinement_initialization(model, buffer, SimpleNamespace(seed=42,event_lora_checkpoint='fixture'))
        for key,value in state.items():
            torch.testing.assert_close(model.vae.decoder.feature_fusion_blocks.state_dict()[key],value,rtol=0,atol=0)
