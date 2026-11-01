"""同一 Stage4 权重的 H1 开关消融；不修改原有普通测试的流水线校验。"""
import argparse
import json
from pathlib import Path
import sys
import test_idea8_stage4_sweep as sweep


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--guidance', choices=['on','off'], required=True)
    p.add_argument('--reference_h1_run', required=True)
    toggle,rest=p.parse_known_args()
    cli=sweep.parse_args(rest)
    original=sweep.load_model_and_data

    def load(args):
        # 先按训练记录严格加载原模型，再明确记录这次消融的推理覆盖。
        model,dataset,meta=original(args)
        trained_h1=meta['is_h1']
        ref=json.loads((Path(toggle.reference_h1_run)/'run_config.json').read_text())
        headpath=ref['h1_structure_checkpoint']
        assert sweep.file_hash(headpath)==ref['h1_pipeline']['h1_sha256']
        assert ref['event_lora_checkpoint']==meta['train_config']['event_lora_checkpoint']
        if toggle.guidance=='on':
            if not trained_h1:
                from models.event_structure_head import load_structure_checkpoint
                from models.event_structure_guidance import GuidanceConfig
                from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset,StructureEventConfig
                head,headmeta,operator=load_structure_checkpoint(headpath,require_calibrated=True)
                model.h1_predictor=head.cuda().eval().requires_grad_(False)
                model.h1_operator=operator
                model.h1_config=GuidanceConfig(**ref['h1_pipeline']['guidance'])
                dataset=SDEEventStructureDataset('test',model.args,
                    StructureEventConfig.from_dict(headmeta['event_config']),args.test_root)
            assert model.h1_config.steps==1
            args.h1_decoder_path='full_refinement'
        else:
            # 不调用结构预测/候选求解；只取冻结 FP32 Stage3 latent，再原样解码。
            args.h1_decoder_path='legacy_stage3'
        meta.update(trained_with_h1=trained_h1,is_h1=toggle.guidance=='on',
            h1_decoder_path=args.h1_decoder_path,inference_ablation='same_checkpoint_h1_toggle',
            guidance_enabled=toggle.guidance=='on',guidance_steps=1 if toggle.guidance=='on' else 0,
            reference_h1_sha256=ref['h1_pipeline']['h1_sha256'])
        return model,dataset,meta

    sweep.load_model_and_data=load
    sweep.aggregate(cli) if cli.aggregate else sweep.infer(cli)


if __name__=='__main__':main()
