"""Stage4 配对测试：复用每帧 frozen latent，扫描 decoder 执行层与全局残差强度。

每个进程只处理不重叠的 index % shards 分片；逐帧原子记录可恢复，FID在合并后算全量。
训练源码/权重保持只读；H1观测与内层优化不接收GT。
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

SETTINGS = {f'{label}_w{round(weight*100):03d}': dict(levels=levels, weight=weight)
            for label,levels in [('d23',[2,3]),('d123',[1,2,3])]
            for weight in (.25,.5,.75,1.)}
ALL_SETTINGS = {**SETTINGS, **{f'd0123_w{round(w*100):03d}': dict(levels=[0,1,2,3], weight=w)
                              for w in (.25,.5,.75,1.)}}


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--run_dir',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--test_root',default='/home/data/Chen_data/SDE_dataset/sde_in_release/test')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=0,help='仅小样本验证使用；0为完整测试集。')
    p.add_argument('--aggregate',action='store_true')
    p.add_argument('--h1_decoder_path', choices=('auto','legacy_stage3','full_refinement'), default='auto',
                   help='auto 继承训练路径；full_refinement 在最终 Stage4 decoder 上分别优化每个设置。')
    p.add_argument('--settings', nargs='+', choices=tuple(ALL_SETTINGS), default=list(SETTINGS),
                   help='选择层组合与强度；d0123 为训练时使用的全部四层。')
    args=p.parse_args(argv)
    if args.shards<1 or not 0<=args.shard<args.shards or args.limit<0:p.error('Invalid shard/limit')
    if len(set(args.settings)) != len(args.settings):p.error('Duplicate settings')
    return args


def load_model_and_data(cli):
    from accelerate.utils import set_seed
    from dataloaders.sde_event_film_dataset import SDEEventFilmDataset
    from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset,StructureEventConfig
    from models.h1_decoder_refinement import H1DecoderRefinementGen
    from models.decoder_refinement_control import NoH1DecoderRefinementGen
    from test_idea8_structure_dps import configure_inference_numerics
    run=Path(cli.run_dir)
    state=json.loads((run/'launch/manager_state.json').read_text())
    assert state['status']=='completed',state
    train=json.loads((run/'run_config.json').read_text())
    trained_path=train.get('h1_decoder_path','legacy_stage3')
    if cli.h1_decoder_path == 'auto':cli.h1_decoder_path=trained_path
    if trained_path == 'full_refinement' and cli.h1_decoder_path != trained_path:
        raise ValueError('A full-refinement H1 training run requires matching inference guidance.')
    checkpoint_path=run/'checkpoints/model_final.pkl'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==20000
    info=checkpoint.get('additional_info',{})
    is_h1=info.get('h1_pipeline') is not None
    if cli.h1_decoder_path == 'full_refinement' and not is_h1:
        raise ValueError('Full-refinement H1 requires a Stage4 checkpoint with H1 bundle.')
    assert is_h1==bool(train['h1_structure_checkpoint'])
    assert train['clora_event_active_bins']==5 and train['rgb_event_active_bins']==1
    assert not train['unet_lora_active_parts'] and train['generator_timestep']==500
    assert info['event_active_bins']==dict(clora=5,rgb_event_encoder=1,total_channels=5)
    assert file_hash(train['event_lora_checkpoint'])=='5496c3bfe39d59c73d9a211ce1d1c1d16dd2725475b61ebaf8fd0636932f5b31'
    if is_h1:
        assert info['h1_pipeline']==train['h1_pipeline']
        assert file_hash(train['h1_structure_checkpoint'])==info['h1_pipeline']['h1_sha256']
    args=argparse.Namespace(**train)
    args.decoder_refinement_enabled=False
    set_seed(cli.seed)
    numerics=configure_inference_numerics(True)
    model=(H1DecoderRefinementGen if is_h1 else NoH1DecoderRefinementGen)(args)
    model.load_stage1_full(args.stage1_checkpoint)
    model.load_event_checkpoint(torch.load(args.event_lora_checkpoint,map_location='cpu',weights_only=False))
    args.decoder_refinement_enabled=True;model.decoder_refinement_enabled=True
    model.install_decoder_refinement()
    model.load_event_checkpoint(checkpoint)
    model.activate_lora_adapters(args.unet_lora_active_parts)
    model.requires_grad_(False);model.eval();model.cuda()
    if is_h1:
        # 同时核查bundle，防止外部结构权重被替换后悄悄改变推理流水线。
        assert all(torch.equal(v.cpu(),info['h1_checkpoint']['state_dict'][k])
                   for k,v in model.h1_predictor.state_dict().items())
        dataset=SDEEventStructureDataset('test',args,
            StructureEventConfig.from_dict(info['h1_pipeline']['event_config']),cli.test_root)
    else:
        dataset=SDEEventFilmDataset('test',args,cli.test_root)
    assert len(dataset)==2741,len(dataset)
    metadata=dict(train_config=train,checkpoint=str(checkpoint_path),checkpoint_sha256=file_hash(checkpoint_path),
                  is_h1=is_h1,pipeline=info.get('h1_pipeline'),numeric_settings=numerics,
                  seed=cli.seed,test_root=cli.test_root,dataset_count=len(dataset),limit=cli.limit,
                  shards=cli.shards,settings={name:ALL_SETTINGS[name] for name in cli.settings},
                  h1_decoder_path=cli.h1_decoder_path, decoder_level_order='up_blocks execution, deepest to shallowest',
                  backbone_dtype='float32',refinement_autocast='float16',uses_gt_for_guidance=False,
                  metric_protocol='PSNR(Y)/SSIM(Y) ycbcr; LPIPS float RGB; whole-set FID from PNG')
    del checkpoint
    return model,dataset,metadata


def render_variants(model,latent,rgb,events_refinement,settings=SETTINGS):
    """冻结骨干产生的同一个latent，八个设置只改变decoder融合层/残差系数。"""
    for name,setting in settings.items():
        model.vae.decoder.set_inference_fusion_levels(setting['levels'])
        model.decoder_fusion_weight=setting['weight']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            prediction=model.decode_denoised_latent(latent,rgb,events_refinement=events_refinement,
                decoder_fusion_enabled=True,refinement_fusion_mode='both')
        yield name,prediction


def infer(cli):
    import pyiqa
    from accelerate.utils import set_seed
    from models.h1_decoder_refinement import guided_latents, refinement_guided_sample
    from models.decoder_refinement_control import frozen_stage3_latents
    from test_idea3_decoder_refinement import quality_metrics
    from test_event_film_lora_osediff import tensor_to_pil
    model,dataset,metadata=load_model_and_data(cli)
    settings=metadata['settings']
    output=Path(cli.output_dir);output.mkdir(parents=True,exist_ok=True)
    manifest=output/f'manifest_shard{cli.shard}.json'
    if manifest.exists():assert json.loads(manifest.read_text())==metadata,'Resume configuration mismatch'
    else:atomic_json(manifest,metadata)
    metrics=dict(psnr=pyiqa.create_metric('psnr',test_y_channel=True,color_space='ycbcr').cuda(),
                 ssim=pyiqa.create_metric('ssim',test_y_channel=True,color_space='ycbcr').cuda(),
                 lpips=pyiqa.create_metric('lpips',device='cuda').cuda())
    indices=list(range(cli.limit or len(dataset)))[cli.shard::cli.shards]
    start=time.time()
    shape_trace={}
    handles=[block.register_forward_pre_hook(
        lambda module,inputs,i=i:shape_trace.setdefault(f'd{i}',dict(condition=list(inputs[0].shape),decoder=list(inputs[1].shape))) and None)
        for i,block in enumerate(model.vae.decoder.feature_fusion_blocks)]
    for ordinal,index in enumerate(indices):
        record_path=output/'records'/f'{index:06d}.json'
        if record_path.exists():
            old=json.loads(record_path.read_text())
            if set(old['results'])==set(settings) and all(Path(v['result_path']).is_file() for v in old['results'].values()):continue
        set_seed(cli.seed+index)
        sample=dataset[index]
        rgb=sample['conditioning_pixel_values'][None].cuda()
        events=sample['events'][None].cuda()
        events_refinement=sample['events_refinement'][None].cuda()
        assert torch.count_nonzero(events_refinement[:,1:])==0,'HDRev event tail bins must be zero'
        batch=dict(prompt=[sample['prompt']],neg_prompt=[sample['neg_prompt']])
        timesteps=torch.tensor([model.args.generator_timestep],device='cuda')
        diagnostics=None
        if metadata['is_h1'] and cli.h1_decoder_path == 'legacy_stage3':
            latent,diagnostics=guided_latents(model,rgb,events,sample['events_structure'][None].cuda(),batch,
                model.h1_predictor,model.h1_operator,model.h1_config)
        elif not metadata['is_h1']:
            latent=frozen_stage3_latents(model,rgb,events,timesteps,batch)
        else:
            latent=None
        target=sample['output_pixel_values'][None].cuda()
        if cli.limit and ordinal==0 and latent is not None:
            # 真实模型验收：默认全层与显式全层一致；空集合与关闭fusion逐位一致。
            def decode(enabled=True):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    return model.decode_denoised_latent(latent,rgb,events_refinement=events_refinement,
                        decoder_fusion_enabled=enabled,refinement_fusion_mode='both')
            model.decoder_fusion_weight=1.
            model.vae.decoder.set_inference_fusion_levels(None);reference=decode()
            model.vae.decoder.set_inference_fusion_levels((0,1,2,3));all_levels=decode()
            torch.testing.assert_close(reference,all_levels,rtol=0,atol=0)
            model.vae.decoder.set_inference_fusion_levels(());empty=decode()
            torch.testing.assert_close(empty,decode(False),rtol=0,atol=0)
            atomic_json(output/'real_entry_audit.json',dict(all_level_parity=True,empty_is_fusion_off=True,
                decoder_shapes=shape_trace,clora_shape=list(events.shape),refinement_shape=list(events_refinement.shape),
                refinement_tail_nonzero=int(torch.count_nonzero(events_refinement[:,1:])),h1_diagnostics=diagnostics))
            del reference,all_levels,empty
        stem=f'{index:06d}_{sample["scene_name"]}_{sample["frame_index"]+1:03d}'
        refs=output/'references';refs.mkdir(exist_ok=True)
        low_path=refs/f'{stem}_ll.png';gt_path=refs/f'{stem}_gt.png'
        tensor_to_pil(rgb[0]).save(low_path);tensor_to_pil(target[0]).save(gt_path)
        results={}
        if cli.h1_decoder_path == 'full_refinement':
            # 各层设置改变优化目标：每个设置必须从同一随机种子独立求解。
            def predictions():
                for name,setting in settings.items():
                    set_seed(cli.seed+index)
                    model.vae.decoder.set_inference_fusion_levels(setting['levels'])
                    model.decoder_fusion_weight=setting['weight']
                    image,z,stats=refinement_guided_sample(model,rgb,events,
                        sample['events_structure'][None].cuda(),batch,model.h1_predictor,
                        model.h1_operator,model.h1_config,events_refinement=events_refinement,
                        refinement_autocast_dtype=torch.float16)
                    yield name,image,z,stats
        else:
            def predictions():
                for name,image in render_variants(model,latent,rgb,events_refinement,settings):
                    yield name,image,latent,diagnostics
        for name,prediction,z,stats in predictions():
            path=output/name/'images'/f'{stem}_result.png';path.parent.mkdir(parents=True,exist_ok=True)
            tensor_to_pil(prediction[0]).save(path)
            with torch.no_grad():values=quality_metrics(prediction,target,metrics)
            assert all(math.isfinite(v) for v in values.values()),values
            results[name]=dict(**values,result_path=str(path),
                latent_sha256=hashlib.sha256(z.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                h1_diagnostics=stats)
        record=dict(index=index,scene=sample['scene_name'],frame_id=sample['frame_id'],frame_index=sample['frame_index'],
                    seed=cli.seed+index,low_path=sample['low_path'],normal_path=sample['normal_path'],
                    low_png=str(low_path),gt_png=str(gt_path),results=results,h1_diagnostics=diagnostics,
                    latent_sha256=(hashlib.sha256(latent.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                   if latent is not None else None))
        atomic_json(record_path,record)
        if ordinal%10==0 or ordinal+1==len(indices):
            status=dict(shard=cli.shard,done=ordinal+1,total=len(indices),global_index=index,elapsed=time.time()-start,
                        decoder_shapes=shape_trace)
            atomic_json(output/f'progress_shard{cli.shard}.json',status);print(json.dumps(status),flush=True)
    for h in handles:h.remove()
    assert all((output/'records'/f'{i:06d}.json').is_file() for i in indices)
    atomic_json(output/f'done_shard{cli.shard}.json',dict(indices=indices,count=len(indices),complete=True))


def aggregate(cli):
    import pyiqa
    import shutil
    from test_idea3_decoder_refinement import write_csv
    output=Path(cli.output_dir)
    manifests=[json.loads((output/f'manifest_shard{s}.json').read_text()) for s in range(cli.shards)]
    assert all(m==manifests[0] for m in manifests)
    count=cli.limit or 2741
    indices=[]
    for s in range(cli.shards):indices+=json.loads((output/f'done_shard{s}.json').read_text())['indices']
    assert len(indices)==len(set(indices))==count and sorted(indices)==list(range(count))
    records=[json.loads((output/'records'/f'{i:06d}.json').read_text()) for i in range(count)]
    gt=output/'fid_ground_truth';gt.mkdir(exist_ok=True)
    for record in records:
        p=gt/Path(record['gt_png']).name
        if not p.exists():os.link(record['gt_png'],p)
    fid=pyiqa.create_metric('fid',device='cuda')
    summaries=[]
    for name in manifests[0]['settings']:
        result_dir=output/name
        finished=result_dir/'summary.json'
        if finished.exists():
            existing=json.loads(finished.read_text());assert existing['count']==count
            summaries.append(existing);continue
        per_image=[dict(index=r['index'],scene=r['scene'],frame_id=r['frame_id'],
                        **{k:r['results'][name][k] for k in ('psnr','ssim','lpips','result_path')}) for r in records]
        assert len(list((result_dir/'images').glob('*.png')))==count
        with torch.inference_mode():fid_score=float(fid(str(result_dir/'images'),str(gt)).item())
        assert math.isfinite(fid_score)
        summary=dict(experiment=name,count=count,**{k:sum(r[k] for r in per_image)/count for k in ('psnr','ssim','lpips')},fid=fid_score)
        write_csv(str(result_dir/'per_image.csv'),per_image,('index','scene','frame_id','psnr','ssim','lpips','result_path'))
        scenes=defaultdict(list)
        for row in per_image:scenes[row['scene']].append(row)
        scene_rows=[dict(scene=scene,count=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in ('psnr','ssim','lpips')}) for scene,rows in sorted(scenes.items())]
        write_csv(str(result_dir/'per_scene.csv'),scene_rows,('scene','count','psnr','ssim','lpips'))
        atomic_json(finished,summary);summaries.append(summary);print(json.dumps(summary),flush=True)
    write_csv(str(output/'summary.csv'),summaries,('experiment','count','psnr','ssim','lpips','fid'))
    atomic_json(output/'complete.json',dict(count=count,settings=len(summaries),complete=True,summary=summaries))


if __name__=='__main__':
    cli=parse_args()
    aggregate(cli) if cli.aggregate else infer(cli)
