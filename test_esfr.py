"""E-SFR 联合权重完整测试：恢复学生与固定教师，逐帧配对结构修正开/关。"""
import argparse,json,math,time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from train_esfr import build,frozen_hash,clear
from models.event_structure_guidance import GuidanceConfig
from models.esfr import guided_forward
from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset,StructureEventConfig
from my_utils.evaluation import configure_inference_numerics
from my_utils.evaluation import atomic_json,file_hash,aggregate
from models.fusion_selection import parse_fusion_layers,select_fusion_layers
from test_idea3_decoder_refinement import quality_metrics
from test_event_film_lora_osediff import tensor_to_pil


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint_run',required=True)
    p.add_argument('--checkpoint',default='checkpoint_best.pt')
    p.add_argument('--expected_sha256')
    p.add_argument('--fusion_layers',type=parse_fusion_layers,default=None,help='Default: inherit training layers; explicit subset enables inference ablation.')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--test_root',required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--aggregate',action='store_true')
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.shards<1 or not 0<=a.shard<a.shards or a.limit<0:p.error('Invalid shard/limit')
    return a


def infer(a):
    import pyiqa
    run=Path(a.joint_run);config=json.loads((run/'run_config.json').read_text())
    checkpoint=run/a.checkpoint
    actual_hash=file_hash(checkpoint)
    if a.expected_sha256 and actual_hash!=a.expected_sha256:raise ValueError('Checkpoint SHA256 mismatch')
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert bundle['format']=='joint-h1-refinement-v1'
    checkpoint_step=bundle['step']
    assert json.dumps(bundle['signature'],sort_keys=True)==json.dumps(config['signature'],sort_keys=True)
    assert bundle['signature']['weight']==1.
    set_seed(a.seed);numerics=configure_inference_numerics(True)
    gen,head,teacher,meta,operator,_,valset,validation,_=build(argparse.Namespace(**config),0,load_data=a.audit)
    gen.vae.decoder.feature_fusion_blocks.load_state_dict(bundle['fusion'],strict=True)
    head.load_state_dict(bundle['head'],strict=True);teacher.load_state_dict(bundle['teacher'],strict=True)
    # 按训练的可训练/冻结集合校验；再把全部参数冻结供推理，仅临时delta可求导。
    assert frozen_hash(gen,teacher,head)==bundle['frozen_sha256']
    for name,module in [('fusion',gen.vae.decoder.feature_fusion_blocks),('head',head),('teacher',teacher)]:
        assert all(torch.equal(v.cpu(),bundle[name][k]) for k,v in module.state_dict().items())
    for m in [gen,head,teacher]:m.eval().requires_grad_(False)
    levels=parse_fusion_layers(bundle['signature']['levels']) if a.fusion_layers is None else a.fusion_layers
    select_fusion_layers(gen.vae.decoder,levels)
    if a.audit and list(levels)!=bundle['signature']['levels']:raise ValueError('Audit requires training fusion layers')
    guidance=GuidanceConfig(**bundle['signature']['guidance'])
    dataset=SDEEventStructureDataset('test',gen.args,StructureEventConfig.from_dict(meta['event_config']),a.test_root)
    assert len(dataset)>0 and a.limit<=len(dataset)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    settings={name:dict(levels=list(levels),weight=1.,guidance_enabled=enabled) for name,enabled in [('ESFR_off',False),('ESFR_on',True)]}
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=actual_hash,checkpoint_step=checkpoint_step,
        joint_signature=bundle['signature'],numeric_settings=numerics,seed=a.seed,test_root=a.test_root,
        dataset_count=len(dataset),limit=a.limit,shards=a.shards,settings=settings,teacher='frozen_initial_full_head',
        backbone_dtype='float32',refinement_autocast='float16',uses_gt_for_guidance=False,
        metric_protocol='PSNR(Y)/SSIM(Y) ycbcr; LPIPS float RGB; whole-set FID from PNG')
    # JSON规范化tuple/list，避免恢复时只因序列类型不同拒绝同一配置。
    manifest=json.loads(json.dumps(manifest))
    mp=out/f'manifest_shard{a.shard}.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Resume manifest mismatch'
    else:atomic_json(mp,manifest)
    del bundle
    metrics=dict(psnr=pyiqa.create_metric('psnr',test_y_channel=True,color_space='ycbcr').cuda(),
        ssim=pyiqa.create_metric('ssim',test_y_channel=True,color_space='ycbcr').cuda(),lpips=pyiqa.create_metric('lpips',device='cuda').cuda())
    def predict(sample,seed,enabled):
        # 显式剔除GT，保证完整decoder引导接口只收到图像、事件和提示词。
        inputs={k:sample[k] for k in ['conditioning_pixel_values','events','events_refinement','events_structure','prompt','neg_prompt']}
        set_seed(seed)
        with torch.enable_grad():image,_,stats=guided_forward(gen,head,teacher,inputs,guidance,operator,enabled=enabled,create_graph=False)
        clear(gen)
        return image.detach(),stats
    if a.audit:
        old=json.loads((run/f'validation_{checkpoint_step:06d}.json').read_text());audits=[]
        for index in [validation[0],validation[-1]]:
            set_seed(a.seed+100000+index);sample=valset[index]
            previous=next(x for x in old['rows'] if x['index']==index)
            for enabled in [False,True]:
                image,_=predict(sample,a.seed+200000+index,enabled)
                with torch.no_grad():values=quality_metrics(image,sample['output_pixel_values'][None].cuda(),metrics)
                diffs={k:abs(values[k]-previous['on' if enabled else 'off'][k]) for k in values}
                assert diffs['psnr']<.02 and diffs['ssim']<1e-4 and diffs['lpips']<1e-4,diffs
                audits.append(dict(index=index,guidance=enabled,absolute_metric_differences=diffs))
        atomic_json(out/'load_audit.json',dict(step=checkpoint_step,all_loaded_tensors_exact=True,frozen_hash_matched=True,validation_replay=audits))
    start=time.time();indices=list(range(a.limit or len(dataset)))[a.shard::a.shards]
    for ordinal,index in enumerate(indices):
        rp=out/'records'/f'{index:06d}.json'
        if rp.exists():
            record=json.loads(rp.read_text())
            if set(record['results'])==set(settings) and all(Path(x['result_path']).is_file() for x in record['results'].values()):continue
        set_seed(a.seed+index);sample=dataset[index]
        stem=f'{index:06d}_{sample["scene_name"]}_{sample["frame_index"]+1:03d}'
        refs=out/'references';refs.mkdir(exist_ok=True)
        low=refs/f'{stem}_ll.png';gt=refs/f'{stem}_gt.png'
        tensor_to_pil(sample['conditioning_pixel_values']).save(low);tensor_to_pil(sample['output_pixel_values']).save(gt)
        results={}
        for name,setting in settings.items():
            image,stats=predict(sample,a.seed+index,setting['guidance_enabled'])
            path=out/name/'images'/f'{stem}_result.png';path.parent.mkdir(parents=True,exist_ok=True)
            tensor_to_pil(image[0]).save(path)
            with torch.no_grad():values=quality_metrics(image,sample['output_pixel_values'][None].cuda(),metrics)
            assert all(math.isfinite(v) for v in values.values())
            results[name]=dict(**values,result_path=str(path),png_sha256=file_hash(path),h1_diagnostics=stats)
            del image
        atomic_json(rp,dict(index=index,scene=sample['scene_name'],frame_id=sample['frame_id'],frame_index=sample['frame_index'],
            seed=a.seed+index,low_path=sample['low_path'],normal_path=sample['normal_path'],low_png=str(low),gt_png=str(gt),results=results))
        if ordinal%10==0 or ordinal+1==len(indices):
            progress=dict(shard=a.shard,done=ordinal+1,total=len(indices),elapsed=time.time()-start)
            atomic_json(out/f'progress_shard{a.shard}.json',progress);print(json.dumps(progress),flush=True)
    assert all((out/'records'/f'{i:06d}.json').is_file() for i in indices)
    atomic_json(out/f'done_shard{a.shard}.json',dict(indices=indices,count=len(indices),complete=True))


if __name__=='__main__':
    args=parse_args()
    aggregate(args) if args.aggregate else infer(args)
