"""E-SFR：冻结 Stage3，联合微调事件结构目标和所选 decoder 融合层；一步可微引导，独立固定置信度。

torchrun 一卡一进程，外层梯度完成后手动平均，避免 DDP reducer 与 autograd.grad 冲突。
训练/验证按原场景划分；固定验证帧仅用于本次微调选模，不读取正式 test 图像。
"""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import set_seed
from dataloaders.sde_event_structure_dataset import SDEEventStructureDataset,StructureEventConfig
from models.decoder_refinement_control import NoH1DecoderRefinementGen
from models.event_structure_head import load_structure_checkpoint
from models.event_structure_guidance import GuidanceConfig
from models.event_lora_utils import iter_event_lora_modules,clear_event_feats_for_all_event_lora
from models.image_structure import image_structure
from models.structure_supervision import StructureSupervisionConfig,structure_supervision_loss
from models.esfr import guided_forward
from my_utils.evaluation import configure_inference_numerics
from train_event_structure_head import read_scene_split
from train_idea3_decoder_refinement import event_edge_loss
from my_utils.evaluation import atomic_json,file_hash
from models.fusion_selection import parse_fusion_layers, select_fusion_layers


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4_run',required=True)
    p.add_argument('--structure_checkpoint', '--h1_checkpoint', dest='h1_checkpoint', required=True)
    p.add_argument('--fusion_layers', type=parse_fusion_layers, default=(1,2,3), help='Active and trainable decoder layers; default d1,d2,d3.')
    p.add_argument('--stage4_checkpoint', default='checkpoints/model_final.pkl')
    p.add_argument('--scene_split',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--max_steps',type=int,default=1000)
    p.add_argument('--validation_every',type=int,default=250)
    p.add_argument('--validation_per_scene',type=int,default=12)
    p.add_argument('--refinement_lr',type=float,default=1e-5)
    p.add_argument('--head_lr',type=float,default=1e-6)
    p.add_argument('--lambda_structure',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume_from_checkpoint')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not a.fusion_layers:p.error('Joint training requires at least one fusion layer.')
    for path in [Path(a.stage4_run)/'run_config.json', Path(a.stage4_run)/a.stage4_checkpoint, Path(a.h1_checkpoint), Path(a.scene_split)]:
        if not path.is_file():p.error(f'Missing input: {path}')
    if min(a.max_steps,a.validation_every,a.validation_per_scene,a.refinement_lr,a.head_lr,a.lambda_structure)<=0:p.error('Positive parameters required')
    return a


def frozen_hash(gen,teacher,head):
    """精确核对全部冻结参数（包括 d0、教师共享特征和原置信度末端）。"""
    h=hashlib.sha256()
    for prefix,module in [('gen',gen),('teacher',teacher),('head',head)]:
        for name,p in module.named_parameters():
            if p.requires_grad:continue
            h.update((prefix+'.'+name).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clear(gen):
    clear_event_feats_for_all_event_lora(gen.unet)
    for _,layer in iter_event_lora_modules(gen.unet):
        layer.last_aux={};layer.last_visuals={};layer.last_z_delta=None


def build(a,rank,load_data=True):
    run=Path(a.stage4_run);train=json.loads((run/'run_config.json').read_text())
    final=run/getattr(a,'stage4_checkpoint','checkpoints/model_final.pkl')
    cfg=argparse.Namespace(**train);cfg.decoder_refinement_enabled=False
    gen=NoH1DecoderRefinementGen(cfg)
    gen.load_stage1_full(cfg.stage1_checkpoint)
    gen.load_event_checkpoint(torch.load(cfg.event_lora_checkpoint,map_location='cpu',weights_only=False))
    cfg.decoder_refinement_enabled=True;gen.decoder_refinement_enabled=True
    gen.install_decoder_refinement()
    checkpoint=torch.load(final,map_location='cpu',weights_only=False)
    gen.load_event_checkpoint(checkpoint);del checkpoint
    gen.activate_lora_adapters(cfg.unet_lora_active_parts)
    gen.eval().requires_grad_(False).cuda()
    # 旧 run_config 无层配置时继承原联合训练 d123；新配置显式保存。
    select_fusion_layers(gen.vae.decoder, getattr(a,'fusion_layers',(1,2,3)), trainable=True)
    gen.decoder_fusion_weight=1.
    for m in gen.modules():
        if hasattr(m,'gradient_checkpointing'):m.gradient_checkpointing=False
    head,meta,operator=load_structure_checkpoint(a.h1_checkpoint,require_calibrated=True)
    teacher=copy.deepcopy(head).eval().requires_grad_(False).cuda()
    head.cuda().eval();head.set_training_phase('structure')
    if not load_data:return gen,head,teacher,meta,operator,None,None,None,train
    split=read_scene_split(a.scene_split)
    evcfg=StructureEventConfig.from_dict(meta['event_config'])
    trainset=SDEEventStructureDataset('train',cfg,evcfg,cfg.train_data_root,scenes=split['train'])
    valset=SDEEventStructureDataset('test',cfg,evcfg,cfg.train_data_root,scenes=split['validation'])
    validation=[]
    for scene in split['validation']:
        ids=[i for i,s in enumerate(valset.samples) if s.scene_name==scene]
        validation.extend(ids[i] for i in np.linspace(0,len(ids)-1,a.validation_per_scene,dtype=int))
    assert len(validation)==len(set(validation))
    return gen,head,teacher,meta,operator,trainset,valset,validation,train


def main():
    a=parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:dist.init_process_group('nccl')
    set_seed(a.seed);configure_inference_numerics(True)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not a.resume_from_checkpoint:raise RuntimeError('Existing run requires explicit resume')
    gen,head,teacher,meta,operator,trainset,valset,validation,base_config=build(a,rank)
    rp=[p for p in gen.parameters() if p.requires_grad];hp=[p for p in head.parameters() if p.requires_grad]
    params=rp+hp
    optimizer=torch.optim.AdamW([dict(params=rp,lr=a.refinement_lr),dict(params=hp,lr=a.head_lr)],betas=(.9,.999),weight_decay=0.,eps=1e-8)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    import pyiqa
    lpips=pyiqa.create_metric('lpips',device='cuda',as_loss=True).eval().requires_grad_(False)
    metrics={k:pyiqa.create_metric(k,device='cuda',**({'test_y_channel':True,'color_space':'ycbcr'} if k in ['psnr','ssim'] else {})).eval().requires_grad_(False) for k in ['psnr','ssim','lpips']}
    guidance=GuidanceConfig(steps=1)
    supervision=StructureSupervisionConfig(**meta['supervision_config']) if 'supervision_config' in meta else StructureSupervisionConfig(mode='region_balanced')
    initial_hash=frozen_hash(gen,teacher,head)
    initial_ref={n:p.detach().cpu().clone() for n,p in gen.named_parameters() if p.requires_grad} if a.smoke else None
    initial_head={n:p.detach().cpu().clone() for n,p in head.named_parameters() if p.requires_grad} if a.smoke else None
    signature=dict(format='joint-h1-refinement-v1',stage4_sha256=file_hash(Path(a.stage4_run)/a.stage4_checkpoint),stage3_sha256=file_hash(base_config['event_lora_checkpoint']),h1_sha256=file_hash(a.h1_checkpoint),levels=list(a.fusion_layers),weight=1.,confidence='frozen_initial_full_head',guidance=asdict(guidance),structure_supervision=asdict(supervision),world_size=world,global_batch_size=world,seed=a.seed,split=read_scene_split(a.scene_split),validation_indices=validation,refinement_lr=a.refinement_lr,head_lr=a.head_lr,lambda_structure=a.lambda_structure,max_steps=a.max_steps)
    step0=0;best=float('-inf')
    if a.resume_from_checkpoint:
        c=torch.load(a.resume_from_checkpoint,map_location='cpu',weights_only=False);assert c['signature']==signature and c['frozen_sha256']==initial_hash
        gen.vae.decoder.feature_fusion_blocks.load_state_dict(c['fusion']);head.load_state_dict(c['head']);teacher.load_state_dict(c['teacher'])
        optimizer.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler']);step0=c['step'];best=c['best_psnr'];del c
    if rank==0:atomic_json(out/'run_config.json',dict(**vars(a),signature=signature,base_train_config=base_config,train_count=len(trainset),validation_count=len(validation),frozen_initial_sha256=initial_hash,gradient_mode='exact_one_step_unroll_piecewise_acceptance',refinement_precision='fp16',backbone_precision='fp32',trainable_refinement=sum(p.numel() for p in rp),trainable_head=sum(p.numel() for p in hp)))
    def save(step,label):
        if rank==0:
            path=out/(label+'.pt');temp=path.with_suffix('.tmp')
            torch.save(dict(format=signature['format'],step=step,signature=signature,frozen_sha256=initial_hash,
                fusion=gen.vae.decoder.feature_fusion_blocks.state_dict(),head=head.state_dict(),teacher=teacher.state_dict(),
                head_metadata=meta,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),best_psnr=best),temp)
            os.replace(temp,path)
        if world>1:dist.barrier()
    def validate(step):
        rows=[]
        for index in validation[rank::world]:
            set_seed(a.seed+100000+index);sample=valset[index];target=sample['output_pixel_values'][None].cuda()
            row=dict(index=index,scene=sample['scene_name'])
            for enabled in [False,True]:
                set_seed(a.seed+200000+index)
                with torch.enable_grad():pred,edges,stats=guided_forward(gen,head,teacher,sample,guidance,operator,enabled=enabled,create_graph=False)
                with torch.no_grad():
                    x=(pred.float()+1)/2;y=(target.float()+1)/2
                    row['on' if enabled else 'off']={k:float(m(x,y)) for k,m in metrics.items()}
                    if enabled:row['structure_loss']=float(structure_supervision_loss(edges,image_structure(y,operator),supervision))
                del pred,edges;clear(gen)
            rows.append(row)
        if world>1:
            gathered=[None]*world;dist.all_gather_object(gathered,rows);rows=[r for part in gathered for r in part]
        assert len(rows)==len(validation) and len({r['index'] for r in rows})==len(rows)
        result=dict(step=step,count=len(rows),on={k:float(np.mean([r['on'][k] for r in rows])) for k in metrics},off={k:float(np.mean([r['off'][k] for r in rows])) for k in metrics},structure_loss=float(np.mean([r['structure_loss'] for r in rows])),rows=sorted(rows,key=lambda r:r['index']))
        if rank==0:atomic_json(out/f'validation_{step:06d}.json',result);print('VALIDATION',step,result['on'],result['off'],flush=True)
        optimizer.zero_grad(set_to_none=True)
        return result['on']['psnr']
    if not a.smoke and step0==0:
        best=validate(0);save(0,'checkpoint_best');save(0,'checkpoint_last')
    started=time.time();order=None;epoch=-1
    for step in range(step0+1,a.max_steps+1):
        position=(step-1)*world+rank;current_epoch=position//len(trainset)
        if current_epoch!=epoch:
            epoch=current_epoch;order=np.random.default_rng(a.seed+epoch).permutation(len(trainset))
        index=int(order[position%len(trainset)]);set_seed(a.seed+step*world+rank)
        sample=trainset[index];target=sample['output_pixel_values'][None].cuda()
        optimizer.zero_grad(set_to_none=True)
        pred,edges,stats=guided_forward(gen,head,teacher,sample,guidance,operator,create_graph=True)
        l2=F.mse_loss(pred.float(),target.float())
        perceptual=lpips((pred.float()+1)/2,(target.float()+1)/2).mean()
        assert perceptual.requires_grad,'LPIPS must participate in outer backward'
        edge=event_edge_loss(pred.float(),target.float(),sample['event_voxel_raw'][None].cuda())
        anchor=structure_supervision_loss(edges,image_structure((target.float()+1)/2,operator),supervision)
        image_loss=l2+.1*perceptual+.1*edge
        # 首次真实反传单独验证图像损失→H1，排除只靠结构监督更新的假联合训练。
        image_head_norm=None
        if step==1:
            gs=torch.autograd.grad(image_loss,hp,retain_graph=True,allow_unused=True)
            image_head_norm=sum(float(g.detach().float().square().sum()) for g in gs if g is not None)**.5
            assert np.isfinite(image_head_norm) and image_head_norm>0,'Image loss does not reach H1'
            del gs
        loss=image_loss+a.lambda_structure*anchor
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        for p in params:
            if p.grad is None:p.grad=torch.zeros_like(p)
            if world>1:dist.all_reduce(p.grad);p.grad.div_(world)
            if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite outer gradient')
        rn=float(torch.nn.utils.clip_grad_norm_(rp,1.));hn=float(torch.nn.utils.clip_grad_norm_(hp,1.))
        assert not any(p.grad is not None for p in gen.parameters() if not p.requires_grad)
        assert not any(p.grad is not None for p in teacher.parameters())
        scaler.step(optimizer);scaler.update()
        values=torch.tensor([float(loss),float(l2),float(perceptual),float(edge),float(anchor),stats['accepted_steps'],rn,hn],device='cuda')
        if world>1:dist.all_reduce(values);values/=world
        if rank==0:
            record=dict(step=step,**dict(zip(['loss','l2','lpips','event_edge','structure','accepted_fraction','refinement_grad_norm','head_grad_norm'],values.tolist())),image_loss_head_grad_norm=image_head_norm,elapsed=time.time()-started,max_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if step==1 or step%10==0:print(json.dumps(record),flush=True)
        del pred,edges,loss,image_loss,l2,perceptual,edge,anchor;clear(gen)
        if a.smoke:save(step,f'checkpoint_{step:06d}')
        if not a.smoke and (step%a.validation_every==0 or step==a.max_steps):
            score=validate(step)
            if score>best:best=score;save(step,'checkpoint_best')
            save(step,'checkpoint_last');save(step,f'checkpoint_{step:06d}')
    final_hash=frozen_hash(gen,teacher,head);assert final_hash==initial_hash
    if a.smoke:
        changed_ref=[n for n,p in gen.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),initial_ref[n])]
        changed_head=[n for n,p in head.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),initial_head[n])]
        assert changed_ref and changed_head
        digest=hashlib.sha256()
        for p in params:digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        digests=[digest.hexdigest()]
        if world>1:
            digests=[None]*world;dist.all_gather_object(digests,digest.hexdigest())
        assert len(set(digests))==1,'Ranks diverged after gradient synchronization'
        if rank==0:atomic_json(out/'smoke_audit.json',dict(frozen_bitwise_unchanged=True,changed_refinement=changed_ref,changed_head=changed_head,image_loss_head_grad_norm=image_head_norm if a.max_steps==1 else 'verified_step1',synchronized_parameter_sha256=digests,max_memory_gib=torch.cuda.max_memory_allocated()/2**30))
    save(a.max_steps,'checkpoint_final')
    if rank==0:atomic_json(out/'complete.json',dict(step=a.max_steps,frozen_bitwise_unchanged=True,best_validation_psnr=best if np.isfinite(best) else None))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
