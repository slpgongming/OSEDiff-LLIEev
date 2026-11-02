"""最终方案的原子记录、权重摘要与完整数据集指标汇总。"""
from my_utils.quality_metrics import METRIC_NAMES, METRIC_PROTOCOL
import os,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import torch

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

def aggregate(cli):
    import pyiqa
    import shutil
    from test_idea3_decoder_refinement import write_csv
    output=Path(cli.output_dir)
    manifests=[json.loads((output/f'manifest_shard{s}.json').read_text()) for s in range(cli.shards)]
    assert all(m==manifests[0] for m in manifests)
    if manifests[0].get('metric_protocol') != METRIC_PROTOCOL:
        raise ValueError('Old metric protocol: rerun dual-channel evaluation in a new output directory.')
    count=manifests[0]['limit'] or manifests[0]['dataset_count']
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
            existing=json.loads(finished.read_text());assert existing['count']==count and set(METRIC_NAMES).issubset(existing)
            summaries.append(existing);continue
        per_image=[dict(index=r['index'],scene=r['scene'],frame_id=r['frame_id'],
                        **{k:r['results'][name][k] for k in (*METRIC_NAMES,'result_path')}) for r in records]
        assert len(list((result_dir/'images').glob('*.png')))==count
        with torch.inference_mode():fid_score=float(fid(str(result_dir/'images'),str(gt)).item())
        assert math.isfinite(fid_score)
        summary=dict(experiment=name,count=count,**{k:sum(r[k] for r in per_image)/count for k in METRIC_NAMES},fid=fid_score)
        write_csv(str(result_dir/'per_image.csv'),per_image,('index','scene','frame_id',*METRIC_NAMES,'result_path'))
        scenes=defaultdict(list)
        for row in per_image:scenes[row['scene']].append(row)
        scene_rows=[dict(scene=scene,count=len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in METRIC_NAMES}) for scene,rows in sorted(scenes.items())]
        write_csv(str(result_dir/'per_scene.csv'),scene_rows,('scene','count',*METRIC_NAMES))
        atomic_json(finished,summary);summaries.append(summary);print(json.dumps(summary),flush=True)
    write_csv(str(output/'summary.csv'),summaries,('experiment','count',*METRIC_NAMES,'fid'))
    atomic_json(output/'complete.json',dict(count=count,settings=len(summaries),complete=True,summary=summaries))

def configure_inference_numerics(reproducible):
    if reproducible:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch 2.0.1 的 replicate padding 反向不支持严格确定性，明确记录 warn-only。
        torch.use_deterministic_algorithms(True, warn_only=True)
    return dict(reproducible_requested=reproducible, torch_version=str(torch.__version__),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                determinism_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
