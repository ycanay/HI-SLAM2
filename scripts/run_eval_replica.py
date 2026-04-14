import os
import re
import ast
import numpy as np
import open3d as o3d
from collections import defaultdict
from glob import glob
from shutil import copyfile
from tqdm import tqdm

out = 'outputs/replica'
os.makedirs(f'{out}/meshes', exist_ok=True)
seqs = sorted(glob('data/Replica_semantics/ro*')) + sorted(glob('data/Replica_semantics/off*'))
print(f'Evaluating on {len(seqs)} sequences: {[os.path.basename(seq) for seq in seqs]}')
metrics = defaultdict(float)
for seq in tqdm(seqs, desc='Evaluating sequences', total=len(seqs)):
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f'python demo.py --imagedir {seq}/frames --gtdepthdir {seq}/depths '
    cmd += f'--config config/replica_config.yaml --calib calib/replica.txt --output {out}/{name} > {out}/{name}/log.txt'
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        os.system(cmd)

    # eval ate
    if not os.path.exists(f'{out}/{name}/ape.txt') or len(open(f'{out}/{name}/ape.txt').readlines()) < 10:
        os.system(f'evo_ape tum {seq}/traj_tum.txt {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
        os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
    metrics['ATE full'] += ATE
    print(f'- full ATE: {ATE:.4f}')

    # eval render
    psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
    print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
    metrics['PSNR'] += psnr['mean_psnr']
    metrics['SSIM'] += psnr['mean_ssim']
    metrics['LPIPS'] += psnr['mean_lpips']
    
    # eval panoptic segmentation
    cmd = f'python scripts/pq_new.py --scene {name} --run -1 --mapping none --save-json --output-base-path {out} > {out}/{name}/pq_none.txt'
    os.system(cmd)
    cmd = f'python scripts/pq_new.py --scene {name} --run -1 --mapping lifting --save-json --output-base-path {out} > {out}/{name}/pq_lifting.txt'
    os.system(cmd)

    def parse_pq_file(path):
        """Parse PQ, SQ, RQ, and mIoU from a pq output text file."""
        lines = open(path).read()
        pq_match = re.search(r'PQ=([0-9.]+)\s+SQ=([0-9.]+)\s+RQ=([0-9.]+)', lines)
        miou_match = re.search(r'mIoU=([0-9.]+)', lines)
        things_match = re.search(r'Mean \(things\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', lines)
        stuff_match = re.search(r'Mean \(stuff\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', lines)

        def _f(m, g): return float(m.group(g)) if m else float('nan')
        pq = _f(pq_match, 1); sq = _f(pq_match, 2); rq = _f(pq_match, 3)
        miou = _f(miou_match, 1)
        pq_th = _f(things_match, 1); sq_th = _f(things_match, 2); rq_th = _f(things_match, 3)
        pq_st = _f(stuff_match, 1); sq_st = _f(stuff_match, 2); rq_st = _f(stuff_match, 3)
        return pq, sq, rq, miou, pq_th, sq_th, rq_th, pq_st, sq_st, rq_st

    pq_n, sq_n, rq_n, miou_n, pq_th_n, sq_th_n, rq_th_n, pq_st_n, sq_st_n, rq_st_n = parse_pq_file(f'{out}/{name}/pq_none.txt')
    pq_l, sq_l, rq_l, miou_l, pq_th_l, sq_th_l, rq_th_l, pq_st_l, sq_st_l, rq_st_l = parse_pq_file(f'{out}/{name}/pq_lifting.txt')
    print(f'- PQ (none):    PQ={pq_n:.2f}  SQ={sq_n:.2f}  RQ={rq_n:.2f}  mIoU={miou_n:.2f}  PQ_th={pq_th_n:.2f}  PQ_st={pq_st_n:.2f}')
    print(f'- PQ (lifting): PQ={pq_l:.2f}  SQ={sq_l:.2f}  RQ={rq_l:.2f}  mIoU={miou_l:.2f}  PQ_th={pq_th_l:.2f}  PQ_st={pq_st_l:.2f}')
    metrics['PQ (none)'] += pq_n
    metrics['SQ (none)'] += sq_n
    metrics['RQ (none)'] += rq_n
    metrics['mIoU (none)'] += miou_n
    metrics['PQ_things (none)'] += pq_th_n
    metrics['SQ_things (none)'] += sq_th_n
    metrics['RQ_things (none)'] += rq_th_n
    metrics['PQ_stuff (none)'] += pq_st_n
    metrics['SQ_stuff (none)'] += sq_st_n
    metrics['RQ_stuff (none)'] += rq_st_n
    metrics['PQ (lifting)'] += pq_l
    metrics['SQ (lifting)'] += sq_l
    metrics['RQ (lifting)'] += rq_l
    metrics['mIoU (lifting)'] += miou_l
    metrics['PQ_things (lifting)'] += pq_th_l
    metrics['SQ_things (lifting)'] += sq_th_l
    metrics['RQ_things (lifting)'] += rq_th_l
    metrics['PQ_stuff (lifting)'] += pq_st_l
    metrics['SQ_stuff (lifting)'] += sq_st_l
    metrics['RQ_stuff (lifting)'] += rq_st_l

    # run tsdf fusion
    w = 2
    weight = f'w{w:.1f}'
    if not os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
        os.system(f'python tsdf_integrate.py --result {out}/{name} --voxel_size 0.006 --weight {w} > /dev/null')
        ply = o3d.io.read_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}.ply')
        ply = ply.transform(np.load(f'{out}/{name}/evo/alignment_transformation_sim3.npy'))
        o3d.io.write_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', ply)
        copyfile(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', f'{out}/meshes/{name}.ply')
    
    # eval 3d recon
    if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
        os.system(f'python scripts/eval_recon.py {out}/{name}/tsdf_mesh_{weight}_aligned.ply data/Replica/gt_mesh_culled/{name}.ply --eval_3d --save {out}/{name}/eval_recon_{weight}.txt > /dev/null')
    result = ast.literal_eval(open(f'{out}/{name}/eval_recon_{weight}.txt').read())
    metrics['accu'] += result['mean precision']
    metrics['comp'] += result['mean recall']
    metrics['compr'] += result['recall']
    print(f"- acc: {result['mean precision']:.3f}, comp: {result['mean recall']:.3f}, comp rat: {result['recall']:.3f}\n")

for r in metrics:
    print(f'{r}: \t {metrics[r]/len(seqs):.4f}')

