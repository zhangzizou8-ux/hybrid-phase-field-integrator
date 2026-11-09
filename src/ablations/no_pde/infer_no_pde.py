"""No-PDE/mean retained: original run03 solver/model, 80 learned steps + 2 full FOM bursts.
python run_no_pde_mean.py --runs 0   # acceptance
python run_no_pde_mean.py --runs 31 32 33 34 35
Existing complete runs are audited and reused; incomplete runs are never aggregated.
"""
import os,sys,json,hashlib,argparse,importlib.util,traceback,time
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1] if HERE.name=='取消PDE修正' else HERE.parents[0]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def steps():return list(range(2000,20001,2000))+[20000+6000*i+2000*(i//40) for i in range(1,81)]
def audit(folder):
    fs=sorted(folder.glob('frame_*.npy'));assert [int(p.stem[6:]) for p in fs]==steps()
    means=[]
    for f in fs:
        a=np.load(f);assert a.shape==(7,512,512) and a.dtype==np.float32 and np.isfinite(a).all()
        assert a[0].min()>=.08-1e-7 and a[0].max()<=.24+1e-7 and a[1:].min()>=0 and a[1:].max()<=1
        means.append(float(a[0].astype('float64').mean()))
    return dict(count=len(fs),terminal=504000,timestamps=steps(),max_mean_shift_vs_history=max(abs(x-means[9]) for x in means[10:]))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',nargs='+',type=int,default=[0,31,32,33,34,35]);ap.add_argument('--project',default=str(PROJECT));args=ap.parse_args()
    root=Path(args.project);base=load(root/'run03.py','original_run03');train=load(HERE/'01_train_no_pde_keep_mean.py','no_pde_train')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for formal inference')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weight=HERE/'best_no_pde_mean.pt';ck=torch.load(weight,map_location='cpu',weights_only=False)
    assert ck['formal_best'] and ck['unroll']==40 and ck['seq_len']==10 and ck['frame_step']==3
    cfg=ck['training_config'];assert cfg['batch']==2 and cfg['gradient_accumulation']==1 and not cfg['local_pde_correction'] and cfg['normalized_mean_control']
    assert ck['local_pde_correction'] is False and ck['normalized_mean_control'] is True
    assert ck['model_state_dict']['enc1.0.weight'].shape[1]==9 and ck['model_state_dict']['dec1.weight'].shape[0]==7
    model=base.ConvLSTMUNet(9,7).cuda().eval();model.load_state_dict(ck['model_state_dict']);norm=ck['norm_params']
    out=HERE/'results_no_pde_mean';ev=out/'evaluation';ev.mkdir(parents=True,exist_ok=True);mp=ev/'inference_manifest.json'
    manifest=json.loads(mp.read_text(encoding='utf8')) if mp.exists() else dict(runs={})
    manifest.update(weight=str(weight),weight_sha256=sha(weight),model_key='model_state_dict',checkpoint={k:ck[k] for k in ['kind','epoch','unroll','formal_best','training_config','norm_params']},device=torch.cuda.get_device_name(0),run03_sha256=sha(root/'run03.py'),training_sha256=sha(HERE/'01_train_no_pde_keep_mean.py'),local_pde_correction=False,fom_burst_interval=40,history=10,learned_steps=80,input_channels=9,output_channels=7)
    def save():mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    for rid in args.runs:
        name=f'Run_{rid:02d}';folder=out/name
        if manifest['runs'].get(name,{}).get('success'):
            assert manifest['runs'][name]['weight_sha256']==sha(weight);audit(folder);print(name,'audited existing complete output',flush=True);continue
        if folder.exists() and list(folder.glob('frame_*.npy')):raise RuntimeError(f'Incomplete output exists: {folder}; inspect before rerunning')
        folder.mkdir(exist_ok=True);info=dict(success=False,source=str(root/'run'/name),bursts=[],weight_sha256=sha(weight));manifest['runs'][name]=info;save()
        try:
            raw=[np.load(root/'run'/name/f'frame_{s:06d}.npy').astype(np.float32) for s in steps()[:10]]
            def normalize(a):
                t=torch.from_numpy(base.add_curvature(a)).cuda();t[0]=(t[0]-norm['c_min'])/(norm['c_max']-norm['c_min'])*2-1;t[1:7]=t[1:7]*2-1;return t
            history=torch.stack([normalize(a) for a in raw])[None]
            # Fixed initial-history target, exactly the original Hybrid scalar convention.
            mean_physical=float(raw[-1][0].mean());target=(mean_physical-norm['c_min'])/(norm['c_max']-norm['c_min'])*2-1
            pde=base.PhaseFieldSimulator(run_id=rid);pde.inject_state(base.add_curvature(raw[-1]))
            for s,a in zip(steps()[:10],raw):np.save(folder/f'frame_{s:06d}.npy',a)
            current=20000;last=None
            with torch.no_grad():
                for i in range(1,81):
                    if i==1:pred,(h,c)=model(history)
                    else:pred,h,c=model.forward_step(last.squeeze(1),h,c)
                    pred=train.range_project(pred,norm)
                    cn=pred[:,:,0];drift=target-cn.mean().item()
                    pred[:,:,0]=(cn+drift).clamp((.08-norm['c_min'])/(norm['c_max']-norm['c_min'])*2-1,(.24-norm['c_min'])/(norm['c_max']-norm['c_min'])*2-1)
                    a=pred[0,0].clone();a[0]=(a[0]+1)/2*(norm['c_max']-norm['c_min'])+norm['c_min'];a[1:]=(a[1:]+1)/2;a=a.cpu().numpy();current+=6000
                    if i%40==0:
                        pde.inject_state(a);before=pde.step_count
                        for _ in range(2000):pde._single_step()
                        assert pde.step_count-before==2000
                        a=np.stack([base.cp.asnumpy(pde.c).astype(np.float32)]+[base.cp.asnumpy(e).astype(np.float32) for e in pde.etas]);current+=2000
                        info['bursts'].append(dict(learned_step=i,fine_steps=2000,physical_endpoint=current))
                    assert current==steps()[9+i] and np.isfinite(a).all()
                    np.save(folder/f'frame_{current:06d}.npy',a.astype(np.float32));last=normalize(a)[None,None]
                    if i%10==0:print(name,i,'/80',current,flush=True)
            assert len(info['bursts'])==2
            info.update(audit(folder));info['success']=True;info['initial_mean_target']=mean_physical;save();print(name,'PASS 90 frames',flush=True)
        except Exception:
            info['error']=traceback.format_exc();save();raise
if __name__=='__main__':main()
