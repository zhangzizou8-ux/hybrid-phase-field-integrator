"""Frozen No-local runner. Original F9a_run.py and run03.py are imported unchanged.
Instrumentation records calls only; no physical or learned updates are replaced.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['NVIDIA_TF32_OVERRIDE']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,hashlib,traceback,importlib.util,platform
import numpy as np
import torch
import cupy as cp
HERE=Path(__file__).resolve().parent;SRC=HERE/'source'
EXPECTED='fc30c893680e7d564888a891c368ff39d587fff86e9e63fb7dcc8926c04c76e0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def schedule():return list(range(2000,20001,2000))+[20000+6000*i+2000*(i//40) for i in range(1,81)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=int,required=True);ap.add_argument('--repeat',action='store_true');args=ap.parse_args()
 cfg=json.loads((HERE/'inference_config.json').read_text(encoding='utf8'));root=Path(cfg['reference_root']);weight=HERE/'weights/F9a.pth'
 assert sha(weight)==EXPECTED
 if not torch.cuda.is_available():raise RuntimeError('Formal run requires CUDA')
 torch.manual_seed(cfg['torch_seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 folder=HERE/('repeatability' if args.repeat else 'predictions')/f'Run_{args.run:02d}'
 if folder.exists() and any(folder.iterdir()):raise RuntimeError(f'Will not overwrite existing outputs: {folder}')
 folder.mkdir(parents=True,exist_ok=True);logs=HERE/'logs';logs.mkdir(exist_ok=True)
 tag=('repeat_' if args.repeat else '')+f'Run_{args.run:02d}';mp=logs/(tag+'_manifest.json')
 ck=torch.load(weight,map_location='cpu',weights_only=True)
 assert ck['frame_step']==3 and ck['seq_len']==10 and ck['unroll']==40
 record=dict(success=False,run_id=args.run,repeat=args.repeat,weight_sha256=sha(weight),run03_sha256=sha(SRC/'run03.py'),entry_sha256=sha(SRC/'图/图图/图9/F9a_run.py'),runner_sha256=sha(__file__),model_key='model_state_dict',checkpoint={k:ck[k] for k in ['epoch','unroll','frame_step','seq_len','norm_params']},device=torch.cuda.get_device_name(0),precision=dict(network='float32',solver='float64',allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True),current_learned_step=0,bursts=[],local_correction_calls=0,files=[])
 def flush():mp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
 flush()
 sys.path.insert(0,str(SRC));import run03 as base
 assert Path(base.__file__).resolve()==(SRC/'run03.py').resolve()
 Original=base.PhaseFieldSimulator
 class AuditedFOM(Original):
  def inject_state(self,a):
   if hasattr(self,'_observed_burst'):self._observed_burst['fine_steps']=self.step_count-self._observed_burst['counter_start']
   super().inject_state(a)
   if record['current_learned_step']:
    row=dict(learned_step=record['current_learned_step'],counter_start=int(self.step_count),fine_steps=0,physical_endpoint=20000+6000*record['current_learned_step']+2000*(record['current_learned_step']//40),injected_concentration_mean=float(np.asarray(a[0],dtype=float).mean()))
    record['bursts'].append(row);self._observed_burst=row;flush()
  def _single_step(self):
   super()._single_step()
   if hasattr(self,'_observed_burst'):self._observed_burst['fine_steps']=int(self.step_count)-self._observed_burst['counter_start']
 base.PhaseFieldSimulator=AuditedFOM
 def prohibited(*a,**k):
  record['local_correction_calls']+=1;raise RuntimeError('No-local must not call local correction')
 base.physics_correct=prohibited
 spec=importlib.util.spec_from_file_location('frozen_entry',SRC/'图/图图/图9/F9a_run.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 def progress(iterable,**kwargs):
  for n in iterable:
   record['current_learned_step']=int(n)+1
   yield n
   if (n+1)%10==0:flush();print(tag,'learned',n+1,'/80',flush=True)
 entry.tqdm=progress
 framepaths=[root/f'Run_{args.run:02d}'/f'frame_{s:06d}.npy' for s in schedule()[:10]]
 sys.argv=['F9a_run.py','--model',str(weight),'--output',str(folder),'--total-steps','80','--pde-correct','40','--run-id',str(args.run),'--frames',*[str(p) for p in framepaths]]
 record['entry_argv']=sys.argv;flush()
 try:
  entry.main();cp.cuda.Stream.null.synchronize();torch.cuda.synchronize()
  fs=sorted(folder.glob('frame_*.npy'));assert [int(p.stem[6:]) for p in fs]==schedule()
  for p in fs:
   a=np.load(p);step=int(p.stem[6:]);assert a.shape==(7,512,512) and a.dtype==np.float32 and np.isfinite(a).all()
   assert a[0].min()>=.08-1e-7 and a[0].max()<=.24+1e-7 and a[1:].min()>=0 and a[1:].max()<=1
   row=dict(path=str(p),pde_step=step,T_star=step*.0005,shape=list(a.shape),dtype=str(a.dtype),SHA256=sha(p),c_min=float(a[0].min()),c_max=float(a[0].max()),eta_min=float(a[1:].min()),eta_max=float(a[1:].max()))
   if step<=20000:
    g=root/f'Run_{args.run:02d}'/p.name;assert np.array_equal(a,np.load(g).astype(np.float32));row['history_reference_SHA256']=sha(g)
   record['files'].append(row)
  assert [(b['learned_step'],b['fine_steps'],b['physical_endpoint']) for b in record['bursts']]==[(40,2000,262000),(80,2000,504000)]
  assert record['local_correction_calls']==0
  record['success']=True;record['count']=len(fs);record['terminal']=504000;flush();print(tag,'ACCEPTED',len(fs),'files',flush=True)
 except Exception:
  record['error']=traceback.format_exc();flush();raise
if __name__=='__main__':main()
