"""FWHM only. One frozen extractor for all methods; old extraction is diagnostic only."""
from pathlib import Path
import sys,json,argparse,hashlib,importlib.util
import numpy as np,pandas as pd
from fwhm_unified import ROOT,OUT,RUNS,CONFIG,steps,sha,sample,extract,candidates
METHODS=['Hybrid','No-PDE','No-local']
H=ROOT/'投稿改/论文修改/V1.5/V1.5_NoLocal_Unified'
NP=ROOT/'投稿改/取消PDE修正/results_no_pde_mean'
def csv(d,p):pd.DataFrame(d).to_csv(p,index=False,encoding='utf-8-sig')
def folder(method,rid):
 return (OUT/'predictions/Hybrid' if method=='Hybrid' else NP if method=='No-PDE' else OUT/'no_local_bundle/predictions')/f'Run_{rid:02d}'
def evaluate(method,rid):
 gate=json.loads((OUT/'reference_gate.json').read_text());assert gate['passed'] and gate['evaluation_SHA256']==sha(OUT/'fwhm_unified.py')
 ev=OUT/'per_run';ev.mkdir(exist_ok=True);dest=ev/f'{method}_Run_{rid:02d}_profiles.csv'
 assert not dest.exists(),'Existing output will not be silently reused'
 c,_=candidates();c=c[c.run_id==f'Run_{rid:02d}']
 refs=pd.read_csv(OUT/'reference_profile_verification.csv');refs=refs[refs.Run==f'Run_{rid:02d}'].set_index(['Physical_step','Candidate_ID'])
 spec=importlib.util.spec_from_file_location('old_diagnostic',OUT/'no_local_bundle/source/historical_evaluate_no_pde_mean.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 f=folder(method,rid);fs=sorted(f.glob('*.npy'));assert len(fs)==90 and [int(p.stem[6:]) for p in fs]==list(range(2000,20001,2000))+steps()
 allrows=[];frame=[];files=[];repeat=[]
 for step in steps():
  p=f/f'frame_{step:06d}.npy';a=np.load(p);assert a.shape==(7,512,512) and a.dtype==np.float32 and np.isfinite(a).all()
  assert a[0].min()>=.08-1e-7 and a[0].max()<=.24+1e-7 and a[1:].min()>=0 and a[1:].max()<=1
  files.append(dict(Method=method,Run=rid,Physical_step=step,path=str(p),SHA256=sha(p),dtype=str(a.dtype),shape='7x512x512'))
  g=c[c.pde_step==step];pr=sample(a,g);again=sample(np.load(p),g);assert np.array_equal(pr,again)
  # Full second extraction on the identical NPY, not a copy of first-pass widths.
  first=[extract(pr[j],z.GT_center_pixel,6) for j,z in enumerate(g.itertuples(index=False))]
  second=[extract(again[j],z.GT_center_pixel,6) for j,z in enumerate(g.itertuples(index=False))]
  assert all(x[2]==y[2] and np.array_equal(np.asarray(x[:2]),np.asarray(y[:2]),equal_nan=True) for x,y in zip(first,second))
  historical=old.sample(a,g);rows=[]
  for j,z in enumerate(g.itertuples(index=False)):
   w,pk,status=first[j];rv=refs.loc[(step,z.profile_id)];valid=np.isfinite(w)
   ow,op=old.width(historical[j],z.GT_center_pixel,6)
   rows.append(dict(Method=method,Run=rid,Physical_step=step,T_star=step*.0005,Candidate_ID=z.profile_id,Candidate_x=z.x_pixel,Candidate_y=z.y_pixel,Normal_x=z.normal_x,Normal_y=z.normal_y,Reference_FWHM_nm=z.width_GT_nm,Recomputed_reference_FWHM_nm=rv.Recomputed_reference_FWHM_nm,Reference_peak_px=z.GT_center_pixel,Predicted_FWHM_nm=w,Predicted_peak_px=pk,Relative_error_percent=100*abs(w-z.width_GT_nm)/z.width_GT_nm,Absolute_error_nm=abs(w-z.width_GT_nm),Reference_valid=True,Prediction_valid=valid,Pair_valid=valid,Invalid_reason='' if valid else status,Diagnostic_previous_float64_width_nm=ow,Diagnostic_previous_float64_error_percent=100*abs(ow-z.width_GT_nm)/z.width_GT_nm,Diagnostic_previous_float64_valid=np.isfinite(ow)))
  d=pd.DataFrame(rows);v=d[d.Pair_valid];ov=d[d.Diagnostic_previous_float64_valid]
  frame.append(dict(Method=method,Run=rid,Physical_step=step,T_star=step*.0005,Candidate_count=len(d),Reference_valid_count=len(d),Prediction_valid_count=len(v),Pair_valid_count=len(v),Coverage_percent=100*len(v)/len(d),FWHM_relative_error_percent=v.Relative_error_percent.mean() if len(v)>=10 else np.nan,FWHM_absolute_error_nm=v.Absolute_error_nm.mean() if len(v)>=10 else np.nan,Frame_valid=len(v)>=10,Diagnostic_previous_float64_error_percent=ov.Diagnostic_previous_float64_error_percent.mean() if len(ov)>=10 else np.nan,Diagnostic_valid_classification_changes=int((d.Pair_valid!=d.Diagnostic_previous_float64_valid).sum())))
  allrows.extend(rows);repeat.append(dict(Method=method,Run=rid,Physical_step=step,profile_count=len(d),sampled_profile_arrays_bitwise_equal=True,valid_flags_equal=True,widths_equal_including_nan=True,max_absolute_difference_nm=0.,max_relative_difference=0.,frame_error_equal=True))
  if len(frame)%20==0:print('FWHM',method,rid,len(frame),'/80',flush=True)
 csv(allrows,dest);csv(frame,ev/f'{method}_Run_{rid:02d}_frames.csv');csv(files,ev/f'{method}_Run_{rid:02d}_input_manifest.csv');csv(repeat,ev/f'{method}_Run_{rid:02d}_repeat.csv')
 stamp=dict(method=method,run=rid,extractor_SHA256=sha(OUT/'fwhm_unified.py'),evaluator_SHA256=sha(__file__),candidate_SHA256=gate['candidate_SHA256'],profiles=len(allrows),frames=len(frame),repeatability_passed=True)
 (ev/f'{method}_Run_{rid:02d}_receipt.json').write_text(json.dumps(stamp,indent=2),encoding='utf8')
def aggregate():
 ev=OUT/'per_run'
 p=pd.concat([pd.read_csv(ev/f'{m}_Run_{r:02d}_profiles.csv') for m in METHODS for r in RUNS],ignore_index=True)
 f=pd.concat([pd.read_csv(ev/f'{m}_Run_{r:02d}_frames.csv') for m in METHODS for r in RUNS],ignore_index=True)
 assert len(p)==3*6*80*256 and len(f)==1440
 csv(p,OUT/'03_三模型逐剖面FWHM评价.csv');csv(f,OUT/'04_逐帧FWHM及覆盖率.csv')
 per=f.groupby(['Method','Run']).agg(FWHM_relative_error_percent=('FWHM_relative_error_percent','mean'),FWHM_absolute_error_nm=('FWHM_absolute_error_nm','mean'),Coverage_percent=('Coverage_percent','mean'),Valid_frames=('Frame_valid','sum'),Diagnostic_previous_float64_error_percent=('Diagnostic_previous_float64_error_percent','mean'),Diagnostic_valid_classification_changes=('Diagnostic_valid_classification_changes','sum')).reset_index();per['Invalid_frames']=80-per.Valid_frames
 csv(per,OUT/'05_逐Run_FWHM结果.csv')
 for n,ids,name in [(6,RUNS,'06_六轨迹FWHM汇总.csv'),(5,RUNS[1:],'07_五条HeldOut轨迹汇总.csv')]:
  z=per[per.Run.isin(ids)];rr=[]
  for m,g in z.groupby('Method'):
   for k in ['FWHM_relative_error_percent','FWHM_absolute_error_nm','Coverage_percent','Diagnostic_previous_float64_error_percent']:
    rr.append(dict(Method=m,Metric=k,Run_set='Run00+Run31-35' if n==6 else 'Run31-35',N_runs=int(g[k].count()),Mean=g[k].mean(),Sample_SD=g[k].std(ddof=1),Aggregation='valid profiles per frame, valid frames per Run, equal Runs',Invalid_frames=int(g.Invalid_frames.sum())))
  csv(rr,OUT/name)
 # All methods' candidate keys and archived reference values must be identical.
 keys=['Run','Physical_step','Candidate_ID'];pv=p.pivot(index=keys,columns='Method',values='Pair_valid');valid=pv.all(axis=1);valid=valid[valid].reset_index()[keys]
 common=p.merge(valid,on=keys,how='inner',validate='many_to_one');cf=common.groupby(['Method','Run','Physical_step']).agg(Common_pairs=('Candidate_ID','count'),Common_FWHM_error_percent=('Relative_error_percent','mean')).reset_index()
 cf= f[['Method','Run','Physical_step','Reference_valid_count']].merge(cf,on=['Method','Run','Physical_step'],how='left').fillna({'Common_pairs':0})
 cf['Common_coverage_percent']=100*cf.Common_pairs/cf.Reference_valid_count;cf['Common_frame_valid']=cf.Common_pairs>=10;cf.loc[~cf.Common_frame_valid,'Common_FWHM_error_percent']=np.nan
 csv(cf,OUT/'common_valid_per_frame.csv')
 cr=cf.groupby(['Method','Run']).agg(Common_FWHM_error_percent=('Common_FWHM_error_percent','mean'),Common_coverage_percent=('Common_coverage_percent','mean'),Common_valid_frames=('Common_frame_valid','sum')).reset_index()
 cr=per.merge(cr,on=['Method','Run']);csv(cr,OUT/'08_三模型共同有效剖面比较.csv')
 pairs=[];cs=[]
 for n,ids in [(6,RUNS),(5,RUNS[1:])]:
  for m,g in cr[cr.Run.isin(ids)].groupby('Method'):
   cs.append(dict(Method=m,N_runs=n,Mean=g.Common_FWHM_error_percent.mean(),Sample_SD=g.Common_FWHM_error_percent.std(ddof=1),Common_coverage_mean=g.Common_coverage_percent.mean(),Common_coverage_SD=g.Common_coverage_percent.std(ddof=1)))
 for a,b in [('No-PDE','Hybrid'),('No-local','Hybrid'),('No-local','No-PDE')]:
  x=cr[cr.Method==a].set_index('Run');y=cr[cr.Method==b].set_index('Run')
  for r in RUNS:pairs.append(dict(Contrast=a+' - '+b,Run=r,Formal_difference_pp=x.loc[r,'FWHM_relative_error_percent']-y.loc[r,'FWHM_relative_error_percent'],Common_valid_difference_pp=x.loc[r,'Common_FWHM_error_percent']-y.loc[r,'Common_FWHM_error_percent']))
 csv(cs,OUT/'common_valid_summary.csv');csv(pairs,OUT/'paired_Run_differences.csv')
 repeats=pd.concat([pd.read_csv(ev/f'{m}_Run_{r:02d}_repeat.csv') for m in METHODS for r in RUNS]);csv(repeats,OUT/'repeatability_per_frame.csv')
 inputs=pd.concat([pd.read_csv(ev/f'{m}_Run_{r:02d}_input_manifest.csv') for m in METHODS for r in RUNS]);csv(inputs,OUT/'prediction_files_manifest.csv')
 print(per.to_string(index=False));print(pd.DataFrame(cs).to_string(index=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','aggregate']);ap.add_argument('--method',choices=METHODS);ap.add_argument('--run',type=int);a=ap.parse_args()
 if a.stage=='run':evaluate(a.method,a.run)
 else:aggregate()
if __name__=='__main__':main()
