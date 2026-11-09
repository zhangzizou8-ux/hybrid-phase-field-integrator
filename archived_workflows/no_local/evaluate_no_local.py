"""All official No-local metrics are read from this bundle's predictions/ NP files.
The historical No-local tables are consulted only after evaluation, for audit.
"""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;OUT=HERE/'evaluation';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'inference_config.json').read_text(encoding='utf8'));REF=Path(CFG['reference_root']);ROOT=REF.parent;RUNS=CFG['runs']
S=np.arange(-12,12.001,.25)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(d,n):pd.DataFrame(d).to_csv(OUT/n,index=False,encoding='utf-8-sig')
def phi(a):return np.clip(2*(1-np.sum(a[1:]**2,axis=0)),0,1)
def morphology(a):
 mask=a[0]>.18;N=ndi.label(mask)[1];f=mask.mean();R=2*np.sqrt(mask.sum()/N/np.pi) if N else 0.
 gb=phi(a)>.05;gy,gx=np.gradient(gb.astype(float));L=np.hypot(gx,gy).sum()
 bg,nb=ndi.label(~mask);edge=np.unique(np.r_[bg[0],bg[-1],bg[:,0],bg[:,-1]]);holes=nb-len(edge[edge>0]);chi=N-holes if N else 0
 return dict(f_A=f,N=N,R=R,R3=R**3,L=L,chi=chi,overlap_pixels=int((mask&gb).sum()))
def metrics(a,b):
 a=a.astype(float);b=b.astype(float);ma,mb=morphology(a),morphology(b);ex=max(ma['overlap_pixels']-mb['overlap_pixels'],0)
 d=dict(nRMSE=100*np.sqrt(np.sum((a-b)**2)/np.sum(b*b)),RMSE_7=np.sqrt(np.mean((a-b)**2)),RMSE_c=np.sqrt(np.mean((a[0]-b[0])**2)),RMSE_eta=np.sqrt(np.mean((a[1:]-b[1:])**2)),RMSE_phi=np.sqrt(np.mean((phi(a)-phi(b))**2)),SSIM_c=structural_similarity(a[0],b[0],data_range=.16,win_size=7,gaussian_weights=False,use_sample_covariance=True),mean_error=abs(a[0].mean()-b[0].mean()),mean_signed_error=a[0].mean()-b[0].mean(),overlap_excess=ex,overlap=int(ex>=25),**ma)
 for k in ['f_A','N','R','L']:
  d['GT_'+k]=mb[k];d[k+'_error']=100*abs(ma[k]-mb[k])/abs(mb[k]) if mb[k] else np.nan
 d['GT_R3']=mb['R3'];d['GT_overlap_pixels']=mb['overlap_pixels'];return d
def width_at(profile,k):
 h=profile[k]/2;j=k
 while j>0 and profile[j]>=h:j-=1
 if j==0 and profile[j]>=h:return np.nan
 left=S[j]+(h-profile[j])/(profile[j+1]-profile[j])*.25;j=k
 while j<len(S)-1 and profile[j]>=h:j+=1
 if j==len(S)-1 and profile[j]>=h:return np.nan
 right=S[j-1]+(h-profile[j-1])/(profile[j]-profile[j-1])*.25
 return (right-left)*2
def pred_width(profile,center):
 peaks,_=find_peaks(profile,height=.1,prominence=.05);peaks=[p for p in peaks if abs(S[p]-center)<=6]
 if not peaks:return np.nan,np.nan,'missing_peak'
 k=min(peaks,key=lambda p:abs(S[p]-center));w=width_at(profile,k)
 return w,S[k],'valid' if np.isfinite(w) else 'missing_half_crossing'
def sample(a,d):
 ys=(d.y_pixel.to_numpy()[:,None]+d.normal_y.to_numpy()[:,None]*S)%512
 xs=(d.x_pixel.to_numpy()[:,None]+d.normal_x.to_numpy()[:,None]*S)%512
 return ndi.map_coordinates(phi(a.astype(float)),[ys,xs],order=1,mode='grid-wrap')
def fwhm(a,b,c):
 gt=sample(b,c);pr=sample(a,c);rows=[];max_ref_delta=0
 for j,z in enumerate(c.itertuples(index=False)):
  wg=width_at(gt[j],int(np.argmin(abs(S-z.GT_center_pixel))));delta=abs(wg-z.width_GT_nm);max_ref_delta=max(max_ref_delta,delta)
  if not np.isfinite(wg) or delta>2e-4:raise RuntimeError(f'Archived reference width mismatch {z.run_id}/{z.pde_step}/{z.profile_id}: {delta}')
  w,peak,status=pred_width(pr[j],z.GT_center_pixel)
  rows.append(dict(profile_id=z.profile_id,y_pixel=z.y_pixel,x_pixel=z.x_pixel,normal_y=z.normal_y,normal_x=z.normal_x,reference_center_px=z.GT_center_pixel,reference_width_nm=z.width_GT_nm,reference_width_recomputed_nm=wg,reference_width_abs_difference_nm=delta,prediction_peak_px=peak,prediction_width_nm=w,status=status,valid=np.isfinite(w),abs_error_nm=abs(w-z.width_GT_nm),relative_error_percent=100*abs(w-z.width_GT_nm)/abs(z.width_GT_nm)))
 d=pd.DataFrame(rows);v=d[d.valid];npair=len(v);nr=len(c)
 return dict(FWHM_error=v.relative_error_percent.mean() if npair>=10 else np.nan,FWHM_abs_nm=v.abs_error_nm.mean() if npair>=10 else np.nan,coverage=100*npair/nr if nr else np.nan,n_ref=nr,n_pair=npair,reference_width_max_delta_nm=max_ref_delta),d
def integrated(d,k):
 t=d.T_star.to_numpy();y=d[k].to_numpy();mask=(t>=13)&(t<250)
 return np.trapezoid(np.r_[y[mask],np.interp(250,t,y)],np.r_[t[mask],250])/237
CORE=['nRMSE','FWHM_error','FWHM_abs_nm','coverage','N_error','mean_error','RMSE_7','RMSE_c','RMSE_eta','RMSE_phi','SSIM_c','f_A_error','R_error','L_error']
INT=['nRMSE','f_A_error','N_error','R_error','L_error']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=int,default=RUNS);ap.add_argument('--final',action='store_true');a=ap.parse_args()
 candidates=pd.read_csv(HERE/'data_references/all_profile_candidates.csv');frames=[];profiles=[];manifest=[]
 for rid in a.runs:
  info=json.loads((HERE/'logs'/f'Run_{rid:02d}_manifest.json').read_text(encoding='utf8'));assert info['success']
  one=OUT/f'NoLocal_Run_{rid:02d}_frames.csv';pairs=OUT/f'NoLocal_Run_{rid:02d}_fwhm_pairs.csv'
  # Existing per-Run outputs are reused only with a matching evaluator hash.
  stamp=OUT/f'Run_{rid:02d}_evaluation_source.json'
  expected={'evaluator_SHA256':sha(__file__),'inference_manifest_SHA256':sha(HERE/'logs'/f'Run_{rid:02d}_manifest.json'),'candidates_SHA256':sha(HERE/'data_references/all_profile_candidates.csv')}
  if one.exists() and pairs.exists() and stamp.exists() and json.loads(stamp.read_text())==expected:
   frames.append(pd.read_csv(one));profiles.append(pd.read_csv(pairs));print('audited metric cache',rid,flush=True);continue
  rr=[];pp=[]
  for f in info['files'][10:]:
   p=Path(f['path']);step=f['pde_step'];assert sha(p)==f['SHA256'];ref=REF/f'Run_{rid:02d}'/p.name
   pred=np.load(p);gt=np.load(ref);d=metrics(pred,gt)
   c=candidates[(candidates.run_id==f'Run_{rid:02d}')&(candidates.pde_step==step)&(candidates.GT_status=='valid')]
   fw,dt=fwhm(pred,gt,c);d.update(fw,run_id=rid,pde_step=step,T_star=step*.0005,prediction_file=str(p),prediction_SHA256=f['SHA256'],reference_file=str(ref),reference_SHA256=sha(ref));rr.append(d)
   dt['run_id']=rid;dt['pde_step']=step;dt['T_star']=step*.0005;pp.append(dt)
   if len(rr)%20==0:print('evaluated',rid,len(rr),'/80',flush=True)
  d=pd.DataFrame(rr);pd.concat(pp).to_csv(pairs,index=False,encoding='utf-8-sig');d.to_csv(one,index=False,encoding='utf-8-sig');stamp.write_text(json.dumps(expected),encoding='utf8');frames.append(d);profiles.append(pd.concat(pp))
 if not a.final:return
 assert set(a.runs)==set(RUNS)
 full=pd.concat(frames,ignore_index=True).sort_values(['run_id','pde_step']);prof=pd.concat(profiles,ignore_index=True);csv(full,'NoLocal_per_frame_metrics.csv');csv(prof,'NoLocal_FWHM_profile_pairs.csv')
 csv(full[['run_id','pde_step','T_star','n_ref','n_pair','coverage','FWHM_error','FWHM_abs_nm','reference_width_max_delta_nm']],'NoLocal_FWHM_coverage.csv')
 window=full[(full.T_star>=13)&(full.T_star<=250)]
 csv(window[['run_id','pde_step','T_star','GT_overlap_pixels','overlap_pixels','overlap_excess','overlap']],'NoLocal_GB_overlap_79frames.csv')
 per=[];ints=[];gb=[]
 for rid,d in full.groupby('run_id'):
  assert len(d)==80 and len(d[d.T_star<=250])==79
  row=dict(run_id=int(rid),n_outputs=80,**{k:d[k].mean() for k in CORE})
  row['R3_slope']=np.polyfit(d.T_star,d.R3,1)[0];row['GT_R3_slope']=np.polyfit(d.T_star,d.GT_R3,1)[0];row['R3_slope_error']=100*abs(row['R3_slope']-row['GT_R3_slope'])/abs(row['GT_R3_slope']);per.append(row)
  ints.append(dict(run_id=int(rid),window_start=13,window_end=250,duration=237,**{k:integrated(d,k) for k in INT}))
  w=d[d.T_star<=250];gb.append(dict(run_id=int(rid),flagged_frames=int(w.overlap.sum()),native_frames=79,rate=100*w.overlap.mean()))
 per=pd.DataFrame(per);ints=pd.DataFrame(ints);gb=pd.DataFrame(gb);csv(per,'NoLocal_per_run_80outputs.csv');csv(ints,'NoLocal_per_run_time_integrated.csv');csv(gb,'NoLocal_GB_overlap_per_run.csv')
 sums=[]
 for ids,n,name in [(RUNS,6,'NoLocal_summary_6runs.csv'),(RUNS[1:],5,'NoLocal_summary_heldout_5runs.csv')]:
  rows=[]
  for agg,table,cols in [('80_outputs',per,CORE+['R3_slope_error']),('time_integrated',ints,INT),('native_79_frames',gb,['rate'])]:
   d=table[table.run_id.isin(ids)]
   for k in cols:
    metric='GB_overlap' if k=='rate' else k;rows.append(dict(method='No-local',metric=metric,aggregation=agg,run_set='Run00+Run31-35' if n==6 else 'Run31-35',n=n,mean=d[k].mean(),sample_SD=d[k].std(ddof=1),valid_n=int(d[k].notna().sum()),unit='mole fraction' if k in ['mean_error','RMSE_c'] else 'dimensionless' if k in ['RMSE_7','RMSE_eta','RMSE_phi','SSIM_c'] else 'nm' if k=='FWHM_abs_nm' else '%'))
  csv(rows,name);sums.extend(rows)
 summary=pd.DataFrame(sums);csv(summary[(summary.n==6)&(summary.aggregation=='time_integrated')],'NoLocal_Figure9d_integrated.csv')
 curve=full.groupby(['pde_step','T_star']).RMSE_7.agg(['mean','std','count']).reset_index();curve=curve.rename(columns={'mean':'RMSE_7field_mean','std':'RMSE_7field_SD','count':'n_runs','T_star':'X_T_star'});curve['RMSE_7field_lower']=curve.RMSE_7field_mean-curve.RMSE_7field_SD;curve['RMSE_7field_upper']=curve.RMSE_7field_mean+curve.RMSE_7field_SD
 prior_curve=pd.read_csv(ROOT/'图/图图/图9/re/data/Fig9a_No_local_P40_XY.csv');curve['show_errorbar']=curve.X_T_star.map(prior_curve.set_index('X_T_star').show_errorbar).fillna(0).astype(int);csv(curve,'NoLocal_Figure9a_RMSE.csv')
 # Comparison with old data is audit-only, never a source of new final metrics.
 old=pd.read_csv(ROOT/'投稿改/取消PDE修正/results_no_pde_mean/evaluation/per_frame_metrics.csv');old=old[old.method=='No-local'];merged=full.merge(old,on=['run_id','pde_step'],suffixes=('_new','_previous'),validate='one_to_one');comp=merged[['run_id','pde_step','T_star_new']].copy()
 for k in CORE+['overlap','overlap_excess']:
  if k+'_previous' in merged:comp[k+'_new']=merged[k+'_new'];comp[k+'_previous']=merged[k+'_previous'];comp[k+'_difference']=merged[k+'_new']-merged[k+'_previous']
 csv(comp,'previous_evaluation_comparison.csv')
 rep=[]
 for rid in CFG['repeat_runs']:
  m=json.loads((HERE/'logs'/f'repeat_Run_{rid:02d}_manifest.json').read_text(encoding='utf8'));assert m['success']
  formal=json.loads((HERE/'logs'/f'Run_{rid:02d}_manifest.json').read_text(encoding='utf8'))
  for f,g in zip(formal['files'],m['files']):
   assert f['pde_step']==g['pde_step'];aa=np.load(f['path']);bb=np.load(g['path']);step=f['pde_step'];row=dict(run_id=rid,pde_step=step,SHA_equal=f['SHA256']==g['SHA256'],max_absolute_field_difference=float(np.max(abs(aa-bb))),seven_field_relative_difference_percent=100*np.linalg.norm(aa.astype(float)-bb.astype(float))/np.linalg.norm(aa.astype(float)))
   if step>20000:
    gt=np.load(REF/f'Run_{rid:02d}'/Path(f['path']).name);u=metrics(aa,gt);v=metrics(bb,gt)
    for k in ['nRMSE','RMSE_7','N','f_A','R','L','mean_error','overlap_excess','overlap']:row[k+'_difference']=u[k]-v[k]
   rep.append(row)
 csv(rep,'repeatability_per_frame.csv')
 repeat=pd.DataFrame(rep)
 if not repeat.SHA_equal.all():
  raise RuntimeError('Repeatability not bitwise identical: review quantitative repeatability before freezing; no paper replacement performed')
 manifest=[]
 for rid in RUNS:
  m=json.loads((HERE/'logs'/f'Run_{rid:02d}_manifest.json').read_text(encoding='utf8'))
  for f in m['files']:manifest.append(dict(run_id=rid,**f))
 csv(manifest,'NoLocal_inference_manifest.csv')
 (OUT/'evaluation_receipt.json').write_text(json.dumps(dict(evaluator_SHA256=sha(__file__),formal_runs=RUNS,formal_NPY_count=len(manifest),n_predictions=len(full),FWHM_profile_rows=len(prof),FWHM_reference_width_max_difference_nm=float(full.reference_width_max_delta_nm.max()),repeatability_runs=CFG['repeat_runs'],repeatability_all_SHA_equal=bool(repeat.SHA_equal.all()),candidate_SHA256=sha(HERE/'data_references/all_profile_candidates.csv'),inputs_preserved=True),ensure_ascii=False,indent=2),encoding='utf8')
 print('EVALUATION PASS',summary[summary.metric.isin(['nRMSE','FWHM_error','N_error','GB_overlap'])].to_string(index=False),flush=True)
if __name__=='__main__':main()
