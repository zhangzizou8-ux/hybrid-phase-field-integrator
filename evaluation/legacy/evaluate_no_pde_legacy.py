"""Three-way evaluation using archived Fig9 frame statistics and archived FWHM candidates.
Default evaluates completed No-PDE runs and archived Hybrid/No-local statistics.
--regenerate-no-local reconstructs missing No-local profiles using verified F9a.pth,
evaluates them and deletes only these newly generated temporary trajectories.
"""
import sys,os,json,argparse,subprocess,shutil,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if HERE.name=='取消PDE修正' else HERE.parents[0]
RUNS=[0,31,32,33,34,35]
def phi(a):return np.clip(2*(1-(a[1:]**2).sum(0)),0,1)
def morphology(a):
    mask=a[0]>.18;n=ndi.label(mask)[1];f=mask.mean();R=2*np.sqrt(mask.sum()/n/np.pi) if n else 0.
    gb=phi(a)>.05;gy,gx=np.gradient(gb.astype(float));L=np.hypot(gx,gy).sum()
    return f,n,R,L,int((mask&gb).sum())
def metrics(a,b):
    a=a.astype(float);b=b.astype(float);p,q=phi(a),phi(b);f,n,r,l,o=morphology(a);F,N,R,L,O=morphology(b)
    d=dict(nRMSE=100*np.linalg.norm(a-b)/np.linalg.norm(b),RMSE_c=np.sqrt(np.mean((a[0]-b[0])**2)),RMSE_eta=np.sqrt(np.mean((a[1:]-b[1:])**2)),RMSE_phi=np.sqrt(np.mean((p-q)**2)),SSIM_c=structural_similarity(a[0],b[0],data_range=.16,win_size=7,gaussian_weights=False,use_sample_covariance=True),mean_error=abs(a[0].mean()-b[0].mean()),f_A=f,N=n,R=r,R3=r**3,GT_R3=R**3,L=l,overlap_excess=max(o-O,0),overlap=float(max(o-O,0)>=25))
    for key,x,y in [('f_A_error',f,F),('N_error',n,N),('R_error',r,R),('L_error',l,L)]:d[key]=100*abs(x-y)/abs(y) if y else np.nan
    return d
S=np.arange(-12,12.001,.25)
def width(profile,center,limit):
    peaks,_=find_peaks(profile,height=.1,prominence=.05)
    peaks=[p for p in peaks if abs(S[p]-center)<=limit]
    if not peaks:return np.nan,np.nan
    k=min(peaks,key=lambda p:abs(S[p]-center))
    return width_at(profile,k),S[k]
def width_at(profile,k):
    h=profile[k]/2;j=k
    while j>0 and profile[j]>=h:j-=1
    if j==0 and profile[j]>=h:return np.nan
    left=S[j]+(h-profile[j])/(profile[j+1]-profile[j])*.25;j=k
    while j<len(S)-1 and profile[j]>=h:j+=1
    if j==len(S)-1 and profile[j]>=h:return np.nan
    right=S[j-1]+(h-profile[j-1])/(profile[j]-profile[j-1])*.25
    return (right-left)*2
def sample(a,rows):
    ys=(rows.y_pixel.to_numpy()[:,None]+rows.normal_y.to_numpy()[:,None]*S)%512
    xs=(rows.x_pixel.to_numpy()[:,None]+rows.normal_x.to_numpy()[:,None]*S)%512
    return ndi.map_coordinates(phi(a.astype(float)),[ys,xs],order=1,mode='grid-wrap')
def fwhm(a,b,rows):
    # Audit reconstructed extraction against every archived reference width/center first.
    gt=sample(b,rows);pred=sample(a,rows);details=[]
    for j,(_,row) in enumerate(rows.iterrows()):
        cg=row.GT_center_pixel;wg=width_at(gt[j],int(np.argmin(abs(S-cg))))
        if not np.isfinite(wg) or abs(wg-row.width_GT_nm)>2e-4 or abs(cg-row.GT_center_pixel)>1e-8:
            raise RuntimeError(f'FWHM reference reproduction failed: {row.run_id} {row.pde_step} {row.profile_id} extracted={wg}, archived={row.width_GT_nm}. Do not aggregate widths.')
        wp,cp=width(pred[j],cg,6)
        details.append(dict(profile_id=int(row.profile_id),width_GT_nm=row.width_GT_nm,width_prediction_nm=wp,valid=np.isfinite(wp),abs_error_nm=abs(wp-row.width_GT_nm),relative_error_percent=100*abs(wp-row.width_GT_nm)/row.width_GT_nm))
    d=pd.DataFrame(details);v=d[d.valid]
    return dict(FWHM_error=v.relative_error_percent.mean() if len(v)>=10 else np.nan,FWHM_abs_nm=v.abs_error_nm.mean() if len(v)>=10 else np.nan,coverage=100*len(v)/len(d),valid_pairs=len(v)),d
def evaluate_run(folder,rid,method,candidates,ev):
    name=f'Run_{rid:02d}';rows=[];details=[]
    fs=sorted(folder.glob('frame_*.npy'));assert len(fs)==90
    for j,p in enumerate(fs[10:]):
        step=int(p.stem[6:]);a=np.load(p);b=np.load(ROOT/'run'/name/p.name)
        d=metrics(a,b);w,dt=fwhm(a,b,candidates[(candidates.run_id==name)&(candidates.pde_step==step)])
        assert len(dt)>0;d.update(w);d.update(method=method,run_id=rid,pde_step=step,T_star=step*.0005);rows.append(d)
        dt['method']=method;dt['run_id']=rid;dt['pde_step']=step;details.append(dt)
        if j%20==0:print('evaluation',method,name,j+1,'/80',flush=True)
    df=pd.DataFrame(rows);df.to_csv(ev/f'{method}_{name}_frames.csv',index=False)
    pd.concat(details).to_csv(ev/f'{method}_{name}_fwhm_pairs.csv',index=False)
    return df
def archived(method,rid):
    prefix='Hybrid_P40' if method=='Hybrid' else 'No_local_P40'
    d=pd.read_csv(ROOT/'图'/'图图'/'图9'/'re'/'data'/f'{prefix}_Run_{rid:02d}_frames.csv');d=d[d.scope=='rollout'].copy() if 'rollout' in d.scope.unique() else d[d.rollout_index>0].copy()
    mapping={'X_T_star':'T_star','nRMSE_7field_percent':'nRMSE','N_abs_relative_percent':'N_error','Vf_abs_relative_percent':'f_A_error','R_nm_abs_relative_percent':'R_error','Lgb_nm_abs_relative_percent':'L_error','GB_excess_ge25':'overlap','GB_excess_pixels':'overlap_excess','Vf_Pred':'f_A','N_Pred':'N','R_nm_Pred':'R','c_mean_delta':'mean_error'}
    d=d.rename(columns=mapping);d['mean_error']=d.mean_error.abs();d['run_id']=rid;d['method']=method;d['R3']=d.R**3;d['GT_R3']=d.R_nm_GT**3
    if method=='Hybrid':
        x=pd.read_csv(ROOT/'投稿改'/'基线方法'/'对比_六条'/'02_prediction_errors.csv');x=x[(x.method=='Hybrid')&(x.run_id==rid)&(x.scope=='autoregressive')]
        if x.empty:
            x=pd.read_csv(ROOT/'投稿改'/'基线方法'/'对比_六条'/'02_prediction_errors.csv');x=x[(x.method=='Hybrid')&(x.run_id==rid)&(x.pde_step>20000)]
        for key,col in [('FWHM_error','GB_FWHM_MAPE_percent'),('FWHM_abs_nm','GB_FWHM_MAE_nm'),('coverage','width_pair_coverage_percent'),('SSIM_c','SSIM_c'),('RMSE_phi','RMSE_phi')]:d[key]=d.pde_step.map(x.set_index('pde_step')[col])
    return d
def summaries(allf,ev):
    metrics_cols=['nRMSE','FWHM_error','FWHM_abs_nm','coverage','N_error','mean_error','RMSE_c','RMSE_eta','RMSE_phi','SSIM_c','f_A_error','R_error','L_error']
    runs=[];integrated=[]
    for (method,rid),d in allf.groupby(['method','run_id']):
        d=d.sort_values('T_star');assert len(d)==80
        z=dict(method=method,run_id=int(rid),n_outputs=80)
        for k in metrics_cols:z[k]=d[k].mean() if k in d else np.nan
        win=d[d.T_star<=250];assert len(win)==79
        z.update(overlap=100*win.overlap.mean(),overlap_frames=int(win.overlap.sum()),overlap_denominator=79,terminal_nRMSE=d.iloc[-1].nRMSE)
        slope=np.polyfit(d.T_star,d.R3,1)[0];gt=np.polyfit(d.T_star,d.GT_R3,1)[0];z['R3_slope']=slope;z['R3_slope_error']=100*abs(slope-gt)/abs(gt);runs.append(z)
        for k in ['nRMSE','f_A_error','N_error','R_error','L_error']:
            t=d.T_star.to_numpy();y=d[k].to_numpy();mask=t<250;x=np.r_[t[mask],250];v=np.r_[y[mask],np.interp(250,t,y)]
            integrated.append(dict(method=method,run_id=int(rid),metric=k,value=np.trapezoid(v,x)/237))
    per=pd.DataFrame(runs);per.to_csv(ev/'per_run_metrics.csv',index=False);integ=pd.DataFrame(integrated);integ.to_csv(ev/'fig9d_integrated_metrics.csv',index=False)
    for n,ids,file in [(6,RUNS,'summary_6runs.csv'),(5,RUNS[1:],'summary_heldout_5runs.csv')]:
        sub=per[per.run_id.isin(ids)];summary=sub.groupby('method').agg({k:['mean','std','count'] for k in metrics_cols+['overlap','R3_slope_error','terminal_nRMSE']});summary.columns=['_'.join(x) for x in summary.columns];summary.to_csv(ev/file)
        ss=integ[integ.run_id.isin(ids)].groupby(['method','metric']).value.agg(['mean','std','count']);ss.to_csv(ev/f'fig9d_summary_{n}runs.csv')
    pairs=[]
    for a,b in [('No-PDE','Hybrid'),('No-local','No-PDE'),('No-local','Hybrid')]:
        for rid in RUNS:
            x=per[(per.method==a)&(per.run_id==rid)].iloc[0];y=per[(per.method==b)&(per.run_id==rid)].iloc[0]
            for k in ['nRMSE','FWHM_error','overlap','N_error','mean_error']:pairs.append(dict(contrast=f'{a} - {b}',run_id=rid,metric=k,difference=x[k]-y[k]))
    pd.DataFrame(pairs).to_csv(ev/'paired_differences.csv',index=False);per.to_csv(ev/'three_way_comparison.csv',index=False)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--regenerate-no-local',action='store_true');args=ap.parse_args()
    out=HERE/'results_no_pde_mean';ev=out/'evaluation';ev.mkdir(parents=True,exist_ok=True)
    if args.regenerate_no_local:
        weight=ROOT/'图'/'图图'/'图9'/'F9a.pth'
        assert hashlib.sha256(weight.read_bytes()).hexdigest()=='fc30c893680e7d564888a891c368ff39d587fff86e9e63fb7dcc8926c04c76e0'
        os.environ['NVIDIA_TF32_OVERRIDE']='0'
    candidates=pd.read_csv(ROOT/'图'/'图图'/'图6'/'re'/'profiles'/'all_profile_candidates.csv')
    allf=[];status={}
    for rid in RUNS:
        m=json.loads((ev/'inference_manifest.json').read_text(encoding='utf8'));assert m['runs'][f'Run_{rid:02d}']['success']
        file=ev/f'No-PDE_Run_{rid:02d}_frames.csv'
        allf.append(pd.read_csv(file) if file.exists() else evaluate_run(out/f'Run_{rid:02d}',rid,'No-PDE',candidates,ev))
        allf.append(archived('Hybrid',rid));nl=archived('No-local',rid)
        if args.regenerate_no_local:
            temp=out/'temporary_no_local'/f'Run_{rid:02d}';file=ev/f'No-local_Run_{rid:02d}_frames.csv'
            if file.exists():fresh=pd.read_csv(file)
            else:
                frames=[str(ROOT/'run'/f'Run_{rid:02d}'/f'frame_{s:06d}.npy') for s in range(2000,20001,2000)]
                subprocess.run([sys.executable,str(ROOT/'图'/'图图'/'图9'/'F9a_run.py'),'--model',str(ROOT/'图'/'图图'/'图9'/'F9a.pth'),'--run-id',str(rid),'--output',str(temp),'--pde-correct','40','--frames',*frames],check=True)
                fresh=evaluate_run(temp,rid,'No-local',candidates,ev)
            delta=float(np.max(abs(fresh.nRMSE.to_numpy()-nl.nRMSE.to_numpy())));status[str(rid)]={'No_local_regenerated_vs_archive_max_nRMSE_delta_pp':delta}
            # Keep archived original statistics for all previous metrics; add the newly measured FWHM only, flagged if reproduction differs.
            status[str(rid)]['archive_reproduction_within_1e_5_pp']=delta<=1e-5
            # All No-local metrics come from the same regenerated trajectory; never mix runs.
            nl=fresh.copy()
            for k in ['FWHM_error','FWHM_abs_nm','coverage']:nl[k]=nl.pde_step.map(fresh.set_index('pde_step')[k])
            if temp.exists():
                assert temp.resolve().parent==(out/'temporary_no_local').resolve();shutil.rmtree(temp)
        allf.append(nl)
    full=pd.concat(allf,ignore_index=True);full.to_csv(ev/'per_frame_metrics.csv',index=False);summaries(full,ev)
    (ev/'evaluation_audit.json').write_text(json.dumps(status,indent=2),encoding='utf8')
if __name__=='__main__':main()
