"""One v2 FWHM implementation, reference gate before any model evaluation.
Uses the archived candidates/normals, never regenerates their positions.
"""
from pathlib import Path
import argparse,json,hashlib,sys,platform
import numpy as np,pandas as pd,scipy
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
ROOT=Path(r'D:\vscode work\xiangchang\LSTM\model03')
OUT=Path(__file__).resolve().parent
RUNS=[0,31,32,33,34,35]
S=np.arange(-12,12.001,.25)
CONFIG=dict(indicator='clip(2*(1-sum(eta_i**2)),0,1)',sampling_range_px=[-12,12],sampling_step_px=.25,dx_nm=2.,interpolation='scipy.ndimage.map_coordinates order=1 mode=grid-wrap',reference_peak_window_px=2.,prediction_peak_window_px=6.,peak_height=.1,prominence=.05,half_maximum='absolute peak height / 2',minimum_pairs=10,reference_width_verification_tolerance_nm=.0002,tolerance_source='evaluate_no_pde_mean.py and V1.5 evaluate_nolocal.py pre-existing 2e-4 nm',no_extra_width_thresholds=True,no_candidate_resampling=True,field_computation_dtype='float32',precision_selection='reference-only comparison before model results; float32 reproduces all 122880 archived reference peak locations, float64 differs in 1778 flat peaks',aggregation='profiles within frame, valid frames within Run, equal Runs; ddof=1',outputs_per_run=80)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def steps():return [20000+6000*i+2000*(i//40) for i in range(1,81)]
def sample(a,rows):
 a=np.asarray(a,dtype=np.float32);phi=np.clip(2*(1-np.sum(a[1:]**2,axis=0)),0,1)
 y=(rows.y_pixel.to_numpy()[:,None]+rows.normal_y.to_numpy()[:,None]*S)%512
 x=(rows.x_pixel.to_numpy()[:,None]+rows.normal_x.to_numpy()[:,None]*S)%512
 return map_coordinates(phi,[y,x],order=1,mode='grid-wrap')
def extract(profile,center,limit):
 peaks,_=find_peaks(profile,height=CONFIG['peak_height'],prominence=CONFIG['prominence'])
 peaks=[k for k in peaks if abs(S[k]-center)<=limit]
 if not peaks:return np.nan,np.nan,'missing_qualified_peak'
 # Same nearest-peak rule and stable peak ordering as the documented evaluator.
 k=min(peaks,key=lambda p:abs(S[p]-center));h=profile[k]/2;l=k;r=k
 while l>0 and profile[l]>=h:l-=1
 while r<len(S)-1 and profile[r]>=h:r+=1
 if (l==0 and profile[l]>=h) or (r==len(S)-1 and profile[r]>=h):return np.nan,S[k],'missing_half_maximum_crossing'
 left=S[l]+(h-profile[l])/(profile[l+1]-profile[l])*.25
 right=S[r-1]+(h-profile[r-1])/(profile[r]-profile[r-1])*.25
 w=(right-left)*2
 if not np.isfinite(w) or w<=0:return np.nan,S[k],'nonfinite_or_nonpositive_width'
 return w,S[k],'valid'
def candidates():
 p=ROOT/'图/图图/图6/re/profiles/all_profile_candidates.csv'
 d=pd.read_csv(p);d=d[d.pde_step.isin(steps())].copy()
 assert len(d)==6*80*256 and not d.duplicated(['run_id','pde_step','profile_id']).any()
 assert (d.GT_status=='valid').all()
 return d,p
def reference():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 (OUT/'FWHM_config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf8')
 c,cp=candidates();rows=[];files=[]
 for (run,step),g in c.groupby(['run_id','pde_step'],sort=True):
  path=ROOT/'run'/run/f'frame_{step:06d}.npy';a=np.load(path);assert a.shape==(7,512,512) and np.isfinite(a).all()
  z=sample(a,g);files.append(dict(Run=run,Physical_step=int(step),path=str(path),SHA256=sha(path),dtype=str(a.dtype),shape=list(a.shape)))
  for j,v in enumerate(g.itertuples(index=False)):
   w,center,status=extract(z[j],0,2)
   rows.append(dict(Run=run,Physical_step=step,Candidate_ID=v.profile_id,Candidate_x=v.x_pixel,Candidate_y=v.y_pixel,Normal_x=v.normal_x,Normal_y=v.normal_y,Archived_reference_FWHM_nm=v.width_GT_nm,Recomputed_reference_FWHM_nm=w,Archived_reference_peak_px=v.GT_center_pixel,Recomputed_reference_peak_px=center,Reference_status=status,Width_abs_difference_nm=abs(w-v.width_GT_nm),Peak_abs_difference_px=abs(center-v.GT_center_pixel),Width_bitwise_equal=(np.float64(w).tobytes()==np.float64(v.width_GT_nm).tobytes())))
  if step==504000:print('reference',run,'80/80',flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'reference_profile_verification.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(files).to_csv(OUT/'reference_files_manifest.csv',index=False,encoding='utf-8-sig')
 passed=bool((d.Reference_status=='valid').all() and (d.Width_abs_difference_nm<=CONFIG['reference_width_verification_tolerance_nm']).all() and (d.Peak_abs_difference_px==0).all())
 report=dict(passed=passed,candidate_count=len(d),reference_valid_count=int((d.Reference_status=='valid').sum()),width_bitwise_equal_count=int(d.Width_bitwise_equal.sum()),width_max_abs_difference_nm=float(d.Width_abs_difference_nm.max()),width_mean_abs_difference_nm=float(d.Width_abs_difference_nm.mean()),width_outside_preexisting_tolerance_count=int((d.Width_abs_difference_nm>.0002).sum()),peak_max_abs_difference_px=float(d.Peak_abs_difference_px.max()),peak_mismatch_count=int((d.Peak_abs_difference_px!=0).sum()),candidate_source=str(cp),candidate_SHA256=sha(cp),evaluation_SHA256=sha(__file__),environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__))
 (OUT/'reference_gate.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
 if not passed:raise RuntimeError('REFERENCE GATE FAILED: no cross-model aggregation is permitted')
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference']);args=p.parse_args();reference()
if __name__=='__main__':main()
