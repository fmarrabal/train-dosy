"""Public Varian DOSY example; discovery-only common phase, split first."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,importlib
import numpy as np
import prepare_public as loaders
exp=loaders.exp
varian=importlib.import_module('nmrglue.fileio.varian')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent/'prop_pent_suc'
SOURCE=HERE.parent/'legacy_data/TestData/DOSY/PropPentSuc_DOSY.fid'
GNAT=Path('E:/ARTICULOS-CIENTIFICOS/NEW_ITERATIVE_ILT/ALGORITMOS/GNAT/UoM_GNAT-GNAT-2.4.1/Import_Export/varianimport.m')
GNAT_SOURCE_SHA256='301537cff360d146ade72e569693c7e01a5d415c55bcb333b92fcfadaa379f97'

def main():
    HERE.mkdir(exist_ok=True)
    dic,fid=varian.read(str(SOURCE));fid=np.squeeze(fid);pp=dic['procpar']
    def values(k):return np.array(pp[k]['values'],float)
    def scalar(k):return float(values(k)[0])
    n=int(scalar('fn'))//2
    # The Varian and GNAT readers share the +imag convention. GNAT displays
    # flipped FFT on an ascending ppm axis; this is the equivalent descending form.
    f=fid.astype(complex).copy();f[:,0]*=.5
    spec=np.fft.fftshift(np.fft.fft(f,n=n,axis=-1),axes=-1)
    t=np.arange(n)/(n-1)
    test=np.zeros(len(fid),bool);test[3:-1:4]=True;train=~test
    tr=np.flatnonzero(train);iv=np.zeros(len(tr),bool);iv[2:-1:4]=True
    discovery=train.copy();discovery[tr[iv]]=False
    phase_data=spec[discovery].mean(0);phase_data/=abs(phase_data).max()
    starts=[[scalar('rp')+scalar('lp'),-scalar('lp')],[0.,0.],[90.,0.]]
    phase_runs=[minimize(loaders.autophase._ps_acme_score,start,args=(phase_data,),
        method='Nelder-Mead',bounds=[(-360,360),(-180,180)],
        options=dict(maxfev=1000,xatol=1e-5,fatol=1e-9)) for start in starts]
    phase_best=min(phase_runs,key=lambda r:r.fun)
    if not phase_best.success:raise RuntimeError('ACME phase did not converge')
    phase=np.deg2rad(phase_best.x[0]+phase_best.x[1]*np.arange(n)/n)
    y=(spec*np.exp(1j*phase)).real
    ppm=(scalar('rfp')-scalar('rfl')+scalar('sw')*(1-t))/scalar('sfrq')
    g=values('gzlvl1')*scalar('DAC_to_G')*.01
    b=scalar('dosygamma')**2*scalar('dosytimecubed')*g*g
    assert y.shape[0]==len(b)==30
    mask,sigma,details=exp.detect(y,b,ppm,discovery)
    scale=float(np.sqrt(np.mean(y[discovery][:,mask]**2)));indices=np.flatnonzero(mask)
    starts=np.flatnonzero(np.r_[True,np.diff(indices)>1]);windows=np.column_stack((starts,np.r_[starts[1:],len(indices)]))
    np.savez_compressed(HERE/'input.npz',Y=y[:,mask]/scale,b=b,ppm=ppm[mask],indices=indices,sigma=sigma/scale,
        outer_test=test,inner_validation_rows=tr[iv],windows=windows,full_ppm=ppm,full_Y=y/scale,scale=scale)
    negative_out=int(((y[np.flatnonzero(discovery)[:4]].mean(0)-np.median(y[np.flatnonzero(discovery)[:4]].mean(0))<-8*details['low_mean_noise'])&~mask).sum())
    metadata=dict(archive='Official DOSY Toolbox datasets_v09.zip, linked by official downloads page',
        archive_sha256=exp.sha(HERE.parent/'sources/datasets_v09.zip'),source_title=(SOURCE/'text').read_text(),
        source_directory='PropPentSuc_DOSY.fid',documented_solutes_in_text=['propanol','sucrose'],
        possible_additional_solute='Directory name suggests pentanol; not assigned as certified composition without an exact source crosswalk.',
        known_component_D=False,known_global_diffusion_order=False,
        n_acquisitions=len(b),n_original_channels=n,n_selected_channels=int(mask.sum()),mask=details,
        negative_coherent_points_outside_support=negative_out,
        outer_test_positions_one_based=(np.flatnonzero(test)+1).tolist(),
        discovery_positions_one_based=(np.flatnonzero(discovery)+1).tolist(),
        inner_validation_positions_one_based=(tr[iv]+1).tolist(),
        processing=dict(reader='Unmodified nmrglue 0.11 Varian module',first_point_multiplier=.5,zero_fill=n,
            apodization='none',phase='One common ACME phase estimated only from discovery rows and frozen for all methods/rows',
            archived_rp=scalar('rp'),archived_lp=scalar('lp'),phase_degrees=phase_best.x,
            phase_runs=[dict(x=r.x,fun=r.fun,success=r.success,nfev=r.nfev) for r in phase_runs],
            baseline_correction=False,rectification=False,ppm_axis='(rfp-rfl+sw*(1-t))/sfrq',
            phase_selection='Saved metadata phase left dispersion visible in discovery-only QA. ACME fitted to normalized mean discovery complex spectrum using unmodified nmrglue 0.11 objective. No test/internal-validation input.',
            phase_objective_source_sha256=exp.sha(loaders.autophase.__file__)),
        diffusion_encoding=dict(equation='b=dosygamma^2*dosytimecubed*(gzlvl1*DAC_to_G*0.01)^2',
            gamma=scalar('dosygamma'),time_cubed=scalar('dosytimecubed'),DAC_to_G=scalar('DAC_to_G'),
            source=str(GNAT),source_sha256=GNAT_SOURCE_SHA256,source_lines='229-251',
            calibration='Nominal archived Varian scale; no independently verified D accuracy.',
            NUG='Homogeneous-gradient exponential kernel. Archive nugcal400 is retained but not imposed without an exact acquisition-calibration crosswalk; unmodelled nonuniform gradients can create curvature.'),
        sigma_normalized=sigma/scale,source_hashes={str(p.relative_to(HERE.parent)):exp.sha(p) for p in SOURCE.rglob('*') if p.is_file()},
        input_sha256=exp.sha(HERE/'input.npz'))
    exp.save_json(HERE/'preprocessing.json',metadata)
    fig,axs=plt.subplots(2,1,figsize=(10,6))
    for row in np.flatnonzero(discovery):axs[0].plot(ppm,y[row]/scale,lw=.5)
    axs[0].set_xlim(ppm.max(),ppm.min());axs[0].set_title('Discovery acquisitions only; common ACME phase')
    lo=y[discovery].mean(0)/scale;axs[1].plot(ppm,lo,lw=.6);axs[1].plot(ppm[mask],lo[mask],'.',ms=1)
    axs[1].set_xlim(ppm.max(),ppm.min());axs[1].set_xlabel('Chemical shift (ppm)')
    fig.tight_layout();fig.savefig(HERE/'preprocessing_discovery_QA.png',dpi=180);plt.close(fig)
    print(json.dumps({k:metadata[k] for k in ['n_acquisitions','n_original_channels','n_selected_channels','sigma_normalized']},indent=2))

if __name__=='__main__':main()
