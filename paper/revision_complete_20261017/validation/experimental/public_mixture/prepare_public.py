from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,types,importlib
import numpy as np
HERE=Path(__file__).resolve().parent
# Load only the unmodified Bruker reader and processing module: nmrglue 0.11's
# unrelated Tecmag importer uses a NumPy dtype alias removed by NumPy 2.5.
for name,folder in [('nmrglue','nmrglue'),('nmrglue.fileio','nmrglue/fileio'),('nmrglue.process','nmrglue/process')]:
    package=types.ModuleType(name);package.__path__=[str(HERE/'vendor_runtime'/folder)];sys.modules[name]=package
ng=types.SimpleNamespace(bruker=importlib.import_module('nmrglue.fileio.bruker'))
sys.path.insert(0,str(HERE.parent));import run_experimental as exp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
autophase=importlib.import_module('nmrglue.process.proc_autophase')

SOURCE=HERE/'data/Diffusion/1'
GNAT=Path('E:/ARTICULOS-CIENTIFICOS/NEW_ITERATIVE_ILT/ALGORITMOS/GNAT/UoM_GNAT-GNAT-2.4.1/Import_Export/brukerimport.m')

def main():
    dic,fid=ng.bruker.read(str(SOURCE),read_pulseprogram=False)
    proc=ng.bruker.read_jcamp(str(SOURCE/'pdata/1/procs'))
    acq=dic['acqus'];n=int(proc['SI']);swh=float(acq['SW_h']);sf=float(proc['SF'])
    # GNAT importer uses complex(real, -imag), whereas nmrglue reads +imag.
    # Conjugation after digital-filter removal retains the real time shift.
    corrected=np.conj(ng.bruker.remove_digital_filter(dic,fid,truncate=False))
    corrected[:,0]*=float(proc['FCOR'])
    spectrum=np.fft.fftshift(np.fft.fft(corrected,n=n,axis=-1),axes=-1)
    # Freeze split before estimating one common linear phase. Held-out spectra
    # and internal-validation spectra are never used to tune preprocessing.
    test=np.zeros(len(fid),bool);test[[2,5]]=True;train=~test
    tr=np.flatnonzero(train);iv=np.zeros(len(tr),bool);iv[2:-1:4]=True
    discovery=train.copy();discovery[tr[iv]]=False
    phase_data=spectrum[discovery].mean(0);phase_data/=np.max(abs(phase_data))
    phase_runs=[minimize(autophase._ps_acme_score,[start,0.],args=(phase_data,),
        method='Powell',bounds=[(-360,360),(-180,180)],
        options=dict(maxiter=300,xtol=1e-7,ftol=1e-9)) for start in [-180.,-90.,0.,90.,180.]]
    phase_best=min(phase_runs,key=lambda result:result.fun)
    if not phase_best.success:raise RuntimeError('Common discovery-only phase did not converge')
    phase=np.deg2rad(phase_best.x[0]+phase_best.x[1]*np.arange(n)/n)
    y=(spectrum*np.exp(1j*phase)).real
    ppm=float(acq['O1'])/sf+swh/(2*sf)-np.arange(n)*swh/(sf*n)
    gradients=np.loadtxt(SOURCE/'difflist')*.01
    gamma=267524618.573;delta=2*acq['P'][30]*1e-6;Delta=acq['D'][20]
    tau=acq['D'][16]+acq['P'][2]*1e-6;alpha=acq['CNST'][14]
    time3=delta**2*(Delta-(5-3*alpha**2)*delta/16-(1-alpha**2)*tau/2)
    b=gamma**2*time3*gradients**2
    mask,sigma,details=exp.detect(y,b,ppm,discovery)
    scale=float(np.sqrt(np.mean(y[discovery][:,mask]**2)));indices=np.flatnonzero(mask)
    starts=np.flatnonzero(np.r_[True,np.diff(indices)>1]);windows=np.column_stack((starts,np.r_[starts[1:],len(indices)]))
    np.savez_compressed(HERE/'input.npz',Y=y[:,mask]/scale,b=b,ppm=ppm[mask],indices=indices,sigma=sigma/scale,
        outer_test=test,inner_validation_rows=tr[iv],windows=windows,full_ppm=ppm,full_Y=y/scale,scale=scale)
    files={str(p.relative_to(HERE)):exp.sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    metadata=dict(source_title=(SOURCE/'pdata/1/title').read_text(),known_nominal_solutes=['glucose','maltotriose'],
        nominal_concentration_mM_each=100,additional_reference='TSP, concentration unspecified in title',temperature_C=25,
        known_component_D=False,known_global_diffusion_order=False,readme_heading_conflict='Header says Maltose; source title and detailed experimental section both say glucose.',
        n_acquisitions=len(b),n_original_channels=n,n_selected_channels=int(mask.sum()),mask=details,
        outer_test_positions_one_based=[3,6],discovery_positions_one_based=(np.flatnonzero(discovery)+1).tolist(),
        processing=dict(reader='Unmodified nmrglue 0.11 Bruker module; module-only loading bypasses incompatible unrelated Tecmag importer',digital_filter='remove_digital_filter(truncate=False)',GRPDLY=acq['GRPDLY'],
            first_point_multiplier=proc['FCOR'],zero_fill=n,apodization='none; saved LB=0',
            complex_sign='nmrglue positive-imaginary FID conjugated to GNAT negative-imaginary convention after digital-filter correction',
            phase='One common linear ACME phase from discovery rows only, frozen before any model fit; exp(i*(p0+p1*t))',
            saved_PHC0=proc['PHC0'],saved_PHC1=proc['PHC1'],phase_degrees=phase_best.x,
            phase_objective='Unmodified nmrglue 0.11 _ps_acme_score on mean discovery spectrum normalized by complex maximum magnitude',
            phase_start_p0_degrees=[-180,-90,0,90,180],phase_bounds_degrees=[[-360,360],[-180,180]],
            phase_runs=[dict(x=r.x,fun=r.fun,success=r.success,nfev=r.nfev) for r in phase_runs],
            baseline_correction=False,rectification=False,ppm_axis='acquisition O1, SW_h and saved SF; saved processing SW_p is not used to redefine acquired bandwidth',
            acquisition_SW_h=swh,saved_processing_SW_p=proc['SW_p']),
        diffusion_encoding=dict(gradient_units='difflist G/cm converted to T/m; GNAT assumes shape-corrected difflist',
            gamma=gamma,delta=delta,Delta=Delta,tau=tau,alpha=alpha,time_cubed_current_GNAT=time3,
            stored_CNST18=acq['CNST'][18],relative_time_cube_difference=time3/acq['CNST'][18]-1,
            equation='b=gamma^2*g^2*delta^2*[Delta-(5-3*alpha^2)*delta/16-(1-alpha^2)*tau/2]',
            source=str(GNAT),source_sha256=exp.sha(GNAT),source_lines='451-498 and 578-590',calibration='nominal archive scale; no independent D calibration or reference accuracy claimed'),
        sigma_normalized=sigma/scale,source_hashes=files,input_sha256=exp.sha(HERE/'input.npz'))
    exp.save_json(HERE/'preprocessing.json',metadata)
    fig,axs=plt.subplots(2,1,figsize=(10,6))
    for row in np.flatnonzero(discovery):axs[0].plot(ppm,y[row]/scale,lw=.5,label=str(row+1))
    axs[0].set_xlim(ppm.max(),ppm.min());axs[0].set_title('Discovery acquisitions only: common ACME phase');axs[0].legend()
    lo=y[discovery].mean(0)/scale;axs[1].plot(ppm,lo,lw=.6)
    axs[1].plot(ppm[mask],lo[mask],'.',ms=1);axs[1].set_xlim(6,-.5);axs[1].set_xlabel('Chemical shift (ppm)')
    fig.tight_layout();fig.savefig(HERE/'preprocessing_discovery_QA.png',dpi=180);plt.close(fig)
    print(json.dumps({k:metadata[k] for k in ['n_acquisitions','n_original_channels','n_selected_channels','sigma_normalized','diffusion_encoding']},indent=2))

if __name__=='__main__':main()
