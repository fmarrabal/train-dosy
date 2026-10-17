from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
z=loadmat(root/'inputs/rank3_snr100.mat',squeeze_me=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':180})
fig,axs=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
colors=['#176b96','#ad5a20','#4b8744'];x=np.log(z['D']);dx=np.diff(x);w=np.r_[dx[0]/2,(dx[:-1]+dx[1:])/2,dx[-1]/2]
for j,c in enumerate(colors):
 axs[0].plot(z['ppm'],z['Atrue'][j],color=c,label=f'Profile {j+1}')
 axs[1].semilogx(z['D']/1e-9,z['Strue'][:,j]/w,color=c)
axs[0].invert_xaxis();axs[0].set(xlabel='Chemical shift (ppm)',ylabel='True spectral amplitude',title='24 signal-bearing channels')
axs[0].legend(frameon=False,fontsize=8)
axs[1].set(xlabel=r'$D/(10^{-9}\,\mathrm{m^2\,s^{-1}})$',ylabel='True density per unit log D',title='Finite-width generating profiles')
for j in range(z['Y'].shape[1]):
 axs[2].plot(z['b']/1e9,z['Yclean'][:,j],color='#176b96',alpha=.28,lw=.8)
fit=z['fit_rows'].astype(bool)
axs[2].plot(z['b'][fit]/1e9,z['Y'][fit].mean(1),'o',color='#176b96',ms=3,label='Fitting mean')
axs[2].plot(z['b'][~fit]/1e9,z['Y'][~fit].mean(1),'s',color='#ad5a20',ms=3,label='External mean')
axs[2].set(xlabel=r'$b/(10^9\,\mathrm{s\,m^{-2}})$',ylabel='Attenuation',title='32 acquisitions; SNR 100')
axs[2].legend(frameon=False,fontsize=8)
for ax in axs:ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'synthetic_design.pdf');fig.savefig(root/'synthetic_design.png');plt.close(fig)
