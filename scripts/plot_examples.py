"""Scientifically scaled DOSY display: ground truth and both estimators."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
for row,name in enumerate(['one_component','two_components','three_components']):
    folder=ROOT/'examples'/name
    q=json.loads((folder/'input.json').read_text());truth=json.loads((folder/'truth.json').read_text())
    ppm=np.array(q['ppm']);mask=np.array(q['mask']);edges=np.linspace(np.log(.1e-9),np.log(15e-9),257)
    x=np.zeros((256,len(ppm)))
    np.add.at(x,np.clip(np.searchsorted(edges,np.log(truth['D']),side='right')-1,0,255),truth['A'])
    panels=[(x,'Ground truth')]
    for method in ['RAI-S','DOME-S']:
        result=json.loads((folder/(method+'.json')).read_text());xx=np.full_like(x,np.nan);xx[:,mask]=result['X']
        panels.append((xx,f"{method}: r = {result['selected_rank']}"))
    norm=LogNorm(vmin=x.max()*1e-3,vmax=x.max())
    for col,(matrix,title) in enumerate(panels):
        ax=axes[row,col]
        artist=ax.pcolormesh(ppm,np.exp((edges[1:]+edges[:-1])/2)*1e9,np.ma.masked_less(matrix,norm.vmin),norm=norm,cmap='viridis',rasterized=True)
        ax.set_title(f'{row+1} component'+('s' if row else '')+' | '+title,fontsize=10)
        ax.set_yscale('log');ax.set_xlim(9,0.5);ax.set_ylim(15,.1)
        if col==0:ax.set_ylabel('D (10$^{-9}$ m$^2$/s)')
        if row==2:ax.set_xlabel('Chemical shift (ppm)')
    fig.colorbar(artist,ax=axes[row,:].tolist(),label='Bin mass (signal units)',shrink=.8)
fig.suptitle('Correlated spectral frequencies: 1, 2 and 3 discrete components\nShared row scales and 0.1% display floor; no smoothing of diffusion bands',fontsize=13)
fig.savefig(ROOT/'examples/dosy_ground_truth.png',dpi=190)
fig.savefig(ROOT/'examples/dosy_ground_truth.pdf')
