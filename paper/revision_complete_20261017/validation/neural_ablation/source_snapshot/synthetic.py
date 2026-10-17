"""Independent whole-problem partitions with correlated spectral frequencies."""
import numpy as np
from scipy.ndimage import binary_dilation

SCENARIOS = ('single','double','triple','weak','close','proportional','null','broad')


def generate(index, partition='development'):
    base = dict(development=510000, net_train=610000, net_validation=710000, test=810000)[partition]
    seed = base+index
    rng = np.random.default_rng(seed)
    scenario = SCENARIOS[index % len(SCENARIOS)]
    r = {'single':1,'double':2,'null':0}.get(scenario,3)
    snr = (50,100,200)[(index//len(SCENARIOS)) % 3]
    ppm = np.linspace(10,0,1024)
    b = np.linspace(0,1,32)**rng.uniform(1.5,2.4)*rng.uniform(1.3,2.7)*1e9
    centers = np.exp(np.log([.65,2.3,7.1])[:r]+rng.normal(0,.10,r))
    if scenario == 'close':
        centers = np.array([.8,1.05,5.5])*np.exp(rng.normal(0,.02,3))
    a = np.zeros((r,len(ppm)))
    peaks = np.array([7.73,5.35,4.20,3.2,2.05,1.35])+rng.normal(0,.04,6)
    for j in range(r):
        for center, amp in [(peaks[0]+.018*j,rng.uniform(.35,1.)),(peaks[1+j],rng.uniform(.5,1.)),(peaks[4+j%2],rng.uniform(.3,.8))]:
            a[j] += amp*np.exp(-.5*((ppm-center)/rng.uniform(.03,.06))**2)
    if scenario == 'proportional':
        a = np.array([1.,.7,.45])[:,None]*a[0]
    if scenario == 'weak':
        a[-1] *= .04  # Real minor species, cannot be removed by a global height cutoff.
    if r:
        a /= a.sum(0).max()
    widths = np.full(r, .18 if scenario == 'broad' else 0.)
    if scenario == 'broad':
        nodes, weights = np.polynomial.hermite.hermgauss(24)
        curves = np.column_stack([np.exp(-b[:,None]*d*1e-9*np.exp(np.sqrt(2)*w*nodes))@(weights/np.sqrt(np.pi)) for d,w in zip(centers,widths)])
    else:
        curves = np.exp(-b[:,None]*centers*1e-9)
    clean = curves@a
    sigma = 1/snr
    y = clean+rng.normal(0,sigma,clean.shape)
    test = np.arange(len(b))%4 == 3; train = ~test
    # Frequency screening never uses the reserved external test gradients.
    mask = binary_dilation(np.mean(y[train][:6],axis=0)>4*sigma/np.sqrt(6),iterations=2)
    if scenario == 'null':
        # Deliberately inspect a predefined noise window to test false positives.
        mask = (ppm>4)&(ppm<4.65)
    return dict(id=f'{partition}_{index:03d}_{scenario}',seed=seed,scenario=scenario,r_true=r,snr=snr,
        b=b,Y=y,Y_clean=clean,sigma=sigma,ppm=ppm,mask=mask,train=train,test=test,
        truth_D=centers*1e-9,truth_A=a,truth_widths=widths)
