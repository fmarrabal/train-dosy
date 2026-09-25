from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import nnls
from train_dosy import FitRequest,fit
from train_dosy._frozen.support import profile,amplitudes_exact,fit_support
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def payload():return json.loads((ROOT/'examples/one_component/input.json').read_text())

def test_frozen_sources():
    for record in json.loads((ROOT/'provenance/source_hashes.json').read_text()):
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']

@pytest.mark.parametrize('change',[{'sigma':0},{'bins':128},{'r_true':1},{'max_components':5},
    {'ppm':[1.,1.]},{'diffusion_bounds':[1,0]},{'sigma':float('nan')},{'mask':[False]}])
def test_invalid_input(payload,change):
    payload.update(change)
    with pytest.raises(ValidationError):FitRequest.model_validate(payload)

def test_nnls_kkt_and_independent_solver():
    rng=np.random.default_rng(12);b=np.linspace(0,2e9,24)
    for d in ([.5e-9,2e-9,7e-9],[1e-9,1.0001e-9,3e-9]):
        k=np.exp(-b[:,None]*np.array(d));y=rng.normal(size=(24,17))
        a=amplitudes_exact(k,y);ref=np.column_stack([nnls(k,y[:,i],maxiter=200)[0] for i in range(17)])
        assert np.linalg.norm(k@(a-ref))<1e-7
        assert np.max(abs(np.minimum(a,k.T@(k@a-y))))<1e-7

def test_variable_projection_derivative():
    rng=np.random.default_rng(74);b=np.linspace(0,2e9,24);d=np.array([.7,2.1,6.5])*1e-9
    a=rng.uniform(.1,1,(3,8));a[0,4:]=0
    y=np.exp(-b[:,None]*d)@a+rng.normal(0,.002,(24,8));sup=np.ones_like(a,bool);sup[0,4:]=False
    z=np.log(d*1.03);_,_,j=profile(z,b,y,.002,sup,True);h=1e-6
    fd=np.column_stack([(profile(z+np.eye(3)[i]*h,b,y,.002,sup)[0].ravel()-profile(z-np.eye(3)[i]*h,b,y,.002,sup)[0].ravel())/(2*h) for i in range(3)])
    assert np.linalg.norm(fd-j)/np.linalg.norm(fd)<1e-6

def test_adapter_matches_frozen_solver_and_mass(payload):
    q=fit(payload);m=np.array(payload['mask'])
    raw=fit_support(np.array(payload['b']),np.array(payload['Y'])[:,m],payload['sigma'],np.array(payload['ppm'])[m])
    np.testing.assert_allclose(q['prediction'],raw['prediction'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.sum(q['X'],axis=0),np.sum(q['A'],axis=0),rtol=1e-13)
    assert q['success'] and q['selected_rank']==1

def test_frequency_permutation_scaling(payload):
    ref=fit(payload);p=np.random.default_rng(85).permutation(len(payload['ppm']))
    payload.update(Y=(np.array(payload['Y'])[:,p]*100).tolist(),sigma=payload['sigma']*100,
        ppm=np.array(payload['ppm'])[p].tolist(),mask=np.array(payload['mask'])[p].tolist())
    changed=fit(payload)
    np.testing.assert_allclose(changed['D'],ref['D'],rtol=1e-5)
    original_order=np.argsort(np.array(ref['ppm']));new_order=np.argsort(np.array(changed['ppm']))
    np.testing.assert_allclose(np.array(changed['A'])[:,new_order]/100,np.array(ref['A'])[:,original_order],rtol=1e-5,atol=1e-7)

def test_api_errors_and_result(payload):
    from fastapi.testclient import TestClient
    from train_dosy.api import app
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.post('/v1/fit',json={**payload,'bins':128}).status_code==422
        result=client.post('/v1/fit',json=payload)
        assert result.status_code==200 and result.json()['selected_rank']==1
        assert client.get('/openapi.json').json()['info']['version']=='0.1.0'
