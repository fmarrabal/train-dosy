"""Contract and optional real MATLAB integration for the signed MF revision."""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.io import loadmat
from train_dosy import FitRequest, fit

ROOT=Path(__file__).resolve().parents[1]

def test_distinct_mf_protocols_and_search_cap():
    payload=json.loads((ROOT/'examples/one_component/input.json').read_text())
    q=FitRequest.model_validate({**payload,'method':'TRAIn-MF','max_components':2})
    assert q.method=='TRAIn-MF' and q.max_components==2
    with pytest.raises(ValidationError):
        FitRequest.model_validate({**payload,'method':'MF-AUTO','max_components':2})

@pytest.mark.skipif(not os.environ.get('TRAIN_DOSY_MATLAB'),reason='Licensed MATLAB backend not configured')
def test_signed_matlab_integration():
    # Deliberately include one selected frequency to exercise MATLAB's JSON
    # singleton collapse at the Python/API boundary, plus a signed tail.
    b=12e9*np.linspace(0,1,24)**2
    y=np.exp(-b*.7e-9)+.0005*np.sin(np.arange(24)*1.3)
    y[-1]=-.0002
    q=dict(Y=y[:,None].tolist(),b=b.tolist(),ppm=[7.73],sigma=.0005,
           method='TRAIn-MF',mask=[True],diffusion_bounds=[.02e-9,5e-9],bins=256,max_components=1)
    from fastapi.testclient import TestClient
    from train_dosy.api import app
    with TestClient(app) as client:
        response=client.post('/v1/fit',json=q)
    assert response.status_code==200, response.text
    result=response.json()
    assert result['method']=='TRAIn-MF' and result['protocol']=='signed-v2.2'
    assert result['software_version']=='0.2.1'
    assert result['kkt'] is None
    assert np.asarray(result['X']).shape==(256,1)
    assert np.asarray(result['S']).shape==(256,1)
    assert np.asarray(result['A']).shape==(1,1)
    assert result['diagnostics']['input_negative_count']==int(np.count_nonzero(y<0))
    assert result['diagnostics']['model_compatible'] is not None
    assert all(key in result['diagnostics'] for key in (
        'numerical_converged','rank_selection_resolved','active_rank','boundary_hit'))
    d=np.asarray(result['D_grid']);x=np.asarray(result['X'])[:,0]
    assert abs(d@x/x.sum()/.7e-9-1)<.04
    np.testing.assert_allclose(result['prediction'],np.exp(-b[:,None]*d)@x[:,None])


@pytest.mark.skipif(not os.environ.get('TRAIN_DOSY_MATLAB'),reason='Licensed MATLAB backend not configured')
@pytest.mark.parametrize('columns',[1,3])
def test_null_signed_matlab_api_integration(columns):
    """An explicit null window completes through MATLAB, HTTP and JSON."""
    from fastapi.testclient import TestClient
    from train_dosy.api import app
    b=12e9*np.linspace(0,1,24)**2
    q=dict(Y=np.zeros((24,columns)).tolist(),b=b.tolist(),
           ppm=np.linspace(7.,8.,columns).tolist(),sigma=.001,method='TRAIn-MF',
           mask=[True]*columns,diffusion_bounds=[.02e-9,5e-9],bins=256,max_components=3)
    with TestClient(app) as client:
        response=client.post('/v1/fit',json=q)
    assert response.status_code==200,response.text
    result=response.json()
    assert result['protocol']=='signed-v2.2' and result['selected_rank']==0
    assert result['active_rank']==0 and result['diagnostics']['active_rank']==0
    assert result['status']=='no_signal_supported' and result['success']
    assert np.asarray(result['X']).shape==(256,columns)
    assert np.asarray(result['S']).shape==(256,0)
    assert np.asarray(result['A']).reshape(0,columns).shape==(0,columns)
    assert np.asarray(result['prediction']).shape==(24,columns)
    assert not np.any(result['X']) and not np.any(result['prediction'])
    assert result['diagnostics']['model_compatible'] is True
    assert not result['chemical_species_identified']


@pytest.mark.parametrize('columns',[1,3])
def test_null_matlab_result_preserves_factor_shapes_and_diagnostics(monkeypatch,columns):
    """MATLAB collapses empty dimensions; rank zero must survive the adapter."""
    import train_dosy.mf as backend
    monkeypatch.setenv('TRAIN_DOSY_MATLAB','mock-matlab')
    monkeypatch.setenv('TRAIN_DOSY_MATLAB_FUNCTIONS',str(ROOT/'matlab'))
    b=12e9*np.linspace(0,1,24)**2
    d=np.geomspace(.02e-9,5e-9,256)
    q=dict(Y=np.zeros((24,columns)).tolist(),b=b.tolist(),
           ppm=np.linspace(7.,8.,columns).tolist(),sigma=.001,method='TRAIn-MF',
           mask=[True]*columns,diffusion_bounds=[.02e-9,5e-9],bins=256,max_components=3)

    def run(args,**kwargs):
        match=re.search(r"run_mf_api\('((?:[^']|'')*)','((?:[^']|'')*)'\)",args[-1])
        assert match is not None
        source,target=(Path(value.replace("''","'")) for value in match.groups())
        incoming=loadmat(source)
        assert incoming['sigma'].item()==q['sigma']
        assert incoming['Y'].shape==(24,columns)
        output=dict(selected_rank=0,status='no_signal_supported',success=True,
                    X=np.zeros((256,columns)).tolist(),D_grid=d.tolist(),A=[],S=[],kkt=[],
                    chemical_species_identified=False,protocol='signed-v2.2',
                    diagnostics=dict(numerical_converged=True,model_compatible=True,
                                     rank_selection_resolved=True,active_rank=0,boundary_hit=False))
        target.write_text(json.dumps(output),encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend.subprocess,'run',run)
    result=fit(q)
    assert result['selected_rank']==0 and result['success']
    assert np.asarray(result['X']).shape==(256,columns)
    assert np.asarray(result['S']).shape==(256,0)
    # JSON cannot encode the second dimension of an empty leading axis.
    # The declared rank and number of ppm columns restore the (0,p) shape.
    assert np.asarray(result['A']).reshape(0,columns).shape==(0,columns)
    assert np.asarray(result['prediction']).shape==(24,columns)
    assert not np.any(result['X']) and not np.any(result['prediction'])
    assert result['diagnostics']['active_rank']==0
    assert result['diagnostics']['model_compatible'] is True
    assert result['kkt'] is None and not result['chemical_species_identified']


def test_unknown_model_compatibility_serializes_as_null():
    from train_dosy.inversion import clean_json
    diagnostics=clean_json(dict(model_compatible=np.nan,numerical_converged=True,success=False))
    assert diagnostics['model_compatible'] is None
    assert json.loads(json.dumps(diagnostics,allow_nan=False))==diagnostics


def test_inactive_matlab_factor_keeps_selected_rank_but_uses_active_shapes(monkeypatch):
    import train_dosy.mf as backend
    monkeypatch.setenv('TRAIN_DOSY_MATLAB','mock-matlab')
    monkeypatch.setenv('TRAIN_DOSY_MATLAB_FUNCTIONS',str(ROOT/'matlab'))
    b=12e9*np.linspace(0,1,24)**2;d=np.geomspace(.02e-9,5e-9,256)
    s=np.zeros((256,1));s[164,0]=1.;a=np.array([[1.,.7]])
    y=np.exp(-b[:,None]*d)@(s@a)
    q=dict(Y=y.tolist(),b=b.tolist(),ppm=[7.73,4.42],sigma=.001,
           method='TRAIn-MF',mask=[True,True],diffusion_bounds=[.02e-9,5e-9],bins=256,max_components=3)

    def run(args,**kwargs):
        match=re.search(r"run_mf_api\('((?:[^']|'')*)','((?:[^']|'')*)'\)",args[-1])
        assert match is not None
        target=Path(match.group(2).replace("''","'"))
        output=dict(selected_rank=2,active_rank=1,status='train_mf_requires_review',success=False,
                    X=(s@a).tolist(),D_grid=d.tolist(),S=s.ravel().tolist(),A=a.ravel().tolist(),
                    kkt=[],chemical_species_identified=False,protocol='signed-v2.2',
                    diagnostics=dict(active_rank=1,selected_rank=2,numerical_converged=False,
                                     model_compatible=True,rank_selection_resolved=False,boundary_hit=False))
        target.write_text(json.dumps(output),encoding='utf-8')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend.subprocess,'run',run)
    result=fit(q)
    assert result['selected_rank']==2 and result['active_rank']==1 and not result['success']
    assert np.asarray(result['S']).shape==(256,1)
    assert np.asarray(result['A']).shape==(1,2)
    np.testing.assert_allclose(np.asarray(result['S'])@result['A'],result['X'])
    np.testing.assert_allclose(result['prediction'],y)
