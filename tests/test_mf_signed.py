"""Contract and optional real MATLAB integration for the signed MF revision."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
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
    assert result['method']=='TRAIn-MF' and result['protocol']=='signed-v2.1'
    assert result['kkt'] is None
    assert np.asarray(result['X']).shape==(256,1)
    assert np.asarray(result['S']).shape==(256,1)
    assert np.asarray(result['A']).shape==(1,1)
    assert result['diagnostics']['input_negative_count']==int(np.count_nonzero(y<0))
    d=np.asarray(result['D_grid']);x=np.asarray(result['X'])[:,0]
    assert abs(d@x/x.sum()/.7e-9-1)<.04
    np.testing.assert_allclose(result['prediction'],np.exp(-b[:,None]*d)@x[:,None])
