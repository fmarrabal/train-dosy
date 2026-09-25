"""Local REST service. Do not expose without deployment-level authentication."""
import threading
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse
from . import __version__, fit, FitRequest

app=FastAPI(title='TRAIn-DOSY',version=__version__,description='Joint positive ILT research service; SI units; automatic order selection')
gate=threading.BoundedSemaphore(1)
@app.middleware('http')
async def body_limit(request:Request,call_next):
    if request.method=='POST':
        chunks=[];total=0
        async for chunk in request.stream():
            total+=len(chunk)
            if total>16*1024*1024:return JSONResponse({'detail':'Maximum request body is 16 MiB'},status_code=413)
            chunks.append(chunk)
        body=b''.join(chunks)
        async def receive():return {'type':'http.request','body':body,'more_body':False}
        request._receive=receive
        request._body=body
    return await call_next(request)

@app.get('/health')
def health():return {'status':'ok','version':__version__}

@app.post('/v1/fit')
def invert(request:FitRequest):
    if not gate.acquire(blocking=False):raise HTTPException(429,'One fit is already running')
    try:return fit(request)
    except ValueError as e:raise HTTPException(422,str(e)) from e
    except RuntimeError as e:raise HTTPException(503,str(e)) from e
    finally:gate.release()
