"""Conditional positive-moment prior for RAI-Net (experimental).

A permutation-equivariant set network predicts a positive local measure.
Its moments define a fixed quadratic energy in the semidefinite inversion.
Checkpoints contain numeric arrays and JSON only, never Python pickle.
"""
import json
from pathlib import Path
import zipfile
import numpy as np
from .moment_space import moment_sequence


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("RAI-Net moments requires rai-ilt[learning,moments]") from exc
    return torch


def make_network(hidden=48,atoms=4):
    torch=_torch()
    nn=torch.nn

    class PositiveSetPrior(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(4,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
            self.decoder=nn.Sequential(nn.Linear(2*hidden+3,hidden),nn.SiLU(),nn.Linear(hidden,2*atoms))
            self.mass=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))

        def forward(self,features,regions):
            # features: batch, signals, acquisitions, 4. No spectral order or
            # acquisition order is assumed. Pooling couples all DOSY signals.
            own=self.encoder(features).mean(dim=2)
            shared=own.mean(dim=1,keepdim=True).expand_as(own)
            encoded=torch.cat((own,shared),dim=-1)
            r=regions.shape[1]
            descriptors=regions[:,None].expand(-1,features.shape[1],-1,-1)
            output=self.decoder(torch.cat((encoded[:,:,None].expand(-1,-1,r,-1),descriptors),dim=-1))
            nodes=torch.tanh(output[...,:atoms])
            # Region width is a base measure; unequal subdivisions do not
            # automatically receive equal total prior mass.
            logits=output[...,atoms:]+torch.log(regions[:,None,:,2:3].clamp_min(1e-8))
            weights=torch.softmax(logits.flatten(-2),dim=-1).reshape_as(nodes)
            total=torch.nn.functional.softplus(self.mass(encoded))
            return nodes,weights*total[:,:,:,None]

    return PositiveSetPrior()


def features_for(b,y,sigma,bounds):
    b=np.asarray(b,dtype=float).reshape(-1)
    y=np.asarray(y,dtype=float); sigma=np.broadcast_to(sigma,y.shape)
    reference=np.sqrt(bounds[0])*np.sqrt(bounds[1])
    x=b*reference
    return np.stack((np.broadcast_to(np.log1p(x)[:,None],y.shape),y,
                     np.log10(np.maximum(sigma,1e-12))/6,
                     np.broadcast_to((x/max(float(x.max()),1e-12))[:,None],y.shape)),axis=-1).transpose(1,0,2)


def region_features(space):
    lower,upper=space.regions[0].lower,space.regions[-1].upper
    span=np.log(upper/lower)
    return np.array([[np.log(r.center/lower)/span,r.halfwidth/r.center,
                      np.log(r.upper/r.lower)/span] for r in space.regions])


class MomentPrior:
    def __init__(self,network,metadata):
        self.network=network.cpu().eval()
        self.metadata=dict(metadata)

    def reference(self,b,y,sigma,space):
        torch=_torch()
        features=features_for(b,y,sigma,(space.regions[0].lower,space.regions[-1].upper))
        regions=region_features(space)
        with torch.inference_mode():
            u,a=self.network(torch.as_tensor(features[None],dtype=torch.float32),
                             torch.as_tensor(regions[None],dtype=torch.float32))
        u,a=u[0].numpy().astype(float),a[0].numpy().astype(float)
        moments=[]
        for j,r in enumerate(space.regions):
            moments.append(np.column_stack([moment_sequence(u[q,j],a[q,j],r.degree)[:,0] for q in range(y.shape[1])]))
        return np.vstack(moments)

    def save(self,path):
        path=Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        data={"weight_"+key:value.detach().cpu().numpy() for key,value in self.network.state_dict().items()}
        data['metadata_json']=np.array(json.dumps(self.metadata,allow_nan=False))
        with path.open('xb') as stream:
            np.savez_compressed(stream,**data)

    @classmethod
    def load(cls,path):
        path=Path(path)
        # Bound decompression before loading an untrusted numeric archive.
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist())>64 or sum(i.file_size for i in archive.infolist())>20_000_000:
                raise ValueError("Moment prior archive exceeds size limits")
        torch=_torch()
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata_json'].item()))
            if metadata.get('format')!='rai-positive-moment-prior-v1':
                raise ValueError("Unsupported moment prior format")
            hidden,atoms=metadata.get('hidden'),metadata.get('atoms')
            if type(hidden) is not int or type(atoms) is not int or not 8<=hidden<=128 or not 1<=atoms<=8:
                raise ValueError("Invalid prior architecture")
            # Loading a checkpoint must not consume the caller's RNG stream.
            with torch.random.fork_rng(devices=[]):
                network=make_network(hidden,atoms)
            expected=network.state_dict()
            if set(data.files)!={'metadata_json',*("weight_"+key for key in expected)}:
                raise ValueError("Unexpected or missing checkpoint arrays")
            values={}
            for key,template in expected.items():
                array=data['weight_'+key]
                if array.dtype.kind!='f' or tuple(array.shape)!=tuple(template.shape) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid checkpoint array {key}")
                values[key]=torch.as_tensor(array,dtype=template.dtype)
            network.load_state_dict(values)
        return cls(network,metadata)


def train_prior(training,validation,*,epochs=120,batch_size=32,hidden=48,atoms=4,seed=27001):
    """Train from explicit, disjoint whole-problem synthetic dictionaries.

    Arrays: features (N,P,B,4), regions (N,R,3), target (N,P,R,L),
    and integer problem_ids. All examples in each call use the same shapes.
    """
    torch=_torch()
    train_ids=np.asarray(training['problem_ids']); val_ids=np.asarray(validation['problem_ids'])
    if len(np.unique(train_ids))!=len(train_ids) or len(np.unique(val_ids))!=len(val_ids) or np.intersect1d(train_ids,val_ids).size:
        raise ValueError("Training and validation require unique disjoint problem IDs")
    if min(len(train_ids),len(val_ids),epochs,batch_size)<1:
        raise ValueError("Need nonempty data and positive training limits")
    def tensors(data):
        result=[torch.as_tensor(data[key],dtype=torch.float32) for key in ('features','regions','target')]
        if not all(torch.isfinite(t).all() for t in result):
            raise ValueError("Nonfinite training arrays")
        return result
    tr,va=tensors(training),tensors(validation)
    def loss(model,arrays):
        f,r,target=arrays
        u,a=model(f,r)
        powers=torch.arange(target.shape[-1],device=u.device)
        predicted=(u[...,None]**powers*a[...,None]).sum(dim=-2)
        # Region mass is essential; scale even/odd higher moments equally.
        return ((predicted-target)**2).mean()+((predicted[...,0]-target[...,0])**2).mean()
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model=make_network(hidden,atoms)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-5)
            trace=[]; best=float('inf'); state=None
            for epoch in range(epochs):
                model.train()
                order=torch.randperm(len(train_ids))
                total=0.
                for batch in order.split(batch_size):
                    optimizer.zero_grad()
                    value=loss(model,[t[batch] for t in tr]); value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step(); total+=float(value.detach())*len(batch)
                model.eval()
                with torch.no_grad():
                    val=float(loss(model,va))
                trace.append(dict(epoch=epoch+1,training_loss=total/len(train_ids),validation_loss=val))
                if val<best:
                    best=val; state={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(state)
    finally:
        torch.set_num_threads(old_threads)
    metadata=dict(format='rai-positive-moment-prior-v1',hidden=hidden,atoms=atoms,seed=seed,
                  training_problem_ids=train_ids.tolist(),validation_problem_ids=val_ids.tolist(),
                  epochs=epochs,best_validation_loss=best,torch_version=str(torch.__version__),
                  training_domain=training.get('domain',{}),
                  interpretation='synthetic conditional reference; no calibrated uncertainty or recovery guarantee')
    return MomentPrior(model,metadata),trace
