"""RAI-Net: a learned action ranker around the verified RAI-ILT controller.

Checkpoints are numeric NPZ arrays, read with allow_pickle=False. Training a
ranker does not change the physical objective or certify statistical accuracy.
"""
from pathlib import Path
import json
import numpy as np

FEATURES = ("active_fraction", "candidate_fraction", "merge", "position",
    "joint_score", "private_score", "raw_score", "information", "mass_fraction",
    "width_fraction", "penalty", "acquisitions", "signals", "objective")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("RAI-Net requires rai-ilt[learning]") from exc
    return torch


def make_network(hidden=24):
    torch = _torch()
    return torch.nn.Sequential(torch.nn.Linear(len(FEATURES), hidden), torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))


class RAINetPolicy:
    """Inference never receives held-out observations or true distributions."""
    def __init__(self, model, mean, scale, metadata=None):
        self.model = model.cpu().eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (len(FEATURES),) or self.scale.shape != self.mean.shape:
            raise ValueError("Invalid policy normalization shape")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("Invalid policy normalization")
        self.metadata = dict(metadata or {})

    def rank(self, features):
        torch = _torch()
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid RAI-Net features")
        with torch.inference_mode():
            score = self.model(torch.from_numpy((x-self.mean)/self.scale)).numpy().ravel()
        if not np.all(np.isfinite(score)):
            raise ValueError("Nonfinite policy score")
        return np.argsort(-score, kind="stable")

    def save(self, path, *, overwrite=False):
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        data = {"mean": self.mean, "scale": self.scale,
            "metadata_json": np.array(json.dumps(self.metadata)),
            "features": np.array(FEATURES), "format_version": np.array(1)}
        data.update({f"weight_{name}": v.detach().cpu().numpy() for name, v in self.model.state_dict().items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, **data)

    @classmethod
    def load(cls, path):
        torch = _torch()
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"]) != 1 or data["features"].tolist() != list(FEATURES):
                raise ValueError("Unsupported RAI-Net checkpoint")
            hidden = data["weight_0.weight"].shape[0]
            if hidden < 1 or hidden > 1024:
                raise ValueError("Invalid hidden width")
            model = make_network(hidden)
            state = model.state_dict()
            expected = {f"weight_{k}" for k in state}|{"mean", "scale", "metadata_json", "features", "format_version"}
            if set(data.files) != expected:
                raise ValueError("Unexpected checkpoint keys")
            for name, value in state.items():
                array = data[f"weight_{name}"]
                if array.shape != tuple(value.shape) or array.dtype.kind != 'f' or not np.all(np.isfinite(array)):
                    raise ValueError(f"Invalid checkpoint weight {name}")
                state[name] = torch.from_numpy(array.astype(np.float32))
            model.load_state_dict(state, strict=True)
            return cls(model, data["mean"], data["scale"], json.loads(str(data["metadata_json"])))


def train_policy(records, *, seed=1729, epochs=150, hidden=24, learning_rate=.003):
    """Fit action utility on development traces; split by entire problem id.

    Supervision is signed log internal-validation prediction gain. The validation split
    does not share inversion traces with training; its best epoch is restored.
    """
    torch = _torch()
    if not records or epochs < 1:
        raise ValueError("Need records and positive epochs")
    ids = sorted({str(r["problem_id"]) for r in records})
    if len(ids) < 5:
        raise ValueError("Need at least five independent development problems")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids)); validation_ids = {ids[i] for i in perm[:max(1, len(ids)//5)]}
    train = np.array([str(r["problem_id"]) not in validation_ids for r in records])
    x = np.array([r["features"] for r in records], dtype=np.float32)
    gains = np.array([r["validation_gain"] if r["optimizer"]["success"] else -1. for r in records])
    target = (np.sign(gains)*np.log1p(np.abs(gains))).astype(np.float32)
    if x.shape[1:] != (len(FEATURES),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(target)):
        raise ValueError("Nonfinite or incompatible training records")
    mean = x[train].mean(axis=0); scale = np.maximum(x[train].std(axis=0), .1)
    xx = torch.from_numpy((x-mean)/scale); yy = torch.from_numpy(target[:, None])
    # Keep caller RNG state intact; CPU training is reproducible for this model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_network(hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.SmoothL1Loss()
        best = float('inf'); best_state = None; history = []
        tr = torch.from_numpy(train); va = ~tr
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad()
            loss = criterion(model(xx[tr]), yy[tr]); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val = float(criterion(model(xx[va]), yy[va]))
            history.append({"epoch": epoch+1, "train_loss": float(loss.detach()), "validation_loss": val})
            if val < best:
                best = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    metadata = {"format": "RAI-Net ranker v1", "seed": seed, "epochs": epochs,
        "train_problem_ids": [x for x in ids if x not in validation_ids],
        "validation_problem_ids": sorted(validation_ids), "records": len(records),
        "best_validation_loss": best, "supervision": "signed log internal-validation prediction gain",
        "research_only": True, "speedup_established": False}
    return RAINetPolicy(model, mean, scale, metadata), history


def solve_net(problem, policy, config=None):
    from .adaptive import solve_adaptive
    if not isinstance(policy, RAINetPolicy):
        raise TypeError("policy must be a trained RAINetPolicy")
    result = solve_adaptive(problem, config, policy=policy)
    result.diagnostics["method"] = "RAI-Net"
    result.diagnostics["policy_metadata"] = policy.metadata
    return result
