"""RAI v0.4: conservative resolution search with joint Haar regularization.

Fixed-tree optimization is convex. Adaptive selection is empirical, not a
certificate of continuous resolution. The neural policy only proposes actions.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from .model import DecayProblem, bin_kernel
from .tree import ResolutionTree


@dataclass(frozen=True)
class AdaptiveConfig:
    fine_bins: int = 256
    initial_active: int = 8
    max_active: int = 64
    max_rounds: int = 10
    penalties: tuple = (0., .01, .1)
    ridge: float = 1e-4
    joint: bool = True
    conditional: bool = True
    allow_merge: bool = True
    validation_stride: int = 4
    descent_tolerance: float = 1e-9
    split_batch: int = 4
    max_iterations: int = 3000
    gap_tolerance: float = 1e-5
    rho: float = .5
    tangent_threshold: float = 1.
    min_information: float = .05
    selection_slack: float = .005
    policy_shortlist: int = 1

    def __post_init__(self):
        for key in ("fine_bins", "initial_active", "max_active", "max_rounds", "validation_stride",
                    "split_batch", "max_iterations", "policy_shortlist"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if key == "max_rounds" else 1):
                raise ValueError(f"Invalid {key}")
        if self.fine_bins not in (128, 256):
            raise ValueError("fine_bins must be 128 or 256")
        if self.initial_active & (self.initial_active-1) or not self.initial_active <= self.max_active <= self.fine_bins:
            raise ValueError("Invalid active tree limits")
        if self.validation_stride < 3:
            raise ValueError("validation_stride must be at least three")
        for key in ("ridge", "gap_tolerance", "rho", "tangent_threshold"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("descent_tolerance", "min_information", "selection_slack"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        vals = tuple(float(x) for x in self.penalties)
        if not vals or len(set(vals)) != len(vals) or any(not np.isfinite(x) or x < 0 for x in vals):
            raise ValueError("penalties must be distinct finite nonnegative values")
        object.__setattr__(self, "penalties", vals)
        for key in ("joint", "conditional", "allow_merge"):
            if not isinstance(getattr(self, key), (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean")


def _penalty(c, lam, joint):
    if joint:
        return lam*np.sqrt(c.shape[1])*np.linalg.norm(c[1:], axis=1).sum()
    return lam*np.abs(c[1:]).sum()


def fit_tree(kfine, y, sigma, tree, lam, cfg, initial=None):
    """ADMM with a feasible primal/dual gap; lambda=0 uses ridge NNLS.

    Coordinates u are sqrt-width-scaled leaf densities. Q is orthogonal.
    Objective: .5/n ||(A u-y)/sigma||Â² + eta/2 ||u||Â² + group(Q u).
    Returned masses are always nonnegative; failure remains visible.
    """
    pmat = tree.prolongation(); a = kfine @ pmat
    qmat, _ = tree.coefficients()
    n, p = y.shape; m = a.shape[1]
    aw = a[None, :, :]/sigma.T[:, :, None]/np.sqrt(n)
    bw = y.T/sigma.T/np.sqrt(n)
    hess = np.einsum('pni,pnj->pij', aw, aw)+cfg.ridge*np.eye(m)[None]
    rhs = np.einsum('pni,pn->pi', aw, bw).T
    u = np.zeros((m, p)) if initial is None else tree.restrict(initial)
    if lam == 0:
        iterations = 1; success = True
        for j in range(p):
            try:
                u[:, j] = nnls(np.vstack((aw[j], np.sqrt(cfg.ridge)*np.eye(m))),
                               np.r_[bw[j], np.zeros(m)], maxiter=cfg.max_iterations)[0]
            except RuntimeError:
                success = False
        grad = np.einsum('pij,jp->ip', hess, u)-rhs
        projected = u-np.maximum(0, u-grad)
        # Fenchel lower bound with a dual-feasible nonpositive normal.
        dv = np.minimum(-grad, 0); dz = np.zeros_like(u)
    else:
        rho = cfg.rho
        v = np.maximum(u, 0); z = qmat@u
        uv = np.zeros_like(u); uz = np.zeros_like(u)
        success = False
        # Identical acquisition noise permits a single factorization for all columns.
        shared = np.all(sigma == sigma[:, :1])
        factors = [cho_factor(hess[j]+2*rho*np.eye(m), lower=True, check_finite=False)
                   for j in range(1 if shared else p)]
        for iterations in range(1, cfg.max_iterations+1):
            target = rhs+rho*(v-uv+qmat.T@(z-uz))
            if shared:
                u = cho_solve(factors[0], target, check_finite=False)
            else:
                u = np.column_stack([cho_solve(factors[j], target[:, j], check_finite=False) for j in range(p)])
            v = np.maximum(u+uv, 0)
            trial = qmat@u+uz; z = trial.copy()
            if cfg.joint:
                norms = np.linalg.norm(trial[1:], axis=1, keepdims=True)
                z[1:] *= np.maximum(0, 1-lam*np.sqrt(p)/rho/np.maximum(norms, 1e-300))
            else:
                z[1:] = np.sign(trial[1:])*np.maximum(0, np.abs(trial[1:])-lam/rho)
            uv += u-v; uz += qmat@u-z
            if iterations % 25 == 0 or iterations == cfg.max_iterations:
                dv = np.minimum(rho*uv, 0); dz = rho*uz; dz[0] = 0
                if cfg.joint:
                    norms = np.linalg.norm(dz[1:], axis=1, keepdims=True)
                    dz[1:] *= np.minimum(1, lam*np.sqrt(p)/np.maximum(norms, 1e-300))
                else:
                    dz[1:] = np.clip(dz[1:], -lam, lam)
                gap, obj = _gap(v, dv, dz, hess, rhs, qmat, aw, bw, lam, cfg)
                if gap <= cfg.gap_tolerance*(1+obj):
                    success = True
                    break
        u = v
        projected = u-np.maximum(0, u-(np.einsum('pij,jp->ip', hess, u)-rhs+qmat.T@dz))
    gap, obj = _gap(u, dv, dz, hess, rhs, qmat, aw, bw, lam, cfg)
    success = bool(success and gap <= cfg.gap_tolerance*(1+obj))
    return pmat@u, {"success": success, "iterations": iterations, "objective": obj,
        "duality_gap": gap, "relative_gap": gap/(1+obj),
        "projected_stationarity_inf": float(np.max(np.abs(projected))),
        "message": "primal-dual gap satisfied" if success else "iteration or numerical limit",
        "engine": "ridge NNLS" if lam == 0 else "ADMM", "penalty": float(lam)}


def _gap(u, dv, dz, hess, rhs, qmat, aw, bw, lam, cfg):
    target = rhs-dv-qmat.T@dz
    dual_u = np.linalg.solve(hess, target.T[..., None])[..., 0].T
    delta = u-dual_u
    gap = .5*np.einsum('ip,pij,jp->', delta, hess, delta)
    c = qmat@u
    gap += _penalty(c, lam, cfg.joint)-np.sum(dz*c)-np.sum(dv*u)
    residual = np.einsum('pni,ip->pn', aw, u)-bw
    obj = .5*np.sum(residual**2)+.5*cfg.ridge*np.sum(u*u)+_penalty(c, lam, cfg.joint)
    return float(max(0., gap)), float(obj)


def candidate_scores(b, kfine, y, sigma, tree, masses, cfg):
    """Residualized mass-conserving contrasts; training observations only.

    Jacobian columns are scaled by current regional mass and log width. Only
    noise-visible singular directions are removed. Scores are uncalibrated.
    """
    l = len(tree.leaves); p = y.shape[1]
    residual = (y-kfine@masses)/sigma
    regional = np.array([masses[a:c].sum(axis=0) for a, c in tree.leaves])
    kernel = kfine @ tree.prolongation()/np.sqrt(tree.widths)[None, :]
    centers = np.array([np.sqrt(tree.edges[a]*tree.edges[c]) for a, c in tree.leaves])
    shift = -b[:, None]*centers[None, :]*np.exp(-b[:, None]*centers[None, :])
    score = np.zeros((l, p)); raw = np.zeros_like(score); info = np.zeros_like(score)
    rank = []
    for col in range(p):
        mass = regional[:, col]
        occupied = mass > max(float(mass.max())*.02, float(np.median(sigma[:, col])))
        jmass = kernel[:, occupied]*mass[occupied]
        jshift = shift[:, occupied]*(mass[occupied]*tree.widths[occupied]/2)
        jac = np.hstack((jmass, jshift))/sigma[:, col, None]
        if cfg.conditional and jac.shape[1]:
            uu, ss, _ = np.linalg.svd(jac, full_matrices=False)
            u = uu[:, ss >= cfg.tangent_threshold]
        else:
            u = np.zeros((len(b), 0))
        rank.append(u.shape[1])
        for j, (a, c) in enumerate(tree.leaves):
            if c-a == 1:
                continue
            mid = (a+c)//2
            # Unit total mass redistributed between children; equal log widths.
            contrast = .5*(kfine[:, a:mid].mean(axis=1)-kfine[:, mid:c].mean(axis=1))
            g = contrast/sigma[:, col]
            raw[j, col] = np.dot(g, residual[:, col])**2/max(np.dot(g, g), 1e-300)
            orth = g-u@(u.T@g)
            energy = np.dot(orth, orth)
            info[j, col] = np.sqrt(energy)*mass[j]
            if energy > 1e-24 and info[j, col] >= cfg.min_information:
                score[j, col] = np.dot(orth, residual[:, col])**2/energy
    return {"conditional": score, "raw": raw, "information": info,
            "regional_masses": regional, "nuisance_ranks": rank}


def propose_actions(tree, scores, cfg):
    """Joint, private-column, and position proposals; merges are reversible."""
    eligible = [j for j, (a, b) in enumerate(tree.leaves) if b-a > 1]
    capacity = cfg.max_active-len(tree.leaves)
    actions = []
    if capacity > 0 and eligible:
        count = min(cfg.split_batch, capacity, len(eligible))
        criteria = [("joint_detail", scores["conditional"].sum(axis=1)),
                    ("private_detail", scores["conditional"].max(axis=1)),
                    ("position", scores["raw"].sum(axis=1)),
                    # Pure correlation can rank empty leaves whose signed detail
                    # direction is infeasible. Also test occupied broad regions.
                    ("occupied", scores["regional_masses"].max(axis=1)*tree.widths)]
        seen = set()
        for kind, criterion in criteria:
            indices = tuple(sorted(sorted(eligible, key=lambda j: (-criterion[j], j))[:count]))
            if indices in seen or not np.any(criterion[list(indices)] > 0):
                continue
            seen.add(indices)
            actions.append({"kind": kind, "indices": list(indices), "tree": tree.split(indices)})
    if cfg.allow_merge and len(tree.leaves) > cfg.initial_active:
        pairs = tree.mergeable()
        if pairs:
            regional = scores["regional_masses"]
            j = min(pairs, key=lambda i: float(np.linalg.norm(regional[i]-regional[i+1])))
            actions.append({"kind": "merge", "indices": [j], "tree": tree.merge(j)})
    for action in actions:
        idx = action["indices"]
        action["features"] = [
            len(tree.leaves)/cfg.fine_bins, len(action["tree"].leaves)/cfg.fine_bins,
            float(action["kind"] == "merge"), float(action["kind"] in ("position", "occupied")),
            float(np.log1p(scores["conditional"][idx].sum())),
            float(np.log1p(scores["conditional"][idx].max())),
            float(np.log1p(scores["raw"][idx].sum())),
            float(np.log1p(scores["information"][idx].max())),
            float(scores["regional_masses"][idx].sum()/max(scores["regional_masses"].sum(), 1e-300)),
            float(np.mean(tree.widths[idx])/np.sum(tree.widths)),
        ]
    return actions


def solve_adaptive(problem: DecayProblem, config=None, *, policy=None, observer=None):
    """Search nested feasible spaces at fixed regularization, then select lambda.

    Training objective descent accepts splits; objective-equivalent merges
    remove redundant leaves. Reserved rows select between completed paths,
    never gate intermediate numerical refinement. Final refit uses all rows.
    A policy ranks candidates; the same deterministic verifier checks them.
    """
    from .solver import RAIResult
    if not isinstance(problem, DecayProblem):
        raise TypeError("problem must be a DecayProblem")
    cfg = config or AdaptiveConfig()
    if not isinstance(cfg, AdaptiveConfig):
        raise TypeError("config must be AdaptiveConfig")
    start = time.perf_counter()
    order = np.argsort(problem.b, kind="stable")
    b = problem.b[order]; ry = problem.y[order]; rs = problem.sigma[order]
    val = np.zeros(len(b), bool)
    if len(np.unique(b)) >= 12:
        unique = np.unique(b)
        val = np.isin(b, unique[2:-1:cfg.validation_stride])
    train = ~val
    scale = np.maximum(np.max(np.abs(ry[train]), axis=0), 8*np.median(rs[train], axis=0))
    y, sigma = ry/scale, rs/scale
    initial_tree = ResolutionTree.uniform(problem.rate_bounds, cfg.fine_bins, cfg.initial_active)
    k = bin_kernel(b, initial_tree.edges)
    history = []; fit_count = 0; fallbacks = 0; paths = []
    penalties = cfg.penalties if val.any() else (max(cfg.penalties),)
    def heldout(xx):
        return float(np.mean(((k[val]@xx-y[val])/sigma[val])**2)) if val.any() else None
    def fit(t, lam, initial):
        nonlocal fit_count
        fit_count += 1
        return fit_tree(k[train], y[train], sigma[train], t, lam, cfg, initial)
    for path_id, lam in enumerate(penalties):
        tree = initial_tree
        x, diag = fit(tree, lam, None)
        history.append({"path": path_id, "penalty": lam, "round": 0, "kind": "initialize",
            "accepted": bool(diag["success"]), "active_bins": len(tree.leaves),
            "leaves": [list(v) for v in tree.leaves], "optimizer": diag,
            "validation_loss": heldout(x)})
        visited = {tree.leaves}; stop = "max_rounds"
        if not diag["success"]:
            stop = "initial_optimizer_failed"
        for round_id in range(1, cfg.max_rounds+1 if diag["success"] else 1):
            scores = candidate_scores(b[train], k[train], y[train], sigma[train], tree, x, cfg)
            actions = [a for a in propose_actions(tree, scores, cfg) if a["tree"].leaves not in visited]
            if not actions:
                stop = "no_new_candidates"; break
            for a in actions:
                a["features"] += [float(np.log1p(lam)), float(np.log1p(y.shape[0])),
                    float(np.log1p(y.shape[1])), float(np.log1p(diag["objective"]))]
            priority = list(range(len(actions))); proposed = False
            if policy is not None:
                try:
                    ranking = np.asarray(policy.rank(np.array([a["features"] for a in actions])))
                    if ranking.shape != (len(actions),) or ranking.dtype.kind not in 'iu' or set(ranking.tolist()) != set(priority):
                        raise ValueError("Invalid policy permutation")
                    priority = ranking.tolist(); proposed = True
                except (ValueError, RuntimeError, TypeError, FloatingPointError):
                    fallbacks += 1
            evaluated = []
            for pos, index in enumerate(priority):
                action = actions[index]; t = action["tree"]
                new_x, new_diag = fit(t, lam, x)
                visited.add(t.leaves)
                gain = diag["objective"]-new_diag["objective"]
                # Compare common global objective: inactive Haar details are zero.
                # Require decrease beyond both reported inner optimization gaps.
                uncertainty = diag["duality_gap"]+new_diag["duality_gap"]
                threshold = max(cfg.descent_tolerance*(1+diag["objective"]), uncertainty)
                eligible = new_diag["success"] and gain > threshold
                if action["kind"] == "merge":
                    eligible = new_diag["success"] and gain >= -1e-10*(1+diag["objective"])
                record = {"path": path_id, "penalty": lam, "round": round_id,
                    "kind": action["kind"], "indices": action["indices"], "features": action["features"],
                    "active_bins": len(t.leaves), "leaves": [list(v) for v in t.leaves],
                    "objective_gain": gain, "required_gain": threshold,
                    # Independent acquisition utility for DOSY policy training.
                    # These are the controller's internal validation rows,
                    # never the external benchmark test rows.
                    "validation_gain": heldout(x)-heldout(new_x) if val.any() else None,
                    "eligible": bool(eligible), "accepted": False,
                    "optimizer": new_diag, "policy_ordered": proposed,
                    "nuisance_ranks": scores["nuisance_ranks"]}
                history.append(record)
                if observer is not None:
                    observer(dict(record))
                evaluated.append((new_x, new_diag, t, record))
                if proposed and pos+1 >= cfg.policy_shortlist and any(e[3]["eligible"] for e in evaluated):
                    break
                if proposed and pos+1 == cfg.policy_shortlist:
                    fallbacks += 1
            acceptable = [e for e in evaluated if e[3]["eligible"]]
            if not acceptable:
                stop = "no_verified_descent"; break
            # Prefer descent, then fewer leaves for numerical ties.
            x, diag, tree, chosen = min(acceptable, key=lambda e: (e[1]["objective"], len(e[2].leaves)))
            chosen["accepted"] = True
        paths.append({"id": path_id, "tree": tree, "masses": x, "optimizer": diag,
                      "loss": heldout(x), "stop": stop, "penalty": lam})
    good = [p for p in paths if p["optimizer"]["success"]]
    if good and val.any():
        minimum = min(p["loss"] for p in good)
        near = [p for p in good if p["loss"] <= minimum+cfg.selection_slack]
        chosen_path = max(near, key=lambda p: p["penalty"])
    elif good:
        chosen_path = good[0]
    else:
        chosen_path = min(paths, key=lambda p: p["optimizer"]["relative_gap"])
    tree = chosen_path["tree"]
    final_x, final_diag = fit_tree(k, y, sigma, tree, chosen_path["penalty"], cfg, chosen_path["masses"])
    fit_count += 1
    masses = final_x*scale
    reconstruction = bin_kernel(problem.b, tree.edges)@masses
    residual = problem.y-reconstruction
    selected_records = [h for h in history if h["path"] == chosen_path["id"]]
    diagnostics = {"representation": "adaptive", "mode": "single" if y.shape[1] == 1 else "dosy",
        "optimizer": final_diag, "selection_optimizer": chosen_path["optimizer"],
        "fine_bins": cfg.fine_bins, "active_bins": len(tree.leaves),
        "active_leaves": [list(v) for v in tree.leaves],
        "rate_grid_role": "integrated physical operator; conservative active hierarchy",
        "accepted_splits": sum(h["accepted"] and h["kind"] not in ("merge", "initialize") for h in selected_records),
        "accepted_merges": sum(h["accepted"] and h["kind"] == "merge" for h in selected_records),
        "refinement_stop": chosen_path["stop"], "selected_penalty": chosen_path["penalty"],
        "selected_path": chosen_path["id"], "selection_available": bool(val.any()),
        "selection_rows_original": order[val].tolist(), "training_scale": scale.tolist(),
        "selection_loss": chosen_path["loss"], "fixed_tree_fits": fit_count,
        "path_summaries": [{"path": p["id"], "penalty": p["penalty"], "validation_loss": p["loss"],
            "active_bins": len(p["tree"].leaves), "stop": p["stop"], "optimizer": p["optimizer"]} for p in paths],
        "policy": "learned" if policy is not None else "deterministic", "policy_fallbacks": fallbacks,
        "seconds": time.perf_counter()-start, "weighted_rmse": float(np.sqrt(np.mean((residual/problem.sigma)**2))),
        "statistical_coverage_claimed": False, "continuous_identifiability_certified": False,
        "full_grid_optimality_certified": False,
        "noise_model": "known diagonal standard deviations", "graph_edges": 0}
    return RAIResult(tree.edges, masses, reconstruction, residual,
        np.zeros((y.shape[1], y.shape[1])), history, diagnostics,
        {"representation": "adaptive", **asdict(cfg)})
