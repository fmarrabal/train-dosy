"""Conservative dyadic representation on an actual fine log-rate grid."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResolutionTree:
    edges: np.ndarray
    leaves: tuple[tuple[int, int], ...]

    def __post_init__(self):
        edges = np.array(self.edges, dtype=float, copy=True)
        n = len(edges)-1
        if n < 2 or n & (n-1) or not np.all(np.isfinite(edges)) or np.any(edges <= 0):
            raise ValueError("Need a positive dyadic fine grid")
        h = np.diff(np.log(edges))
        if np.any(h <= 0) or not np.allclose(h, h[0], rtol=1e-10):
            raise ValueError("The tree requires equally spaced log edges")
        leaves = tuple((int(a), int(b)) for a, b in self.leaves)
        end = 0
        for a, b in leaves:
            width = b-a
            if a != end or width < 1 or width & (width-1) or a % width:
                raise ValueError("Leaves must be an ordered dyadic partition")
            end = b
        if end != n:
            raise ValueError("Leaves do not cover the grid")
        edges.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "leaves", leaves)

    @classmethod
    def uniform(cls, bounds, bins=256, active=8):
        if active < 1 or active & (active-1) or active > bins or bins % active:
            raise ValueError("Initial active bins must be a power of two dividing the grid")
        width = bins//active
        return cls(np.geomspace(*bounds, bins+1), tuple((a, a+width) for a in range(0, bins, width)))

    @property
    def widths(self):
        z = np.log(self.edges)
        return np.array([z[b]-z[a] for a, b in self.leaves])

    def prolongation(self):
        """Fine masses = P @ u; u = leaf mass / sqrt(log width)."""
        p = np.zeros((len(self.edges)-1, len(self.leaves)))
        h = np.diff(np.log(self.edges))
        for j, (a, b) in enumerate(self.leaves):
            p[a:b, j] = h[a:b]/np.sqrt(self.widths[j])
        return p

    def coefficients(self):
        """Orthonormal density Haar coefficients Q @ u; root is first."""
        n = len(self.edges)-1
        rows = [np.sqrt(self.widths/np.sum(self.widths))]
        nodes = [None]
        leaf_set = set(self.leaves)
        def visit(a, b):
            if (a, b) in leaf_set:
                return
            mid = (a+b)//2
            width = np.log(self.edges[b]/self.edges[a])
            row = np.zeros(len(self.leaves))
            for j, (lo, hi) in enumerate(self.leaves):
                if a <= lo and hi <= b:
                    row[j] = np.sqrt(self.widths[j]/width)*(1 if hi <= mid else -1)
            rows.append(row); nodes.append((a, b))
            visit(a, mid); visit(mid, b)
        visit(0, n)
        return np.array(rows), nodes

    def split(self, indices):
        selected = set(indices)
        if not selected.issubset(range(len(self.leaves))):
            raise ValueError("Unknown split leaf")
        leaves = []
        for j, (a, b) in enumerate(self.leaves):
            if j in selected and b-a > 1:
                mid = (a+b)//2
                leaves.extend(((a, mid), (mid, b)))
            else:
                leaves.append((a, b))
        return ResolutionTree(self.edges, tuple(leaves))

    def mergeable(self):
        out = []
        for j in range(len(self.leaves)-1):
            a, mid = self.leaves[j]; lo, b = self.leaves[j+1]
            if mid == lo and mid-a == b-lo and a % (b-a) == 0:
                out.append(j)
        return out

    def merge(self, left_index):
        if left_index not in self.mergeable():
            raise ValueError("Leaves are not siblings")
        j = left_index
        leaves = self.leaves[:j]+((self.leaves[j][0], self.leaves[j+1][1]),)+self.leaves[j+2:]
        return ResolutionTree(self.edges, leaves)

    def restrict(self, masses):
        """Conservative aggregation, also used to initialize another partition."""
        return np.array([masses[a:b].sum(axis=0)/np.sqrt(w)
                         for (a, b), w in zip(self.leaves, self.widths)])
