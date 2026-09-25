"""Versioned interface to the frozen joint DOSY research solvers."""
__version__ = "0.1.0"
from .models import FitRequest
from .inversion import fit
__all__ = ["fit", "FitRequest", "__version__"]
