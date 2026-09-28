"""Versioned interface to the frozen joint DOSY research solvers."""
__version__ = "0.2.0"
from .models import FitRequest
from .inversion import fit
__all__ = ["fit", "FitRequest", "__version__"]
