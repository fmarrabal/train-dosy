"""RAI-ILT: research software, not a validated scientific release."""
from .model import DecayProblem, bin_kernel, refine_bins
from .solver import RAIConfig, RAIResult, solve
from .adaptive import AdaptiveConfig, solve_adaptive
from .net import RAINetPolicy, solve_net
from .moment_solver import MomentConfig
from .moments import MomentResult, solve_moments, solve_moment_net
from .moment_net import MomentPrior

__version__ = "0.6.0"
__all__ = ["DecayProblem", "bin_kernel", "refine_bins", "RAIConfig", "RAIResult", "solve",
           "AdaptiveConfig", "solve_adaptive", "RAINetPolicy", "solve_net",
           "MomentConfig", "MomentResult", "MomentPrior", "solve_moments", "solve_moment_net"]
