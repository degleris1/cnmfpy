"""Optimization methods for convolutive matrix factorization."""

from .gradient_descent import GradDescent, BlockDescent
from .mult import MultUpdate, StochasticMultUpdate
from .hals import SimpleHALSUpdate, HALSUpdate

# from .mult import fit_mult, mult_step
# from .chals import fit_chals

ALGORITHMS = {
    "gd": GradDescent,
    "bcd": BlockDescent,
    "mult": MultUpdate,
    "stochastic_mult": StochasticMultUpdate,
    "hals": HALSUpdate,
    "hals_simple": SimpleHALSUpdate,
}
