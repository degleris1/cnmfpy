"""Optimization methods for convolutive matrix factorization."""

from .gradient_descent import GradDescent, BlockDescent

# from .mult import fit_mult, mult_step
# from .chals import fit_chals

ALGORITHMS = {
    "gd": GradDescent,
    "bcd": BlockDescent,
}
