import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional


@struct.dataclass
class ConstrainedZonotope:
    """
    Represents a set of Constrained Zonotopes with support for additional linear inequalities.

    Z = {c + G * xi | ||xi||_inf <= 1, A_eq * xi = b_eq, A_in * xi <= b_in}

    Shapes:
        - center: (batch, n_dim)
        - generators: (batch, n_gen, n_dim)
        - constraints_eq_A: (batch, n_cons_eq, n_gen)
        - constraints_eq_b: (batch, n_cons_eq)
        - constraints_ineq_A: (batch, n_cons_in, n_gen)
        - constraints_ineq_b: (batch, n_cons_in)
    """

    center: jax.Array
    generators: jax.Array
    constraints_eq_A: jax.Array
    constraints_eq_b: jax.Array
    constraints_ineq_A: jax.Array
    constraints_ineq_b: jax.Array

    @property
    def n_dim(self) -> int:
        return self.center.shape[-1]

    @property
    def n_gen(self) -> int:
        return self.generators.shape[-2]

    @classmethod
    def create(
        cls,
        center: jax.Array,
        generators: jax.Array,
        constraints_eq_A: Optional[jax.Array] = None,
        constraints_eq_b: Optional[jax.Array] = None,
        constraints_ineq_A: Optional[jax.Array] = None,
        constraints_ineq_b: Optional[jax.Array] = None,
    ):
        """
        Helper constructor.
        """
        center = jnp.asarray(center)
        generators = jnp.asarray(generators)

        batch = center.shape[0]
        n_gen = generators.shape[1]

        if constraints_eq_A is None:
            constraints_eq_A = jnp.zeros((batch, 0, n_gen))
        if constraints_eq_b is None:
            constraints_eq_b = jnp.zeros((batch, 0))

        if constraints_ineq_A is None:
            constraints_ineq_A = jnp.zeros((batch, 0, n_gen))
        if constraints_ineq_b is None:
            constraints_ineq_b = jnp.zeros((batch, 0))

        return cls(
            center=center,
            generators=generators,
            constraints_eq_A=constraints_eq_A,
            constraints_eq_b=constraints_eq_b,
            constraints_ineq_A=constraints_ineq_A,
            constraints_ineq_b=constraints_ineq_b,
        )
