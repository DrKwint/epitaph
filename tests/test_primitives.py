"""
Unit tests for basic ConstrainedZonotope primitives and operations in ops.py.
Verifies linear propagation, bounds computation, and basic ReLU soundness.
"""
import unittest
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from epitaph.structures import ConstrainedZonotope
from epitaph import ops
from hypothesis import given, strategies as st, settings
import jax


class TestPrimitives(unittest.TestCase):
    def setUp(self):
        # Create a simple 2D zonotope
        # Center at [1, 1]
        # Generators: [[1, 0], [0, 1]] (Box of size 2x2 centered at 1,1)
        self.center = jnp.array([[1.0, 1.0]])
        self.generators = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        self.cz = ConstrainedZonotope.create(self.center, self.generators)

    def test_linear(self):
        # x' = 2x
        W = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        cz_new = ops.propagate_linear(self.cz, W)

        # Check center
        expected_center = jnp.array([[2.0, 2.0]])
        self.assertTrue(jnp.allclose(cz_new.center, expected_center))

        # Check generators
        expected_gens = jnp.array([[[2.0, 0.0], [0.0, 2.0]]])
        self.assertTrue(jnp.allclose(cz_new.generators, expected_gens))

    def test_bounds(self):
        lb, ub = ops.compute_bounds(self.cz)
        # Center [1, 1], gens [[1,0], [0,1]] -> sum abs = [1, 1]
        # lb = [0, 0], ub = [2, 2]
        self.assertTrue(jnp.allclose(lb, jnp.array([[0.0, 0.0]])))
        self.assertTrue(jnp.allclose(ub, jnp.array([[2.0, 2.0]])))

    def test_relu_soundness(self):
        # Create a CZ crossing zero
        # Center [0.5, -0.5]
        # Gens [[1, 0], [0, 1]]
        # x range [-0.5, 1.5], y range [-1.5, 0.5]

        center = jnp.array([[0.5, -0.5]])
        gens = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
        cz = ConstrainedZonotope.create(center, gens)

        cz_relu = ops.propagate_relu(cz)

        # Sample a point inside original CZ
        # xi = [0.5, 0.5]
        # x = c + G xi = [0.5 + 0.5, -0.5 + 0.5] = [1.0, 0.0]
        # relu(x) = [1.0, 0.0]

        # Check if [1.0, 0.0] is in cz_relu
        # This requires solving an LP or checking bounds.
        # Let's simple check bounds over-approximation soundness
        lb, ub = ops.compute_bounds(cz_relu)

        # The Lambda-Zonotope relaxation is a symmetric shape around a center.
        # It does not guarantee lb >= 0 for the unconstrained set (it spills over).
        # We only check that the true point is included (soundness).
        # And that the bounds aren't trivially loose (like [-inf, inf]).

        # Check containment of the transformed point [1.0, 0.0]
        # Since 'in' check is hard without an LP solver, we check bounds containment.
        # The point [1.0, 0.0] must be within [lb, ub].

        target_point = jnp.array([[1.0, 0.0]])
        self.assertTrue(jnp.all(target_point >= lb - 1e-5))
        self.assertTrue(jnp.all(target_point <= ub + 1e-5))

    def test_constraints_slice(self):
        # Slice x[0] <= 0.5
        # Original x range [0, 2].
        # H = [1, 0], d = [0.5]

        H = jnp.array([[1.0, 0.0]])
        d = jnp.array([0.5])

        cz_sliced = ops.apply_constraints(self.cz, H, d)

        # Check if constraints_ineq were added
        # Should correspond to: HG xi <= d - Hc
        # HG = [[1, 0]] @ [[1, 0], [0, 1]].T = [[1, 0]]
        # d - Hc = 0.5 - 1.0 = -0.5
        # 1*xi_1 <= -0.5 -> xi_1 <= -0.5

        # Note: xi is bound by [-1, 1].
        # So effective xi_1 range is [-1, -0.5].
        # x = 1 + xi_1 -> [0, 0.5].

        # Verify the constraint matrix values
        self.assertTrue(jnp.allclose(cz_sliced.constraints_ineq_A[:, :, 0], 1.0))
        self.assertTrue(jnp.allclose(cz_sliced.constraints_ineq_b, -0.5))

    @settings(deadline=None, max_examples=10)
    @given(
        n_dim=st.integers(min_value=2, max_value=4),
        n_gen=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_linear_property(self, n_dim, n_gen, seed):
        """
        Property: Linear propagation W(c + G*xi) = Wc + WG*xi.
        Verify that applying a linear transform to samples matches the zonotope transform.
        """
        key = jax.random.PRNGKey(seed)
        
        # Create random zonotope
        key, k1, k2 = jax.random.split(key, 3)
        center = jax.random.normal(k1, (1, n_dim))
        generators = jax.random.normal(k2, (1, n_gen, n_dim))
        cz = ConstrainedZonotope.create(center, generators)
        
        # Random linear transform
        key, kw, kb = jax.random.split(key, 3)
        W = jax.random.normal(kw, (n_dim, n_dim))  # Keep same dim for simplicity
        b = jax.random.normal(kb, (n_dim,))
        
        # Propagate zonotope
        cz_out = ops.propagate_linear(cz, W, b)
        
        # Sample points and propagate
        n_samples = 10
        key, k_samples = jax.random.split(key)
        xi = jax.random.uniform(k_samples, (n_samples, n_gen), minval=-1.0, maxval=1.0)
        
        gens_matrix = generators[0]
        x_pts = center + xi @ gens_matrix
        y_pts = x_pts @ W.T + b
        
        # Check all points are in zonotope bounds
        lb, ub = ops.compute_bounds(cz_out)
        in_bounds = jnp.all((y_pts >= lb - 1e-4) & (y_pts <= ub + 1e-4), axis=1)
        
        self.assertTrue(jnp.all(in_bounds))

    @settings(deadline=None, max_examples=10)
    @given(
        n_dim=st.integers(min_value=2, max_value=4),
        n_gen=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_bounds_property(self, n_dim, n_gen, seed):
        """
        Property: Bounds are sound - all samples must be within computed bounds.
        """
        key = jax.random.PRNGKey(seed)
        
        # Create random zonotope
        key, k1, k2 = jax.random.split(key, 3)
        center = jax.random.normal(k1, (1, n_dim))
        generators = jax.random.normal(k2, (1, n_gen, n_dim))
        cz = ConstrainedZonotope.create(center, generators)
        
        # Compute bounds
        lb, ub = ops.compute_bounds(cz)
        
        # Sample many points
        n_samples = 50
        key, k_samples = jax.random.split(key)
        xi = jax.random.uniform(k_samples, (n_samples, n_gen), minval=-1.0, maxval=1.0)
        
        gens_matrix = generators[0]
        x_pts = center + xi @ gens_matrix
        
        # Check all points are within bounds
        in_bounds = jnp.all((x_pts >= lb - 1e-5) & (x_pts <= ub + 1e-5), axis=1)
        
        self.assertTrue(jnp.all(in_bounds))


if __name__ == "__main__":
    unittest.main()
