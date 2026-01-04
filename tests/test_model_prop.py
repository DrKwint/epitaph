"""
Integration tests for Epinet model propagation.
Verifies that the Epinet model (Base + Learnable + Prior) can be lifted 
to work with ConstrainedZonotopes using lift_epinet_propagation.
"""
import unittest
import jax.numpy as jnp
from flax import nnx
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from epitaph.structures import ConstrainedZonotope
from epitaph.models import Epinet
from epitaph.model_utils import lift_epinet_propagation
from epitaph import ops
from hypothesis import given, strategies as st, settings
import jax


class TestModelProp(unittest.TestCase):
    def test_point_propagation(self):
        """
        Verify that propagating a point-zonotope (generators=0) matches standard model output.
        """
        n_features = 2  # s=1, a=1
        z_dim = 2
        n_out = 1

        # Init model
        model = Epinet(n_features, n_out, z_dim=z_dim, rngs=nnx.Rngs(params=0))

        # Input point x=[s,a], z=[z1,z2]
        # Total dims = 4
        # x_val = [0.5, -0.2]
        # z_val = [0.1, 0.0]
        full_input = jnp.array([[0.5, -0.2, 0.1, 0.0]])

        # Standard Output
        # model(x, z)
        x_in = full_input[:, :n_features]
        z_in = full_input[:, n_features:]
        y_std = model(x_in, z_in)

        # Lifted Output
        # Create CZ with 0 generators
        cz_in = ConstrainedZonotope.create(
            full_input, jnp.zeros((1, 0, n_features + z_dim))
        )

        cz_out = lift_epinet_propagation(model, cz_in)

        # Check center matches y_std
        self.assertTrue(jnp.allclose(cz_out.center, y_std, atol=1e-5))

        # Check generators are 0 (should remain 0 for stable network logic)
        # However, new architecture sums 3 branches.
        # Stability logic applies to each branch.
        # If point input -> all branches stable -> no new error generators.
        lb, ub = ops.compute_bounds(cz_out)
        self.assertTrue(jnp.allclose(lb, ub, atol=1e-5))
        self.assertTrue(jnp.allclose(lb, y_std, atol=1e-5))

    def test_volume_propagation(self):
        """
        Verify soundness for a small box input.
        """
        n_features = 2
        z_dim = 2
        n_out = 1
        model = Epinet(n_features, n_out, z_dim=z_dim, rngs=nnx.Rngs(params=1))

        center = jnp.zeros((1, 4))
        # Generator: epsilon on first dim (state)
        generators = jnp.zeros((1, 1, 4))
        generators = generators.at[0, 0, 0].set(0.1)

        cz_in = ConstrainedZonotope.create(center, generators)

        # Run lifted
        cz_out = lift_epinet_propagation(model, cz_in)

        # Get bounds
        lb_out, ub_out = ops.compute_bounds(cz_out)

        # Run standard on extreme points
        # x1 = [0.1, 0, 0, 0] -> s,a=[0.1, 0], z=[0,0]
        x1_in = jnp.array([[0.1, 0.0]])
        z1_in = jnp.array([[0.0, 0.0]])
        y1 = model(x1_in, z1_in)

        x2_in = jnp.array([[-0.1, 0.0]])
        z2_in = jnp.array([[0.0, 0.0]])
        y2 = model(x2_in, z2_in)

        # Check soundness
        self.assertTrue(jnp.all(y1 >= lb_out - 1e-5))
        self.assertTrue(jnp.all(y1 <= ub_out + 1e-5))
        self.assertTrue(jnp.all(y2 >= lb_out - 1e-5))
        self.assertTrue(jnp.all(y2 <= ub_out + 1e-5))

    @settings(deadline=None, max_examples=5)
    @given(
        n_features=st.integers(min_value=2, max_value=3),
        z_dim=st.integers(min_value=1, max_value=2),
        n_gen=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_epinet_propagation_soundness(self, n_features, z_dim, n_gen, seed):
        """
        Property: Epinet lifted propagation is sound.
        Samples from input zonotope must map to points within output zonotope bounds.
        """
        key = jax.random.PRNGKey(seed)
        
        # Create Epinet model
        n_out = 1
        key, model_key = jax.random.split(key)
        model = Epinet(n_features, n_out, z_dim=z_dim, rngs=nnx.Rngs(params=int(model_key[0])))
        
        # Create random input zonotope over [x, z]
        total_dim = n_features + z_dim
        key, k1, k2 = jax.random.split(key, 3)
        center = jax.random.normal(k1, (1, total_dim)) * 0.5  # Scale down for stability
        generators = jax.random.normal(k2, (1, n_gen, total_dim)) * 0.3
        
        cz_in = ConstrainedZonotope.create(center, generators)
        
        # Propagate zonotope
        cz_out = lift_epinet_propagation(model, cz_in)
        
        # Sample points from input zonotope
        n_samples = 10
        key, k_samples = jax.random.split(key)
        xi = jax.random.uniform(k_samples, (n_samples, n_gen), minval=-1.0, maxval=1.0)
        
        gens_matrix = generators[0]
        full_pts = center + xi @ gens_matrix
        
        # Split into x and z
        x_pts = full_pts[:, :n_features]
        z_pts = full_pts[:, n_features:]
        
        # Propagate samples through model
        y_pts = jax.vmap(lambda x, z: model(x[None, :], z[None, :]))(x_pts, z_pts)
        y_pts = y_pts.squeeze()
        
        # Check soundness
        lb, ub = ops.compute_bounds(cz_out)
        in_bounds = jnp.all((y_pts >= lb - 1e-3) & (y_pts <= ub + 1e-3), axis=-1)
        
        if not jnp.all(in_bounds):
            # Debug output for failures
            print(f"\nFailed: n_features={n_features}, z_dim={z_dim}, n_gen={n_gen}, seed={seed}")
            print(f"Violations: {jnp.sum(~in_bounds)}/{n_samples}")
        
        self.assertTrue(jnp.all(in_bounds))

    @settings(deadline=None, max_examples=5)
    @given(
        n_features=st.integers(min_value=2, max_value=3),
        z_dim=st.integers(min_value=1, max_value=2),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_epinet_point_consistency(self, n_features, z_dim, seed):
        """
        Property: For point inputs (zero generators), lifted propagation matches direct model call.
        """
        key = jax.random.PRNGKey(seed)
        
        # Create Epinet model
        n_out = 1
        key, model_key = jax.random.split(key)
        model = Epinet(n_features, n_out, z_dim=z_dim, rngs=nnx.Rngs(params=int(model_key[0])))
        
        # Create random point input
        total_dim = n_features + z_dim
        key, k_point = jax.random.split(key)
        point = jax.random.normal(k_point, (1, total_dim)) * 0.5
        
        # Create point zonotope (zero generators)
        cz_in = ConstrainedZonotope.create(point, jnp.zeros((1, 0, total_dim)))
        
        # Lifted propagation
        cz_out = lift_epinet_propagation(model, cz_in)
        
        # Direct model call
        x_in = point[:, :n_features]
        z_in = point[:, n_features:]
        y_direct = model(x_in, z_in)
        
        # Check centers match
        self.assertTrue(jnp.allclose(cz_out.center, y_direct, atol=1e-4))
        
        # Check output is also a point (tight bounds)
        lb, ub = ops.compute_bounds(cz_out)
        self.assertTrue(jnp.allclose(lb, ub, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
