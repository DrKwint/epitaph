
"""
Tests for ConstrainedZonotope structures and propagation operations.
Uses Hypothesis for property-based testing to verify soundness across random dimensions and inputs.
"""
import unittest
import jax
import jax.numpy as jnp
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from epitaph.structures import ConstrainedZonotope
from epitaph import ops
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

class TestStructures(unittest.TestCase):
    
    @settings(deadline=None, max_examples=20)
    @given(
        n_dim=st.integers(min_value=2, max_value=5),
        n_gen=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_propagation_soundness_hypothesis(self, n_dim, n_gen, seed):
        """
        Property: For any random small MLP (Linear->ReLU->Linear) and random Zonotope,
        samples from the input Zonotope propagated through the network must lie 
        within the computed bounds of the output Zonotope.
        """
        import traceback
        try:
            key = jax.random.PRNGKey(seed)
            
            # 1. Define Zonotope
            key, k1, k2 = jax.random.split(key, 3)
            center = jax.random.normal(k1, (1, n_dim))
            generators = jax.random.normal(k2, (1, n_gen, n_dim))
            
            cz = ConstrainedZonotope.create(center, generators)
            
            # 2. Define Layers
            # Layer 1: Linear
            key, kw1, kb1 = jax.random.split(key, 3)
            W1 = jax.random.normal(kw1, (n_dim, n_dim))
            b1 = jax.random.normal(kb1, (1, n_dim))
            
            # Layer 2: ReLU
            
            # Layer 3: Linear
            key, kw2, kb2 = jax.random.split(key, 3)
            W2 = jax.random.normal(kw2, (n_dim, n_dim)) 
            b2 = jax.random.normal(kb2, (1, n_dim))
            
            # 3. Propagate Zonotope
            cz_1 = ops.propagate_linear(cz, W1, b1)
            cz_2 = ops.propagate_relu(cz_1)
            cz_3 = ops.propagate_linear(cz_2, W2, b2)
            
            # 4. Propagate Samples
            n_samples = 20
            key, k_samples = jax.random.split(key)
            
            # Sample xi from [-1, 1]
            xi = jax.random.uniform(
                k_samples, (n_samples, n_gen), minval=-1.0, maxval=1.0
            )
            
            gens_matrix = generators[0]  # (n_gen, n_dim)
            x0_pts = center + xi @ gens_matrix  # (n_samples, n_dim)
            
            # Apply L1 (W1 is (n_dim, n_dim) and is applied as x @ W1 + b1, following ops.propagate_linear convention)
            x1_pts = x0_pts @ W1 + b1
            
            # Apply ReLU
            x2_pts = jax.nn.relu(x1_pts)
            
            # Apply L3 (W2 is (n_dim, n_dim) and is applied as x @ W2 + b2)
            x3_pts = x2_pts @ W2 + b2
            
            # 5. Check Containment
            lb, ub = ops.compute_bounds(cz_3)
            
            in_bounds_lb = jnp.all(x3_pts >= lb - 1e-3, axis=1) # Tolerance for float precision
            in_bounds_ub = jnp.all(x3_pts <= ub + 1e-3, axis=1)
            
            all_contained = jnp.all(in_bounds_lb & in_bounds_ub)
            
            if not all_contained:
                # Debug info (only printed on failure by Hypothesis)
                msg = f"Seed: {seed}, Dim: {n_dim}, Gen: {n_gen}\n"
                msg += f"LB violation max: {jnp.max(x3_pts[~in_bounds_lb] - lb) if jnp.any(~in_bounds_lb) else 0}\n"
                msg += f"UB violation max: {jnp.max(x3_pts[~in_bounds_ub] - ub) if jnp.any(~in_bounds_ub) else 0}\n"
                print(msg)
                with open("failure_dump.txt", "w") as f:
                    f.write(msg)
            
            self.assertTrue(all_contained)
            
        except Exception as e:
            msg = f"CRASH: Seed: {seed}, Dim: {n_dim}, Gen: {n_gen}\n"
            msg += traceback.format_exc()
            with open("failure_dump.txt", "w") as f:
                f.write(msg)
            raise e

if __name__ == "__main__":
    unittest.main()
