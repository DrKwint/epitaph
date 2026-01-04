import jax
import jax.numpy as jnp
from epitaph.models import Epinet
from epitaph.safety import collect_iterative_safety_constraints
from epitaph import ops
from flax import nnx

def test_iterative_safety():
    print("Testing iterative safety constraint collection...")
    z_dim = 2
    s_dim = 2
    a_dim = 1
    
    # Model: in = 3 ([s, a]), out = 2, z = 2
    model = Epinet(s_dim + a_dim, s_dim, z_dim, rngs=nnx.Rngs(0))
    
    obs = jnp.array([0.0, 0.0])
    action_seq = jnp.array([
        [0.1],
        [0.2],
        [0.3]
    ])
    
    # Safe region: s[0] <= 0.5
    unsafe_A = jnp.array([[1.0, 0.0]])
    unsafe_b = jnp.array([0.5])
    
    final_cz = collect_iterative_safety_constraints(obs, action_seq, unsafe_A, unsafe_b, model)
    
    print(f"Final CZ center shape: {final_cz.center.shape}")
    print(f"Final CZ generators count: {final_cz.n_gen}")
    print(f"Final inequality constraints A shape: {final_cz.constraints_ineq_A.shape}")
    
    # Due to lift_epinet_propagation combining 3 branches, constraints may be duplicated.
    # We just verify that we have accumulated them.
    assert final_cz.constraints_ineq_A.shape[1] >= 3
    
    # Verify we can compute volume/safety
    p_safe_optimistic = ops.calculate_gaussian_safety_prob(final_cz, n_epistemic_dims=z_dim)
    print(f"Computed optimistic safe z probability: {p_safe_optimistic}")
    
    from epitaph.safety import calculate_pessimistic_gaussian_safety_prob
    p_safe_pessimistic = calculate_pessimistic_gaussian_safety_prob(final_cz, z_dim)
    print(f"Computed pessimistic safe z probability: {p_safe_pessimistic}")
    
    assert p_safe_pessimistic.shape == (1,)
    # Pessimistic should always be <= Optimistic
    assert p_safe_pessimistic[0] <= p_safe_optimistic[0] + 1e-6
    
    # 3. Monte Carlo Comparison
    from epitaph.safety import estimate_safety_mc
    p_safe_mc = estimate_safety_mc(obs, action_seq, unsafe_A, unsafe_b, model, n_samples=5000)
    print(f"Computed Monte Carlo safe z probability: {p_safe_mc}")
    
    # The MC estimate should ideally be between Pessimistic and Optimistic.
    # (Pessimistic accounts for non-linear error that MC explores via actual sampling)
    # Note: If the model is nearly linear, they should be very close.
    print(f"Gap (MC - Pessimistic): {p_safe_mc - p_safe_pessimistic[0]}")
    
    print("[OK] Iterative safety test (with MC) successful.")

if __name__ == "__main__":
    test_iterative_safety()
