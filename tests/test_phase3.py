import jax
import jax.numpy as jnp
from epitaph.structures import ConstrainedZonotope
from epitaph.models import Epinet
from epitaph import ops
from flax import nnx

def test_epinet_propagate_set():
    print("Testing Epinet.propagate_set...")
    z_dim = 2
    in_features = 3
    out_features = 1
    batch_size = 2
    
    model = Epinet(in_features, out_features, z_dim, rngs=nnx.Rngs(0))
    
    # Input CZ: x in [0, 1]^3
    cz_input = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, in_features)),
        generators=jnp.eye(in_features).reshape(1, in_features, in_features).repeat(batch_size, axis=0)
    )
    
    # Z CZ: z in [-1, 1]^2
    cz_z = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, z_dim)),
        generators=jnp.eye(z_dim).reshape(1, z_dim, z_dim).repeat(batch_size, axis=0)
    )
    
    cz_out = model.propagate_set(cz_input, cz_z)
    
    assert isinstance(cz_out, ConstrainedZonotope)
    assert cz_out.center.shape == (batch_size, out_features)
    print("[OK] Epinet.propagate_set successful.")

def test_volume_monotonicity():
    print("Testing Volume Monotonicity...")
    batch_size = 1
    n_gen = 4
    n_dim = 2
    
    # Start with unconstrained unit hypercube in xi space
    cz = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, n_dim)),
        generators=jnp.zeros((batch_size, n_gen, n_dim)) # Generators don't matter for latent volume, but we need n_gen
    )
    # The current create() doesn't store n_gen if generators are zero/empty? 
    # Actually cz.n_gen is generators.shape[-2].
    
    # Let's use real generators
    cz = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, n_dim)),
        generators=jnp.eye(n_gen, n_dim).reshape(1, n_gen, n_dim)
    )
    
    vol_full = ops.compute_cz_volume_approx(cz)
    print(f"Full volume: {vol_full}")
    
    # Add a constraint that cuts the space: xi[0] <= 0
    # HG * xi <= d - Hc
    # G is (1, 4, 2). 
    # Let's say we want to constrain the first latent xi[0] <= 0
    # In my current implementation, constraints are applied in s-space: Hx <= d
    # Then it translates to (HG)xi <= d - Hc
    
    # Let's just manually add the constraint A_in, b_in
    A_in = jnp.zeros((batch_size, 1, n_gen))
    A_in = A_in.at[:, 0, 0].set(1.0) # xi[0]
    b_in = jnp.zeros((batch_size, 1)) # <= 0
    
    cz_constrained = cz.replace(constraints_ineq_A=A_in, constraints_ineq_b=b_in)
    vol_half = ops.compute_cz_volume_approx(cz_constrained)
    print(f"Half volume: {vol_half}")
    
    assert vol_half < vol_full
    
    # Add an impossible constraint: xi[0] <= -2
    b_impossible = jnp.array([[-2.0]])
    cz_impossible = cz.replace(constraints_ineq_A=A_in, constraints_ineq_b=b_impossible)
    vol_zero = ops.compute_cz_volume_approx(cz_impossible)
    print(f"Impossible volume: {vol_zero}")
    
    assert vol_zero < vol_half
    print("[OK] Volume monotonicity successful.")

def test_safety_probability():
    print("Testing Safety Probability...")
    batch_size = 1
    n_gen = 2
    n_dim = 2
    
    cz = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, n_dim)),
        generators=jnp.eye(n_gen, n_dim).reshape(1, n_gen, n_dim)
    )
    
    p_safe_initial = ops.calculate_safety_probability(cz)
    print(f"Initial P_safe: {p_safe_initial}")
    assert jnp.allclose(p_safe_initial, 1.0, atol=1e-3)
    
    # Add constraint that removes half the latent space
    A_in = jnp.array([[[1.0, 0.0]]])
    b_in = jnp.array([[0.0]])
    cz_half = cz.replace(constraints_ineq_A=A_in, constraints_ineq_b=b_in)
    p_safe_half = ops.calculate_safety_probability(cz_half)
    print(f"Half P_safe: {p_safe_half}")
    assert jnp.allclose(p_safe_half, 0.5, atol=0.1) # Soft approximation might not be exact
    
    print("[OK] Safety probability successful.")

def test_gaussian_safety():
    print("Testing Gaussian Safety Probability...")
    batch_size = 1
    n_z = 2
    n_dim = 1
    
    # State s = center + G_z * xi_z
    # Let's say s = xi_z[0]
    cz = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, n_dim)),
        generators=jnp.array([[[1.0], [0.0]]]) # G_z is (1, 2, 1)
    )
    
    # Constraint: s <= 0  ->  xi_z[0] <= 0
    # Since xi_z[0] ~ N(0, 1), P(xi_z[0] <= 0) = 0.5
    A_in = jnp.array([[[1.0, 0.0]]])
    b_in = jnp.array([[0.0]])
    cz_cons = cz.replace(constraints_ineq_A=A_in, constraints_ineq_b=b_in)
    
    p_safe = ops.calculate_gaussian_safety_prob(cz_cons, n_epistemic_dims=n_z)
    print(f"Gaussian P_safe (s <= 0): {p_safe}")
    assert jnp.allclose(p_safe, 0.5, atol=1e-3)
    
    # Constraint: s <= 1.96 -> P approx 0.975
    b_high = jnp.array([[1.96]])
    cz_cons_high = cz.replace(constraints_ineq_A=A_in, constraints_ineq_b=b_high)
    p_safe_high = ops.calculate_gaussian_safety_prob(cz_cons_high, n_epistemic_dims=n_z)
    print(f"Gaussian P_safe (s <= 1.96): {p_safe_high}")
    assert jnp.allclose(p_safe_high, 0.975, atol=1e-2)
    
    print("[OK] Gaussian safety successful.")

def test_trajectory_consistency():
    print("Testing Trajectory Temporal Consistency...")
    z_dim = 1
    in_features = 2 # [s, a]
    out_features = 1 # [s']
    batch_size = 1
    model = Epinet(in_features, out_features, z_dim, rngs=nnx.Rngs(0))
    
    # Initial state zonotope
    cz_s0 = ConstrainedZonotope.create(
        center=jnp.zeros((batch_size, 1)),
        generators=jnp.zeros((batch_size, 0, 1))
    )
    
    actions = jnp.zeros((5, batch_size, 1)) # 5 steps
    
    traj = model.propagate_trajectory(cz_s0, actions, n_z_gens=z_dim)
    
    assert len(traj) == 6 # s0 + 5 steps
    # Verify that the next states are still zonotopes
    for cz in traj:
        assert isinstance(cz, ConstrainedZonotope)
        
    print("[OK] Trajectory consistency successful.")

if __name__ == "__main__":
    test_epinet_propagate_set()
    test_volume_monotonicity()
    test_safety_probability()
    test_gaussian_safety()
    test_trajectory_consistency()
