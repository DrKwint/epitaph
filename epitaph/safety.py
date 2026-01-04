import jax
import jax.numpy as jnp
from typing import Optional, Union, TYPE_CHECKING
from epitaph.structures import ConstrainedZonotope
from epitaph import ops
from epitaph.model_utils import lift_epinet_propagation

if TYPE_CHECKING:
    from epitaph.models import Epinet

def collect_iterative_safety_constraints(
    obs: jax.Array, 
    action_seq: jax.Array, 
    unsafe_A: jax.Array, 
    unsafe_b: jax.Array, 
    model: "Epinet", 
    z_scale: float = 1.0
) -> ConstrainedZonotope:
    """
    Iteratively builds a ConstrainedZonotope that captures the safe set of z.
    
    Args:
        obs: Initial observation [State_Dim]
        action_seq: Sequence of actions [Horizon, Action_Dim]
        unsafe_A: [N_cons, State_Dim]
        unsafe_b: [N_cons]
        model: Epinet instance
        z_scale: Scaling for the initial z-box (radius of the hypercube).
        
    Returns:
        The final ConstrainedZonotope containing all accumulated constraints.
    """
    z_dim = model.z_dim
    s_dim = obs.shape[-1]
    action_dim = action_seq.shape[-1]
    horizon = action_seq.shape[0]
    
    # Normalize inputs
    if obs.ndim == 1:
        obs = obs[None, :]
    batch_size = obs.shape[0]
    
    # 1. Initialize the state zonotope.
    # Start with s = obs (fixed). We dedicate the first z_dim generators to z.
    cz_s = ConstrainedZonotope.create(
        center=obs,
        generators=jnp.zeros((batch_size, z_dim, s_dim))
    )
    
    # 2. Iterate through actions
    for t in range(horizon):
        a = action_seq[t]
        if a.ndim == 1:
            a = a[None, :]
            
        # A. Prepare the joint input [s, a, z]
        # s depends on current generators. a is fixed. z depends on initial generators.
        n_gen = cz_s.n_gen
        center_joint = jnp.concatenate([cz_s.center, a, jnp.zeros((batch_size, z_dim))], axis=-1)
        
        G_joint = jnp.zeros((batch_size, n_gen, s_dim + action_dim + z_dim))
        # s part
        G_joint = G_joint.at[:, :, :s_dim].set(cz_s.generators)
        # z part (identity on the first z_dim generators)
        z_block = (jnp.eye(z_dim) * z_scale).reshape(1, z_dim, z_dim).repeat(batch_size, axis=0)
        G_joint = G_joint.at[:, :z_dim, s_dim + action_dim:].set(z_block)
        
        cz_joint = ConstrainedZonotope(
            center=center_joint,
            generators=G_joint,
            constraints_eq_A=cz_s.constraints_eq_A,
            constraints_eq_b=cz_s.constraints_eq_b,
            constraints_ineq_A=cz_s.constraints_ineq_A,
            constraints_ineq_b=cz_s.constraints_ineq_b
        )
        
        # B. Propagate
        cz_next = lift_epinet_propagation(model, cz_joint)
        
        # C. Apply Safety Constraints
        cz_next = ops.apply_constraints(cz_next, unsafe_A, unsafe_b)
        
        cz_s = cz_next
        
    return cz_s

def calculate_pessimistic_gaussian_safety_prob(cz: ConstrainedZonotope, z_dim: int) -> jax.Array:
    """
    Computes the probability of safety P(z in SafeZone) where z ~ N(0, I),
    accounting for the worst-case impact of non-epistemic generators (ReLU noise).
    
    Args:
        cz: ConstrainedZonotope containing accumulated constraints.
        z_dim: The number of initial generators corresponding to z.
        
    Returns:
        p_safe: Probability [Batch]
    """
    from jax.scipy.special import erfc
    
    A = cz.constraints_ineq_A # [Batch, N_cons, N_gen]
    b = cz.constraints_ineq_b # [Batch, N_cons]
    
    # 1. Split generators: Epistemic (z) vs Approximation Noise (xi_noise)
    A_z = A[:, :, :z_dim]
    A_noise = A[:, :, z_dim:]
    
    # 2. Calculate Worst-Case Noise Impact
    # Since xi_noise are in [-1, 1], the max value sum(A_noise * xi_noise) 
    # for each constraint row is sum(abs(A_noise)).
    max_noise_impact = jnp.sum(jnp.abs(A_noise), axis=2) # [Batch, N_cons]
    
    # 3. Adjust RHS (Head-room)
    # Effective constraint: A_z * z <= b - max_noise_impact
    effective_b = b - max_noise_impact

    # 4. Standard Gaussian CDF on the remaining z-polytope
    # u = A_z * z ~ N(0, sigma^2) where sigma^2 = sum(A_z^2)
    sigma_sq = jnp.sum(jnp.square(A_z), axis=2)
    sigma = jnp.sqrt(jnp.maximum(sigma_sq, 1e-10))
    
    def normal_cdf(x):
        # 0.5 * (1 + erf(x / sqrt(2)))
        # Here we use erfc(-x/sqrt(2)) for better numerical stability at far tails
        return 0.5 * erfc(-x / jnp.sqrt(2.0))

    # P(u_i <= effective_b_i)
    probs = normal_cdf(effective_b / sigma) # [Batch, N_cons]
    
    # 5. Combine (Product assuming independent constraints for the signal)
    p_safe = jnp.prod(probs, axis=1)
    
    return p_safe

def estimate_safety_mc(
    obs: jax.Array, 
    action_seq: jax.Array, 
    unsafe_A: jax.Array, 
    unsafe_b: jax.Array, 
    model: "Epinet", 
    n_samples: int = 1000, 
    key: Optional[jax.Array] = None
) -> jax.Array:
    """
    Estimates safety probability using Monte Carlo sampling of z.
    
    Args:
        obs: Initial observation [State_Dim]
        action_seq: Sequence of actions [Horizon, Action_Dim]
        unsafe_A: [N_cons, State_Dim]
        unsafe_b: [N_cons]
        model: Epinet instance
        n_samples: Number of samples to use.
        key: PRNG key. If None, uses jax.random.PRNGKey(0).
        
    Returns:
        prob: Estimated probability of the entire trajectory being safe [1] or [Batch]
    """
    if key is None:
        key = jax.random.PRNGKey(0)
        
    # Handle batching
    obs_is_unbatched = (obs.ndim == 1)
    if obs_is_unbatched:
        obs = obs[None, :]
    
    batch_size = obs.shape[0]
    z_dim = model.z_dim
    horizon = action_seq.shape[0]
    
    # Sample z ~ N(0, I) for each batch and each MC sample
    # Shape: [Batch, n_samples, z_dim]
    z_samples = jax.random.normal(key, (batch_size, n_samples, z_dim))
    
    def single_trajectory_rollout(s0, action_seq, z):
        """
        Runs a single trajectory and returns whether it remained safe.
        s0: [State_Dim]
        action_seq: [Horizon, Action_Dim]
        z: [z_dim]
        """
        def step_fn(curr_s, action):
            # next_s = model(x, z)
            # x = [s, a]
            x = jnp.concatenate([curr_s, action], axis=-1)
            # Epinet expects batched input, so we add-then-remove batch dim
            next_s = model(x[None, :], z[None, :])[0]
            
            # Check safety: H * s <= d
            is_safe = jnp.all(unsafe_A @ next_s <= unsafe_b)
            return next_s, is_safe

        _, safety_flags = jax.lax.scan(step_fn, s0, action_seq)
        return jnp.all(safety_flags)

    # Vectorize over MC samples AND original batch
    # vmap(single_trajectory_rollout, (None, None, 0)) -> Vectorize over z
    # vmap(..., (0, None, 0)) -> Vectorize over (Batch, n_samples)
    
    # Inner vmap: over n_samples
    mc_vmap = jax.vmap(single_trajectory_rollout, in_axes=(None, None, 0))
    
    # Outer vmap: over batch_size
    batch_vmap = jax.vmap(mc_vmap, in_axes=(0, None, 0))
    
    # results: [Batch, n_samples]
    results = batch_vmap(obs, action_seq, z_samples)
    
    # Probability per batch
    probs = jnp.mean(results.astype(jnp.float32), axis=1)
    
    if obs_is_unbatched:
        return probs[0]
    return probs

def estimate_reward_mc(
    obs: jax.Array, 
    action_seq: jax.Array, 
    unsafe_A: jax.Array, 
    unsafe_b: jax.Array, 
    model: "Epinet", 
    reward_fn, 
    n_samples: int = 1000, 
    key: Optional[jax.Array] = None
) -> jax.Array:
    """
    Estimates expected cumulative reward using Monte Carlo sampling of z,
    incorporating safety-based early termination.
    
    Args:
        obs: Initial observation [State_Dim]
        action_seq: Sequence of actions [Horizon, Action_Dim]
        unsafe_A: [N_cons, State_Dim]
        unsafe_b: [N_cons]
        model: Epinet instance
        reward_fn: Function (state, action) -> reward
        n_samples: Number of samples to use.
        key: PRNG key. If None, uses jax.random.PRNGKey(0).
        
    Returns:
        expected_reward: [1] or [Batch]
    """
    if key is None:
        key = jax.random.PRNGKey(0)
        
    # Handle batching
    obs_is_unbatched = (obs.ndim == 1)
    if obs_is_unbatched:
        obs = obs[None, :]
    
    batch_size = obs.shape[0]
    z_dim = model.z_dim
    
    # Sample z ~ N(0, I)
    z_samples = jax.random.normal(key, (batch_size, n_samples, z_dim))
    
    def single_trajectory_rollout(s0, action_seq, z):
        def step_fn(carry, action):
            curr_s, still_safe = carry
            # x = [s, a]
            x = jnp.concatenate([curr_s, action], axis=-1)
            next_s = model(x[None, :], z[None, :])[0]
            
            # Check safety: H * s <= d
            is_safe = jnp.all(unsafe_A @ next_s <= unsafe_b)
            now_safe = still_safe & is_safe
            
            # Mask reward
            reward = reward_fn(next_s, action) * now_safe
            return (next_s, now_safe), reward

        _, step_rewards = jax.lax.scan(step_fn, (s0, True), action_seq)
        return jnp.sum(step_rewards)

    # vmap over n_samples and batch_size
    mc_vmap = jax.vmap(single_trajectory_rollout, in_axes=(None, None, 0))
    batch_vmap = jax.vmap(mc_vmap, in_axes=(0, None, 0))
    
    results = batch_vmap(obs, action_seq, z_samples)
    expected_reward = jnp.mean(results, axis=1)
    
    if obs_is_unbatched:
        return expected_reward[0]
    return expected_reward
