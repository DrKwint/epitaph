from epitaph.model_utils import lift_epinet_propagation
import jax
import jax.numpy as jnp
from typing import Tuple, List, Callable, Optional
from flax import nnx
from .models import Epinet
from .structures import ConstrainedZonotope
from . import ops
import gymnasium as gym
from tqdm import trange

def initialize_cz(state, z_dim):
    """
    Args:
        state: Shape [Batch, State_Dim] or [State_Dim]
        z_dim: Int, dimension of epistemic index
    Returns:
        ConstrainedZonotope object
    """
    if state.ndim == 1:
        state = state[None, :]
    batch_size = state.shape[0]
    state_dim = state.shape[1]
    
    # 1. Center is the state
    center = state
    
    # 2. Generators: Reserve slots for 'z', but set influence to 0
    # Shape: [Batch, z_dim, State_Dim]
    generators = jnp.zeros((batch_size, z_dim, state_dim))
    
    # 3. Constraints: Initially empty
    # We initialize with a batch dimension
    A = jnp.zeros((batch_size, 0, z_dim))
    b = jnp.zeros((batch_size, 0))
    
    return ConstrainedZonotope(
        center=center, 
        generators=generators, 
        constraints_eq_A=A, 
        constraints_eq_b=b, 
        constraints_ineq_A=A, 
        constraints_ineq_b=b
    )

def manual(model: Epinet):
    cz = initialize_cz(jnp.zeros((1, 2)), 2)
    

def get_safe_z_constraints(model, s0, action_seq, unsafe_A, unsafe_b):
    """
    Returns the polytope (C, d) defining the Safe Z Set.
    C * xi_z <= d
    
    Args:
        model: Epinet model
        s0: Initial state [State_Dim] or [Batch, State_Dim]
        action_seq: Sequence of actions [Horizon, Action_Dim] or [Batch, Horizon, Action_Dim]
        unsafe_A: [N_cons, State_Dim]
        unsafe_b: [N_cons]
    """
    # Normalize inputs to have batch dimension
    if s0.ndim == 1:
        s0 = s0[None, :]
    if action_seq.ndim == 2:
        action_seq = action_seq[None, :, :] # [1, Horizon, Action_Dim]
    
    batch_size = s0.shape[0]
    horizon = action_seq.shape[1]
    n_z = model.z_dim
    
    # Permute action_seq for scan: [Horizon, Batch, Action_Dim]
    action_seq_scan = jnp.transpose(action_seq, (1, 0, 2))
    
    def scan_fn(carry, action):
        curr_center, curr_G_z = carry # curr_G_z: [Batch, N_z, State_Dim]
        
        # We manually propagate the center and generators through the network layers.
        # This is equivalent to model.propagate_set but we ignore the ReLU error terms
        # for the purpose of safety constraint DERIVATION (which is often a linear approximation anyway).
        # Actually, for correctness, we should use the same PWL logic but just not track 
        # the new generators in the carry.
        
        # Create a CZ with ONLY the z-generators to propagate
        cz = ConstrainedZonotope.create(
            center=curr_center,
            generators=curr_G_z
        )
        # Augment with action
        cz_action = ConstrainedZonotope.create(
            center=action,
            generators=jnp.zeros((batch_size, 0, action.shape[-1]))
        )
        cz_in = ops.concatenate_zonotopes([cz, cz_action])
        
        # Propagate through the network. 
        # We use a dummy z_cz (zeros) since we already have z-generators in 'cz_in'.
        cz_z_dummy = ConstrainedZonotope.create(
            center=jnp.zeros((batch_size, n_z)),
            generators=jnp.zeros((batch_size, 0, n_z))
        )
        next_cz = model.propagate_set(cz_in, cz_z_dummy)
        
        # The next_cz will have more generators than n_z. 
        # We ONLY keep the first n_z for the next state carry.
        # These correspond to the original xi_z.
        next_center = next_cz.center
        next_G_z = next_cz.generators[:, :n_z, :]
        
        # B. Generate Safety Constraints: unsafe_A * s <= unsafe_b
        # C_step = unsafe_A @ G_z
        C_step = jnp.einsum("ic,bjc->bij", unsafe_A, next_G_z) # [Batch, N_cons, N_z]
        d_step = unsafe_b[None, :] - jnp.einsum("ic,bc->bi", unsafe_A, next_center) # [Batch, N_cons]
        
        return (next_center, next_G_z), (C_step, d_step)

    # Run Scan
    # Initial state has no dependence on z yet, but we are looking for HOW z affects future states.
    # We initialize G_z to zero.
    init_center = s0
    init_G_z = jnp.zeros((batch_size, n_z, s0.shape[-1]))
    
    lift_epinet_propagation(model, ConstrainedZonotope.create(center=init_center, generators=init_G_z))
    _, (C_stack, d_stack) = jax.lax.scan(scan_fn, (init_center, init_G_z), action_seq_scan)
    
    # C_stack: [Horizon, Batch, N_cons, N_z]
    # d_stack: [Horizon, Batch, N_cons]
    
    # Reshape to [Batch, Horizon * N_cons, N_z] and [Batch, Horizon * N_cons]
    C_total = jnp.transpose(C_stack, (1, 0, 2, 3)).reshape(batch_size, -1, n_z)
    d_total = jnp.transpose(d_stack, (1, 0, 2)).reshape(batch_size, -1)
    
    # If input was not batched, squeeze output
    if s0.shape[0] == 1 and action_seq.shape[0] == 1:
        return C_total[0], d_total[0]
        
    return C_total, d_total

class TrialController:
    def __init__(
        self,
        model: Epinet,
        action_space: gym.spaces.Box,
        horizon: int = 15,
        n_samples: int = 10,
        temperature: float = 1.0,
        safety_weight: float = 10.0,
        noise_sigma: float = 0.1,
        seed: int = 0
    ):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        self.safety_weight = safety_weight
        self.noise_sigma = noise_sigma
        self.key = jax.random.PRNGKey(seed)
        self.action_space = action_space
        self.act_sequence = [self.action_space.sample() for _ in range(self.horizon)]
    
    def plan(
        self, 
        state: jax.Array, 
        reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
        constraints: Optional[Tuple[jax.Array, jax.Array]] = None
    ) -> jax.Array:
        """
        Calculates the best next action using MPPI with safety weighting.
        Refactored to use jax.lax.scan for memory efficiency.
        """
        self.key, subkey = jax.random.split(self.key)
        
        # 1. Sample perturbations
        noise = jax.random.normal(subkey, (self.n_samples, self.horizon, self.action_space.shape[0])) * self.noise_sigma
        candidate_sequences = jnp.array(self.act_sequence)[None, :, :] + noise
        safe_z_constraints = get_safe_z_constraints(self.model, state, candidate_sequences, constraints[0], constraints[1])
        print("hello")

class MPPIController:
    def __init__(
        self,
        model: Epinet,
        action_space: gym.spaces.Box,
        horizon: int = 15,
        n_samples: int = 100,
        temperature: float = 1.0,
        safety_weight: float = 10.0,
        noise_sigma: float = 0.1,
        seed: int = 0
    ):
        self.model = model
        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        self.safety_weight = safety_weight
        self.noise_sigma = noise_sigma
        self.key = jax.random.PRNGKey(seed)
        
        # Action mean (sequence for the horizon)
        self.action_space = action_space
        self.act_sequence = [self.action_space.sample() for _ in range(self.horizon)]

    def plan(
        self, 
        state: jax.Array, 
        reward_fn: Callable[[jax.Array, jax.Array], jax.Array],
        constraint_fn: Optional[Callable[[jax.Array], Tuple[jax.Array, jax.Array]]] = None
    ) -> jax.Array:
        """
        Calculates the best next action using MPPI with safety weighting.
        Refactored to use jax.lax.scan for memory efficiency.
        """
        self.key, subkey = jax.random.split(self.key)
        
        # 1. Sample perturbations
        noise = jax.random.normal(subkey, (self.n_samples, self.horizon, self.action_space.shape[0])) * self.noise_sigma
        candidate_sequences = jnp.array(self.act_sequence)[None, :, :] + noise
        
        # 2. Define the Single-Step Transition for Scan
        # This function handles ONE step of simulation for ONE trajectory
        def step_fn(carry, action):
            current_s, current_cz, cum_reward, cum_log_safety = carry
            
            # --- 1. Point Propagation (For Reward) ---
            # Standard forward pass of the neural network
            # We use the mean prediction (z=0) for reward estimation
            z_mean = jnp.zeros((self.model.z_dim,))
            
            # Inputs: [Batch=1, Dim]
            # We unsqueeze to add batch dim, then [0] to remove it after
            next_s = self.model(current_s[None, :], z_mean[None, :], action[None, :])[0]

            # --- 2. Set Propagation (For Safety) ---
            # This is the "abstract interpretation" call
            # We propagate the Zonotope through the SAME layers, but using set algebra
            
            # A. Augment Zonotope with Action
            # The network expects input [s, a]. We have 's' as a Zonotope. 
            # We treat 'a' as a precise value (a point zonotope with 0 generators).
            cz_input = ops.augment_zonotope_with_action(current_cz, action)
            
            # B. Propagate through the network layers
            # This calls the PWL logic (Linear interval arithmetic + ReLU abstractions)
            next_cz = self.model.propagate_symbolic(cz_input)
            
            
            # --- 3. Safety & Reward Checks ---
            step_reward = reward_fn(current_s, action)
            
            # Check intersection with unsafe sets, calculate probability, etc.
            # (Same logic as previous message)
            step_log_safe = ... 
            
            return (next_s, next_cz, cum_reward + step_reward, cum_log_safety + step_log_safe), None

        # 3. Vectorized Evaluation
        @jax.vmap
        def evaluate_trajectory(u_seq):
            # Initial State (Point)
            s0 = state
            
            # Initial State (Zonotope)
            # Create a point zonotope for s0
            cz0 = ConstrainedZonotope.create(
                center=state[None, :],
                generators=jnp.zeros((1, 0, state.shape[0]))
            )
            
            # Initial Accumulators
            init_carry = (s0, cz0, 0.0, 0.0) # (s, cz, reward, log_safety)
            
            # Run Scan
            # This loops over u_seq (Horizon) without unrolling the graph
            (final_s, final_cz, total_reward, total_log_safety), _ = jax.lax.scan(
                step_fn, 
                init_carry, 
                u_seq
            )
            
            # Convert log safety back to probability (or keep in log space for weighting)
            return total_reward, jnp.exp(total_log_safety)

        # Run VMAP
        # Now JAX only traces the logic ONCE, regardless of Horizon or Sample count
        rewards, safety_probs = evaluate_trajectory(candidate_sequences)
        
        # 4. Weighting & Update (Standard MPPI)
        log_weights = (rewards / self.temperature) + self.safety_weight * jnp.log(safety_probs + 1e-10)
        max_log_w = jnp.max(log_weights)
        weights = jnp.exp(log_weights - max_log_w)
        weights = weights / (jnp.sum(weights) + 1e-10)
        
        new_sequence = jnp.sum(weights[:, None, None] * candidate_sequences, axis=0)
        
        # Shift
        self.act_sequence = jnp.roll(new_sequence, shift=-1, axis=0)
        self.act_sequence = self.act_sequence.at[-1].set(jnp.zeros(self.action_dim))
        
        return new_sequence[0]