
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, List

class Interval(NamedTuple):
    lower: jnp.ndarray
    upper: jnp.ndarray

    @property
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @property
    def radius(self):
        return 0.5 * (self.upper - self.lower)

class LinearConstraint(NamedTuple):
    """ Represents A * z <= b """
    A: jnp.ndarray  # Shape [Num_Constraints, Z_Dim]
    b: jnp.ndarray  # Shape [Num_Constraints]

class CrownIBPVerifier:
    def __init__(self, model, z_dim, input_dim):
        self.model = model  # Flax NNX Epinet
        self.z_dim = z_dim
        self.input_dim = input_dim

    def get_safe_z_polytope(self, s0, action_seq, unsafe_A, unsafe_b):
        """
        Calculates the polytope (A_total, b_total) of 'z' values that keep the 
        entire trajectory safe.
        
        Args:
            s0: Initial state [State_Dim]
            action_seq: [Horizon, Action_Dim]
            unsafe_A, unsafe_b: Defines Unsafe Set (A s <= b). 
                                We convert this to Safe Set (complement).
        """
        horizon = action_seq.shape[0]

        # --- STEP 1: Forward IBP (Get bounds for ReLU stability) ---
        # We need the bounds of every intermediate layer to know the ReLU slopes
        # for the backward pass.
        #
        # Input to network is [s, a, z].
        # s, a are FIXED (Radius = 0). z is INTERVAL [-1, 1] (Radius = 1).
        
        ibp_bounds_trace = self._run_trajectory_ibp(s0, action_seq)


        # --- STEP 2: Backward CROWN (Generate Constraints) ---
        # We iterate through time t = 1 to H.
        # At each t, we check the safety constraint: A_safe * s_t <= d_safe
        # We backpropagate this to t=0 to find A_z * z <= d_z
        
        # Note: Usually Unsafe Set is a box. The "Safe Set" is the complement,
        # which is non-convex (Union of half-spaces). 
        # For this implementation, let's assume we are verifying against a 
        # set of LINEAR SAFETY SPECS (e.g. "stay in this box").
        # Safe Spec: H_safe * s <= d_safe
        H_safe, d_safe = self._invert_unsafe_to_safe(unsafe_A, unsafe_b)
        
        constraints_list = []

        # We verify each timestep independently (intersection of constraints)
        for t in range(horizon):
            # Get bounds relevant for step t
            # Trace is a list of lists: trace[t][layer_i]
            step_bounds = ibp_bounds_trace[t]
            
            # Run CROWN for this specific timestep's safety check
            A_z, b_z = self._backward_crown_step(
                H_safe, d_safe, 
                step_bounds, 
                s0, action_seq[:t+1]
            )
            
            constraints_list.append(LinearConstraint(A_z, b_z))

        # Stack all constraints
        # A_final: [Horizon * Num_Specs, Z_Dim]
        A_total = jnp.concatenate([c.A for c in constraints_list], axis=0)
        b_total = jnp.concatenate([c.b for c in constraints_list], axis=0)
        
        return A_total, b_total

    def _run_trajectory_ibp(self, s0, action_seq):
        """
        Runs IBP forward for the whole trajectory.
        Returns a list of bounds for each timestep.
        """
        
        def scan_fn(current_s_interval, action):
            # Create Interval Input for Network: [s_interval, a_fixed, z_interval]
            # s: Computed from previous step
            # a: Fixed value (radius 0)
            # z: Fixed range [-1, 1] (radius 1)
            
            # Action Interval
            a_int = Interval(action, action) 
            
            # Z Interval (Unit Hypercube)
            z_low = -jnp.ones((self.z_dim,))
            z_high = jnp.ones((self.z_dim,))
            z_int = Interval(z_low, z_high)
            
            # Propagate through Epinet (Layer by Layer IBP)
            # This function must return the output interval AND the intermediate 
            # intervals for all layers (for CROWN later).
            next_s_int, layer_bounds = self.model.propagate_ibp(
                current_s_interval, a_int, z_int
            )
            
            return next_s_int, layer_bounds

        # Initial State is fixed point
        s0_int = Interval(s0, s0)
        
        _, bounds_trace = jax.lax.scan(scan_fn, s0_int, action_seq)
        return bounds_trace

    def _backward_crown_step(self, C_spec, d_spec, step_bounds, s0, actions_subset):
        """
        Backpropagates specific linear constraints C_spec * s_t <= d_spec
        all the way to t=0 to isolate 'z'.
        """
        # Initialize Backward Carry
        # A_curr maps current layer output to the bound
        A_curr = C_spec
        b_curr = d_spec
        
        # We iterate BACKWARDS through the layers of the unrolled trajectory network
        # For a trajectory of length T, this is effectively Depth * T layers.
        
        # NOTE: In a real implementation, you'd loop t from T-1 down to 0.
        # Inside each step, you loop layers L down to 0.
        
        # Accumulator for Z constraints (Since z is shared across all steps)
        A_z_acc = jnp.zeros((C_spec.shape[0], self.z_dim))
        
        num_steps = len(actions_subset)
        
        for t in reversed(range(num_steps)):
            # Retrieve IBP bounds for this timestep
            # These determine the ReLU relaxations
            layer_bounds = step_bounds[t] # List of Intervals for this step's layers
            action = actions_subset[t]
            
            # 1. Backprop through Epinet Layers (Output -> Input)
            # This updates A_curr, b_curr, and calculates gradients w.r.t inputs (s, a, z)
            # The 'model.backward_crown' method handles the standard NN layers
            
            # Returns:
            # A_s: Gradient w.r.t state input
            # A_a: Gradient w.r.t action input
            # A_z: Gradient w.r.t epistemic input
            # b_new: Updated bias (absorbed layers)
            A_s, A_a, A_z, b_new = self.model.backward_crown(
                A_curr, b_curr, layer_bounds
            )
            
            # 2. Handle Inputs
            
            # Epistemic Z: It's a variable. Accumulate the constraint gradient.
            # We add it because the final constraint is (Sum A_z) * z <= b
            A_z_acc += A_z
            
            # Action: It's fixed. Absorb into bias.
            # term = A_a * action
            # Note: A_a is [Num_Specs, Action_Dim], action is [Action_Dim]
            bias_from_action = A_a @ action
            b_curr = b_new - bias_from_action
            
            # State:
            if t > 0:
                # Continue backpropagating to previous timestep
                A_curr = A_s
            else:
                # We are at t=0. S0 is fixed. Absorb into bias.
                bias_from_s0 = A_s @ s0
                b_curr = b_curr - bias_from_s0
                
        return A_z_acc, b_curr
    
    def _invert_unsafe_to_safe(self, unsafe_A, unsafe_b):
        # Heuristic Helper:
        # User provides Unsafe: A_u * x <= b_u (e.g. inside a box)
        # Safe is NOT (A_u * x <= b_u) -> Union of (A_u * x > b_u)
        # 
        # For CROWN (Convex), we need a convex safety spec.
        # Usually we pick the specific face of the unsafe box closest to us 
        # and verify we are on the 'safe' side of that face.
        #
        # For now, return a placeholder assuming user passes SAFE specs directly.
        return unsafe_A, unsafe_b