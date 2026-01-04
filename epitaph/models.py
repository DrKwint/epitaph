import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, List, Optional
from .structures import ConstrainedZonotope
from . import ops

class Epinet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        z_dim: int,
        base_width: int = 64,
        base_depth: int = 2,
        enn_width: int = 64,
        enn_depth: int = 1,
        prior_scale: float = 5.0,
        rngs: nnx.Rngs = None,
    ):
        self.prior_scale = prior_scale
        self.z_dim = z_dim
        
        # --- 1. Base Network Components ---
        # Split into representation and head for cleaner access
        base_layers = []
        curr = in_features
        for _ in range(base_depth):
            base_layers.append(nnx.Linear(curr, base_width, rngs=rngs))
            base_layers.append(nnx.relu)
            curr = base_width
        self.base_repr = nnx.Sequential(*base_layers)
        self.base_head = nnx.Linear(curr, out_features, rngs=rngs)

        # --- 2. Epinet Components ---
        # Input: [x, z, base_latent]
        enn_input_dim = in_features + z_dim + base_width
        
        # A. Learnable Epinet
        # We use a specific kernel_init for the FINAL layer to ensure 
        # it starts near zero.
        learnable_layers = []
        curr = enn_input_dim
        for _ in range(enn_depth):
            learnable_layers.append(nnx.Linear(curr, enn_width, rngs=rngs))
            learnable_layers.append(nnx.relu)
            curr = enn_width
            
        # Initialize final layer to Zero so f(x,z) approx equals Prior(x,z) at start
        learnable_layers.append(nnx.Linear(
            curr, 
            out_features, 
            kernel_init=nnx.initializers.zeros, 
            bias_init=nnx.initializers.zeros,
            rngs=rngs
        ))
        self.learnable_enn = nnx.Sequential(*learnable_layers)

        # B. Prior Network (Frozen Anchor)
        # We must ensure this uses a DIFFERENT random initialization than the learnable net
        # if we weren't zero-initializing the learnable net. 
        # Since we zero-init the learnable head, standard init here is fine.
        prior_layers = []
        curr = enn_input_dim
        for _ in range(enn_depth):
            prior_layers.append(nnx.Linear(curr, enn_width, rngs=rngs))
            prior_layers.append(nnx.relu)
            curr = enn_width
        prior_layers.append(nnx.Linear(curr, out_features, rngs=rngs))
        
        self.prior_enn = nnx.Sequential(*prior_layers)

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        # 1. Base Network Forward
        base_latent = self.base_repr(x)
        mu = self.base_head(base_latent)

        # 2. Prepare Epinet Inputs
        # CRITICAL: Stop gradient on base_latent to prevent Epinet 
        # from corrupting the Mean features.
        features_for_enn = jax.lax.stop_gradient(base_latent)
        
        enn_input = jnp.concatenate([x, z, features_for_enn], axis=-1)

        # 3. Prior Forward (Anchor)
        # Stop gradient ensures backprop is cut. 
        # NOTE: You must also exclude self.prior_enn.parameters from 
        # your optimizer to prevent weight decay!
        prior_out = jax.lax.stop_gradient(self.prior_enn(enn_input))
        
        # 4. Learnable Forward
        learnable_out = self.learnable_enn(enn_input)

        # 5. Combine
        # The learnable net learns to cancel out the prior to reduce uncertainty
        epistemic_offset = self.prior_scale * (learnable_out + prior_out)

        return mu + epistemic_offset

    def propagate_set(self, cz_input: ConstrainedZonotope, cz_z: ConstrainedZonotope) -> ConstrainedZonotope:
        """
        Propagates a constrained zonotope through the Epinet.
        
        Args:
            cz_input: Zonotope of [state, action].
            cz_z: Zonotope of epistemic indices [z]. 
                  Usually a standard zonotope (center 0, identity generators).
                  
        Returns:
            Zonotope of next states sn.
        """
        # 1. Base Network Forward
        cz_latent = cz_input
        for layer in self.base_repr.layers:
            if isinstance(layer, nnx.Linear):
                cz_latent = ops.propagate_linear(cz_latent, layer.kernel, layer.bias)
            elif layer is nnx.relu:
                cz_latent = ops.propagate_relu(cz_latent)
        
        cz_mu = ops.propagate_linear(cz_latent, self.base_head.kernel, self.base_head.bias)

        # 2. Prepare Epinet Inputs: [x, z, base_latent]
        cz_enn_input = ops.concatenate_zonotopes([cz_input, cz_z, cz_latent])

        # 3. Propagate through Prior and Learnable ENN
        def propagate_sequential(cz: ConstrainedZonotope, seq: nnx.Sequential) -> ConstrainedZonotope:
            for layer in seq.layers:
                if isinstance(layer, nnx.Linear):
                    cz = ops.propagate_linear(cz, layer.kernel, layer.bias)
                elif layer is nnx.relu:
                    cz = ops.propagate_relu(cz)
            return cz

        cz_prior = propagate_sequential(cz_enn_input, self.prior_enn)
        cz_learnable = propagate_sequential(cz_enn_input, self.learnable_enn)

        # 4. Combine: mu + prior_scale * (learnable + prior)
        cz_offset = ops.add_zonotopes(cz_learnable, cz_prior)
        cz_offset_scaled = ops.propagate_linear(cz_offset, jnp.eye(cz_offset.n_dim) * self.prior_scale)
        
        cz_final = ops.add_zonotopes(cz_mu, cz_offset_scaled)

        return cz_final

    def propagate_trajectory(
        self, 
        cz_state: ConstrainedZonotope, 
        actions: jax.Array, 
        n_z_gens: int
    ) -> List[ConstrainedZonotope]:
        """
        Rolls out a trajectory with a SINGLE value of z (consistent generators).
        
        Args:
            cz_state: Initial state zonotope.
            actions: Sequence of actions (Horizon, Batch, Action_Dim).
            n_z_gens: Number of generators that correspond to epistemic z.
            
        Returns:
            List of state zonotopes for each step.
        """
        states = [cz_state]
        horizon = actions.shape[0]
        batch_size = cz_state.center.shape[0]
        
        # Standard z zonotope (center 0, identity generators for xi_z)
        z_gens = jnp.eye(n_z_gens).reshape(1, n_z_gens, n_z_gens).repeat(batch_size, axis=0)
        cz_z = ConstrainedZonotope.create(
            center=jnp.zeros((batch_size, n_z_gens)),
            generators=z_gens
        )

        curr_cz = cz_state
        for t in range(horizon):
            # 1. Prepare input: [s, a]
            cz_action = ConstrainedZonotope.create(
                center=actions[t],
                generators=jnp.zeros((batch_size, 0, actions.shape[-1]))
            )
            
            # Joint [s_t, a_t]
            cz_in = ops.concatenate_zonotopes([curr_cz, cz_action])
            
            # 2. Propagate
            # cz_z is shared across all steps, ensuring temporal consistency.
            next_cz = self.propagate_set(cz_in, cz_z)
            
            states.append(next_cz)
            curr_cz = next_cz
            
        return states

    def propagate_ibp(self, s_int, a_int, z_int):
        """
        Forward Interval Propagation.
        Input: Intervals for s, a, z
        Returns: Output Interval, List[Interval] (intermediate bounds)
        """
        # Concatenate inputs
        # Bounds logic: [Lower_s, Lower_a, Lower_z]
        x_lower = jnp.concatenate([s_int.lower, a_int.lower, z_int.lower])
        x_upper = jnp.concatenate([s_int.upper, a_int.upper, z_int.upper])
        
        current_lower = x_lower
        current_upper = x_upper
        
        bounds_trace = []
        
        # Iterate layers
        for layer in self.layers:
            if isinstance(layer, nnx.Linear):
                # IBP Linear
                # W * mid + b +/- |W| * rad
                W = layer.kernel
                b = layer.bias
                
                mid = 0.5 * (current_lower + current_upper)
                rad = 0.5 * (current_upper - current_lower)
                
                # Center propagation
                mu = mid @ W + b
                # Radius propagation
                r = rad @ jnp.abs(W)
                
                current_lower = mu - r
                current_upper = mu + r
                
            elif isinstance(layer, nnx.ReLU):
                # Save pre-activation bounds for CROWN
                bounds_trace.append(Interval(current_lower, current_upper))
                
                # IBP ReLU
                current_lower = jax.nn.relu(current_lower)
                current_upper = jax.nn.relu(current_upper)
                
        return Interval(current_lower, current_upper), bounds_trace

    def backward_crown(self, A_out, b_out, layer_bounds):
        """
        Backpropagates CROWN bounds through one step of the network.
        """
        A_curr = A_out
        b_curr = b_out
        
        # Iterate backwards through layers
        # Note: We need to match bounds to layers correctly (reversed)
        
        for i, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, nnx.Linear):
                W = layer.kernel
                b = layer.bias
                
                # Linear Backprop: A_prev = A_curr @ W.T
                # Bias: b_prev = b_curr - A_curr @ b
                
                # A_curr: [Num_Specs, Out_Features]
                # W: [In_Features, Out_Features]
                
                bias_contribution = A_curr @ b
                b_curr = b_curr - bias_contribution
                A_curr = A_curr @ W.T
                
            elif isinstance(layer, nnx.ReLU):
                # CROWN ReLU Relaxation
                bounds = layer_bounds[i] # Pre-activation bounds
                l, u = bounds.lower, bounds.upper
                
                # Calculate Slopes (D) and Intercepts (d) based on l, u
                # If l > 0: D=1, d=0 (Active)
                # If u < 0: D=0, d=0 (Inactive)
                # If Unstable: Use CROWN heuristic
                #   Slope = u / (u - l)
                #   Intercept = -l * Slope
                
                # Note: This logic needs to handle the sign of A_curr to pick 
                # upper/lower relaxation. For simplicity here, we use the 
                # "Parallel" relaxation (always upper bound slope) which is sound.
                
                slope = jnp.where(l >= 0, 1.0, 0.0)
                slope = jnp.where(u <= 0, 0.0, slope)
                
                unstable = (l < 0) & (u > 0)
                slope_u = u / (u - l + 1e-10)
                
                # Use heuristic slope for unstable
                slope = jnp.where(unstable, slope_u, slope)
                intercept = jnp.where(unstable, -l * slope_u, 0.0)
                
                # Propagate A: A_new = A_curr * slope
                # Propagate b: b_new = b_curr - A_curr * intercept
                # Note: Elementwise mult for slope (broadcasting)
                
                bias_contribution = jnp.sum(A_curr * intercept, axis=-1)
                b_curr = b_curr - bias_contribution
                A_curr = A_curr * slope # Implicit diagonal matrix mult
                
        # Split A_curr into inputs [s, a, z]
        # Assuming input order was [s, a, z] in propagate_ibp
        s_dim = self.state_dim
        a_dim = self.action_dim
        
        A_s = A_curr[:, :s_dim]
        A_a = A_curr[:, s_dim:s_dim+a_dim]
        A_z = A_curr[:, s_dim+a_dim:]
        
        return A_s, A_a, A_z, b_curr