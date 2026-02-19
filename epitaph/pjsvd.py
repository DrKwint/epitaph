import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Callable, Optional, List

# ==============================================================================
# 1. Geometric Projections (The "Safety Net")
# ==============================================================================

def get_affine_residuals(outputs_batch: jax.Array, original_outputs_batch: jax.Array) -> jax.Array:
    """
    Projects the perturbed outputs onto the orthogonal complement of the subspace
    spanned by the Bias (1) and the Original Signal (y).
    
    This effectively measures the 'Unfixable Error' that cannot be removed by
    an affine transformation of the next layer.
    """
    # A. Bias Correction (Project onto vector of 1s)
    # This removes the mean shift.
    mean_correction = jnp.mean(outputs_batch, axis=0, keepdims=True)
    centered = outputs_batch - mean_correction
    
    # B. Scale Correction (Project onto original signal y)
    # This removes the correlation shift (scaling).
    y_centered = original_outputs_batch - jnp.mean(original_outputs_batch, axis=0, keepdims=True)
    
    # Gram-Schmidt Projection: v_orth = v - proj_u(v)
    dot_prod = jnp.sum(centered * y_centered, axis=0)
    norm_sq = jnp.sum(y_centered * y_centered, axis=0) + 1e-6
    
    scale_factor = dot_prod / norm_sq
    scale_correction = scale_factor * y_centered
    
    return centered - scale_correction

# ==============================================================================
# 2. The Solver (Projected Jacobian SVD)
# ==============================================================================

@jax.jit(static_argnums=(0,))
def find_optimal_perturbation(
    model_fn_wrt_layer: Callable[[jax.Array], jax.Array], 
    target_param: jax.Array, 
    max_iter: int = 100,
    tol: float = 1e-6,
    orthogonal_directions: Optional[jax.Array] = None,
    seed: int = 42
) -> Tuple[jax.Array, float]:
    """
    Finds the perturbation 'v' that minimizes ||(I-P)Jv|| using Adam.
    This corresponds to the singular vector of the 'Unfixable Jacobian' with 
    the smallest singular value.

    Args:
        model_fn_wrt_layer: Function f(W) -> outputs.
        target_param: The weight matrix W to perturb.
        max_iter: Maximum optimization steps.
        tol: Stop if loss (energy) drops below this threshold.
        orthogonal_directions: Matrix of shape (K, D) containing flattened vectors
                               we must remain orthogonal to.
        seed: Random seed for initialization.

    Returns:
        v: The optimal perturbation (same shape as target_param).
        sigma: The singular value (measure of unfixable error).
    """
    
    # 1. Cache the original output (constant wrt optimization)
    original_outputs = model_fn_wrt_layer(target_param)
    
    # Flatten directions if provided (K, ...) -> (K, D)
    if orthogonal_directions is not None:
        orthogonal_directions = orthogonal_directions.reshape((orthogonal_directions.shape[0], -1))

    
    # 2. Define the Energy Function (The quantity to minimize)
    def energy_fn(v_flat):
        # Enforce unit norm during the forward pass so gradients see it
        v_norm = v_flat / (jnp.linalg.norm(v_flat) + 1e-6)
        v_weight = v_norm.reshape(target_param.shape)
        
        # Compute Jv (Jacobian-Vector Product) efficiently
        _, jvp_out = jax.jvp(model_fn_wrt_layer, (target_param,), (v_weight,))
        
        # Measure only the "Unfixable" residual
        residuals = get_affine_residuals(jvp_out, original_outputs)
        return jnp.sum(residuals ** 2)

    # 3. Setup Optimizer
    optimizer = optax.adam(0.05)
    key = jax.random.PRNGKey(seed)
    
    # Random Initialization
    v_init = jax.random.normal(key, shape=(target_param.size,))
    v_init = v_init / jnp.linalg.norm(v_init)
    
    # 4. Orthogonalization Helper
    # If we have previous directions, we project them out: v = v - sum(proj_u(v))
    def project_orthogonal(v):
        if orthogonal_directions is not None and orthogonal_directions.size > 0:
            # Vectorized projection: v - sum((v . u) * u)
            # Assumes orthogonal_directions are already normalized
            projections = jnp.dot(orthogonal_directions, v) # (K,)
            subtraction = jnp.dot(projections, orthogonal_directions) # (D,)
            return v - subtraction
        return v

    # Initial orthogonalization
    v_init = project_orthogonal(v_init)
    opt_state = optimizer.init(v_init)

    # 5. The Optimization Loop
    # State: (iteration, v, opt_state, current_loss)
    init_val = (0, v_init, opt_state, jnp.inf)

    def cond_fun(val):
        i, _, _, loss = val
        return (i < max_iter) & (loss > tol)

    def body_fun(val):
        i, v, state, _ = val
        
        loss, grads = jax.value_and_grad(energy_fn)(v)
        
        updates, new_state = optimizer.update(grads, state)
        v_new = optax.apply_updates(v, updates)
        
        # Constraint 1: Orthogonality to previous vectors
        v_new = project_orthogonal(v_new)
        
        # Constraint 2: Unit Norm
        v_new = v_new / (jnp.linalg.norm(v_new) + 1e-6)
        
        return (i + 1, v_new, new_state, loss)

    # Execute loop on device
    final_i, final_v, _, final_loss = jax.lax.while_loop(cond_fun, body_fun, init_val)

    sigma = jnp.sqrt(final_loss)
    return final_v.reshape(target_param.shape), sigma


# ==============================================================================
# 3. The Corrector (Affine Adaptation)
# ==============================================================================

def apply_correction(
    next_layer_params: Tuple[jax.Array, jax.Array],
    original_stats: Tuple[jax.Array, jax.Array],
    new_outputs: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Adjusts the next layer's weights and bias to restore the original 
    mean and variance statistics, compensating for the perturbation.
    """
    w_next, b_next = next_layer_params
    mu_old, std_old = original_stats
    
    mu_new = jnp.mean(new_outputs, axis=0)
    std_new = jnp.std(new_outputs, axis=0)
    
    # 1. Scale Correction (Match Variance)
    # We scale the rows of W_next to change the effective input scale.
    scale_factor = std_old / (std_new + 1e-6)
    w_next_new = w_next * scale_factor[:, None] 
    
    # 2. Bias Correction (Match Mean)
    # We adjust b_next to compensate for the shift in expected activation.
    # Formula: b_new = b_old + E[y_old] - E[y_new]
    expected_activation_old = jnp.dot(mu_old, w_next)
    expected_activation_new = jnp.dot(mu_new, w_next_new)
    
    b_next_new = b_next + expected_activation_old - expected_activation_new
    
    return w_next_new, b_next_new