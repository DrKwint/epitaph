import jax
import jax.numpy as jnp
import numpy as np
import pytest
from epitaph import pjsvd
from flax import nnx

def test_get_affine_residuals_zero_for_affine_transform():
    """
    Verify that if outputs are an affine transformation of original_outputs,
    residuals are zero (or close to zero).
    """
    key = jax.random.PRNGKey(0)
    batch_size = 100
    n_features = 10
    
    original_outputs = jax.random.normal(key, (batch_size, n_features))
    
    # Create affine transformation: y = a * x + b
    # Note: The affine correction is per-neuron (column-wise).
    # So we can scale each column and add a bias to each column independently.
    
    scale = jnp.array([2.0] * n_features)
    shift = jnp.array([5.0] * n_features)
    
    transformed_outputs = original_outputs * scale + shift
    
    residuals = pjsvd.get_affine_residuals(transformed_outputs, original_outputs)
    
    # Residuals should be zero
    assert jnp.allclose(residuals, 0.0, atol=1e-5)

def test_get_affine_residuals_orthogonal():
    """
    Verify that residuals are orthogonal to the 'correctable' subspace (1 and y).
    """
    key = jax.random.PRNGKey(1)
    batch_size = 100
    n_features = 5
    
    original_outputs = jax.random.normal(key, (batch_size, n_features))
    perturbed_outputs = original_outputs + jax.random.normal(key, (batch_size, n_features))
    
    residuals = pjsvd.get_affine_residuals(perturbed_outputs, original_outputs)
    
    # 1. Orthogonal to Bias (Mean should be 0)
    # Use larger tolerance for float32 sum accumulation
    assert jnp.allclose(jnp.mean(residuals, axis=0), 0.0, atol=1e-5)
    
    # 2. Orthogonal to Signal (Correlation should be 0)
    # Center original first (as done in function)
    y_centered = original_outputs - jnp.mean(original_outputs, axis=0, keepdims=True)
    dot_prods = jnp.sum(residuals * y_centered, axis=0)
    
    assert jnp.allclose(dot_prods, 0.0, atol=1e-5)

def test_find_optimal_perturbation_simple_linear():
    """
    Test finding optimal perturbation on a simple linear layer y = Wx.
    """
    key = jax.random.PRNGKey(2)
    in_dim = 3
    out_dim = 2
    batch_size = 10
    
    k1, k2, k3 = jax.random.split(key, 3)
    W = jax.random.normal(k1, (in_dim, out_dim))
    x = jax.random.normal(k2, (batch_size, in_dim))
    
    def model_fn(w):
        return x @ w
        
    # Find 1 singular vector
    v, sigma = pjsvd.find_optimal_perturbation(model_fn, W)
    
    assert v.shape == W.shape
    assert sigma.shape == ()
    
    # Check if perturbation is normalized
    assert np.isclose(np.linalg.norm(v), 1.0)
    
    # Verify minimal residual
    # Compute Jacobian explicitly and check norm
    
    # Re-use internal helpers logic via explicit JVP
    _, jvp_out = jax.jvp(model_fn, (W,), (v,))
    res = pjsvd.get_affine_residuals(jvp_out, model_fn(W))
    # Sum of squares match? sigma is sqrt(energy) where energy = sum(res**2)
    norm_res = jnp.sqrt(jnp.sum(res**2))
    
    # The sigma should match this norm (approx)
    assert np.isclose(norm_res, sigma, rtol=1e-3)

    # Compare with a random direction
    random_dir = jax.random.normal(k3, W.shape)
    random_dir = random_dir / jnp.linalg.norm(random_dir)
    _, jvp_rand = jax.jvp(model_fn, (W,), (random_dir,))
    res_rand = pjsvd.get_affine_residuals(jvp_rand, model_fn(W))
    norm_rand = jnp.sqrt(jnp.sum(res_rand**2))
    
    # Optimal sigma should be less than or equal to random direction residual
    assert sigma <= norm_rand + 1e-4

# @pytest.mark.xfail(reason="Correction logic has approx 3% error not yet resolved")
def test_apply_correction_restores_stats():
    """
    Verify apply_correction restores mean and variance of the signal
    propagating through the NEXT layer.
    """
    key = jax.random.PRNGKey(3)
    dim_in = 10
    dim_out = 5 # Next layer output
    batch_size = 100
    
    # "Current" layer output (Input to next layer)
    k1, k2, k3 = jax.random.split(key, 3)
    h_old = jax.random.normal(k1, (batch_size, dim_in))
    mu_old = jnp.mean(h_old, axis=0) # This is what apply_correction uses as 'original_stats' of previous layer
    std_old = jnp.std(h_old, axis=0)
    
    # "Perturbed" output (simulated shift and scale)
    h_new = h_old * 1.5 + 2.0
    
    # Next Layer
    W_next = jax.random.normal(k2, (dim_in, dim_out))
    b_next = jax.random.normal(k3, (dim_out,))
        
    def next_layer_fn(w, b, h):
        return h @ w + b
        
    # Original output of next layer
    out_old = next_layer_fn(W_next, b_next, h_old)
    
    # Perturbed output of next layer (uncorrected)
    out_uncorrected = next_layer_fn(W_next, b_next, h_new)
    
    # Calculate Correction
    W_new, b_new = pjsvd.apply_correction(
        (W_next, b_next),
        (mu_old, std_old),
        h_new
    )
    
    # Corrected output
    out_corrected = next_layer_fn(W_new, b_new, h_new)
    
    # Check that corrected output matches original output statistically? 
    # Or exactly if the transform was affine?
    # Since h_new is exactly affine of h_old, and next_layer is linear,
    # we should theoretically recover the EXACT output if we invert the affine transform.
    
    # However, apply_correction does:
    # 1. Scale W to fix variance of h
    # 2. Shift b to fix mean of h
    
    # This effectively makes h_new act like h_old.
    # Let's check mean and var of out_corrected vs out_old.
    # They won't be identical element-wise unless W correction perfectly cancels h perturbation.
    # But since h_new = h_old * 1.5 + 2.0 is perfectly column-wise affine,
    # and correction is column-wise, it should be exact.
    
    assert jnp.allclose(out_corrected, out_old, rtol=1e-3, atol=1e-3)

def test_pjsvd_integration_two_layers():
    """
    Test PJSVD on a 2-layer network:
    1. Setup network and data.
    2. Find optimal perturbation for Layer 1.
    3. Apply perturbation and Stats Correction for Layer 2.
    4. Verify ID data output assumes minimal change.
    5. Verify OOD data output shows larger change (higher sensitivity).
    """
    key = jax.random.PRNGKey(42)
    in_dim = 20
    hidden_dim = 10
    out_dim = 2
    batch_size = 10
    
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # Data (In-Distribution)
    X_id = jax.random.normal(k1, (batch_size, in_dim))
    
    # Layer 1
    W1 = jax.random.normal(k2, (in_dim, hidden_dim))
    b1 = jnp.zeros((hidden_dim,)) # Simplify bias for layer 1 to focus on weights
    
    # Layer 2
    W2 = jax.random.normal(k3, (hidden_dim, out_dim))
    b2 = jax.random.normal(k4, (out_dim,))
    
    # Forward Pass Helpers
    def layer1(x, w, b):
        return jax.nn.relu(x @ w + b)
        
    def layer2(h, w, b):
        return h @ w + b
        
    def forward(x, w1, b1, w2, b2):
        h = layer1(x, w1, b1)
        return layer2(h, w2, b2)
        
    # Helpers for PJSVD
    def model_fn_l1(w):
        # wrt layer 1 weights, for finding perturbation
        return layer1(X_id, w, b1)
        
    # 1. Find perturbation for W1
    v, sigma = pjsvd.find_optimal_perturbation(model_fn_l1, W1)
    
    # Normalize v just to be sure (it should be unit norm)
    v = v / jnp.linalg.norm(v)
    
    # 2. Perturb W1
    # We want a perturbation size that is visible but small.
    epsilon = 0.1
    W1_new = W1 + epsilon * v
    
    # 3. Calculate Correction for Layer 2
    # Need stats of original Layer 1 output
    h_old = layer1(X_id, W1, b1)
    mu_old = jnp.mean(h_old, axis=0)
    std_old = jnp.std(h_old, axis=0)
    
    # Output of Layer 1 with new weights
    h_new = layer1(X_id, W1_new, b1)
    
    W2_new, b2_new = pjsvd.apply_correction(
        (W2, b2),
        (mu_old, std_old),
        h_new
    )
    
    # 4. Verify ID Fidelity
    y_id_old = forward(X_id, W1, b1, W2, b2)
    y_id_new = forward(X_id, W1_new, b1, W2_new, b2_new)
    
    diff_id = jnp.linalg.norm(y_id_new - y_id_old)
    print(f"\nID Difference Norm: {diff_id}")
    
    # Should be small. Ideally 0 for linear network if correction is perfect?
    # correction fixes mean and variance. If correlation structure changed, it might not be 0.
    # But PJSVD minimizes the residual (change in correlation structure unfixable by affine).
    # Since sigma is the residual, diff_id should be related to epsilon * sigma.
    
    # 5. Verify OOD Sensitivity
    # OOD Data: shifted distribution
    X_ood = X_id + jax.random.normal(k5, X_id.shape) * 2.0
    
    y_ood_old = forward(X_ood, W1, b1, W2, b2)
    y_ood_new = forward(X_ood, W1_new, b1, W2_new, b2_new)
    
    diff_ood = jnp.linalg.norm(y_ood_new - y_ood_old)
    print(f"OOD Difference Norm: {diff_ood}")
    
    # Expectation: OOD difference should be significantly larger than ID difference
    # because the perturbation v was chosen to minimize effect on ID, but not OOD.
    
    assert diff_id < diff_ood
    
    # Strict check for ID fidelity
    # Since we use linear layers, and apply_correction fixes 1st and 2nd moments,
    # and find_optimal_perturbation minimizes the remaining residual...
    # For a linear network, PJSVD finds v such that v is in nullspace of X.
    # If v in nullspace(X), then X @ v ~ 0.
    # Then h_new = X @ (W + eps*v) = XW + eps*Xv ~ XW.
    # So h_new ~ h_old.
    # Then correction should be minimal and output should be preserved perfectly?
    # If dimensions allow perfect nullspace. 
    # Batch=10, In=20. Nullspace exists (dim >= 10).
    # PJSVD should find v such that X @ v ~ 0.
    # So h_new ~ h_old.
    # ID diff should be very small.
    
    assert diff_id < 0.1 # Small change for ID (empirically ~0.06 with finding_optimal_perturbation)
    assert diff_ood > diff_id * 5.0 # OOD should be significantly larger (empirically ~8x)
