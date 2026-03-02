import jax
import jax.numpy as jnp

def infoprop_step(
    ensemble_means: jax.Array,  # Shape (E, Batch, Dim)
    ensemble_vars: jax.Array,   # Shape (E, Batch, Dim) - diagonal variances
    raw_sample: jax.Array       # Shape (Batch, Dim) - s_hat_{t+1} from TS
) -> dict:
    """
    Performs a single Infoprop cleanup step (Algorithm 1).
    
    Returns dictionary with:
      - 'mean': The Infoprop mean \tilde{\mu} (Batch, Dim)
      - 'cov': The Infoprop covariance \tilde{\Sigma} (Batch, Dim, Dim)
      - 'entropy': The Information Loss metric (Batch,)
    """
    E, Batch, Dim = ensemble_means.shape
    
    # --- Step A: Decompose the Ensemble Prediction ---
    
    # 1. Estimate Estimate Environment Signal (\bar{S}_{t+1})
    # Covariance Intersection with uniform weights (w_i = 1/E)
    # \bar{\Sigma} = (1/E * sum((Sigma^e)^-1))^-1
    # Since diagonal, we can just do harmonic mean of variances
    
    # (E, Batch, Dim)
    inv_vars = 1.0 / (ensemble_vars + 1e-9)
    sum_inv_vars = jnp.sum(inv_vars, axis=0) # (Batch, Dim)
    
    # \bar{\Sigma} (diagonal)
    signal_var = E / (sum_inv_vars + 1e-9) # (Batch, Dim)
    
    # \bar{\mu} = \bar{\Sigma} * (1/E * sum((Sigma^e)^-1 * mu^e))
    #           = signal_var * (1/E) * sum(inv_vars * means)
    weighted_means = jnp.sum(inv_vars * ensemble_means, axis=0) # (Batch, Dim)
    signal_mean = signal_var * (1.0 / E) * weighted_means # (Batch, Dim)
    
    # 2. Estimate Epistemic Variance (\bar{\Sigma}^\Delta)
    # Sample variance of ensemble means
    # \bar{\Sigma}^\Delta = 1/E * sum((mu^e - \bar{\mu})(mu^e - \bar{\mu})^T)
    # We will keep this diagonal for efficiency and consistency with diagonal inputs
    diff = ensemble_means - signal_mean[None, :, :] # (E, Batch, Dim)
    epistemic_var = jnp.mean(diff ** 2, axis=0) # (Batch, Dim)
    
    # --- Step B: The Infoprop Update (Kalman Fusion) ---
    
    # Kalman Gain K = \bar{\Sigma} (\bar{\Sigma} + \bar{\Sigma}^\Delta)^{-1}
    # All diagonal here
    total_var = signal_var + epistemic_var + 1e-9
    K = signal_var / total_var # (Batch, Dim)
    
    # \tilde{\mu} = \bar{\mu} + K * (raw_sample - \bar{\mu})
    # This pulls the raw sample towards the signal mean based on confidence
    infoprop_mean = signal_mean + K * (raw_sample - signal_mean)
    
    # \tilde{\Sigma} = (I - K) \bar{\Sigma}
    # Note: K = sig / (sig + epi)
    # I - K = (sig + epi - sig) / (sig + epi) = epi / (sig + epi)
    # So \tilde{\Sigma} = (epi * sig) / (sig + epi)
    # This is essentially the harmonic mean of signal var and epistemic var? 
    # Actually K is a matrix, but here diagonal.
    
    infoprop_var = (1.0 - K) * signal_var
    
    # --- Step C: Metric (Quantized Entropy) ---
    
    # H = 0.5 * log((2*pi*e)^n * |Sigma|)
    #   = 0.5 * (n * log(2*pi*e) + sum(log(sigma_ii)))
    
    log_det = jnp.sum(jnp.log(infoprop_var + 1e-9), axis=-1) # (Batch,)
    const_term = Dim * jnp.log(2 * jnp.pi * jnp.e)
    entropy = 0.5 * (const_term + log_det)
    
    return {
        'mean': infoprop_mean,
        'var': infoprop_var, # Keeping it diagonal (Batch, Dim)
        'entropy': entropy,
        'signal_mean': signal_mean,
        'signal_var': signal_var,
        'epistemic_var': epistemic_var
    }
