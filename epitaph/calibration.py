import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List

def compute_gaussian_metrics(y_true: jnp.ndarray, y_preds: jnp.ndarray) -> Dict[str, float]:
    """
    Computes calibration metrics assuming the ensemble/z-samples represent a Gaussian distribution.
    
    Args:
        y_true: Ground truth targets [N, D]
        y_preds: Predicted values for different samples [S, N, D] (S = number of samples)
        
    Returns:
        Dictionary with NLL, RMSE, and Variance metrics.
    """
    # Mean and variance across z-samples (S dimension)
    mean = jnp.mean(y_preds, axis=0)
    variance = jnp.var(y_preds, axis=0)
    
    # Stable variance
    safe_variance = jnp.maximum(variance, 1e-6)
    
    # 1. Root Mean Squared Error (RMSE)
    rmse = jnp.sqrt(jnp.mean((mean - y_true)**2))
    
    # 2. Negative Log Likelihood (NLL)
    # NLL = 0.5 * log(2 * pi * var) + (y - mean)^2 / (2 * var)
    nll_term1 = 0.5 * jnp.log(2 * jnp.pi * safe_variance)
    nll_term2 = ((y_true - mean)**2) / (2 * safe_variance)
    nll = nll_term1 + nll_term2
    
    # Handle potential overflows/NaNs (clip to a large but finite range)
    avg_nll = jnp.mean(jnp.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=-1e6))
    
    # Final safety check for the float conversion
    rmse_val = float(jnp.nan_to_num(rmse, nan=1e6, posinf=1e6))
    nll_val = float(jnp.nan_to_num(avg_nll, nan=1e6, posinf=1e6, neginf=-1e6))
    avg_var_val = float(jnp.nan_to_num(jnp.mean(variance), nan=0.0, posinf=1e6))
    
    return {
        "rmse": rmse_val,
        "nll": nll_val,
        "avg_var": avg_var_val
    }

def compute_coverage(y_true: jnp.ndarray, y_preds: jnp.ndarray, quantiles: List[float] = [0.95]) -> Dict[str, float]:
    """
    Computes the Prediction Interval Coverage Probability (PICP).
    
    Checks what percentage of true values fall within the predicted intervals.
    """
    mean = jnp.mean(y_preds, axis=0)
    std = jnp.std(y_preds, axis=0)
    
    results = {}
    for q in quantiles:
        # For a normal distribution, the interval for probability q is mean +/- z_score * std
        # For 0.95, z_score is 1.96
        # We'll use a simple normal approximation for the z-score
        from scipy import stats
        z_score = stats.norm.ppf(1 - (1 - q) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = jnp.mean(jnp.nan_to_num(inside.astype(jnp.float32), nan=0.0))
        results[f"coverage_{int(q*100)}"] = float(coverage)
        
    return results

def plot_calibration_curve(y_true: jnp.ndarray, y_preds: jnp.ndarray):
    """
    Generates a calibration curve (Reliability Diagram).
    
    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    mean = jnp.mean(y_preds, axis=0)
    std = jnp.std(y_preds, axis=0)
    
    z_scores = (y_true - mean) / (std + 1e-6)
    cdfs = stats.norm.cdf(z_scores).flatten()
    
    sorted_cdfs = np.sort(np.nan_to_num(cdfs, nan=0.5))
    expected = np.linspace(0, 1, len(sorted_cdfs))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(expected, sorted_cdfs, label='Model Calibration', color='blue', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Cumulative Probability')
    ax.set_ylabel('Empirical Cumulative Probability')
    ax.set_title('Uncertainty Calibration (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def compute_rmsce(y_true: jnp.ndarray, y_preds: jnp.ndarray, bins: int = 10) -> float:
    """
    Computes Root Mean Squared Calibration Error (RMSCE) for regression.
    """
    from scipy import stats
    
    N, D = y_true.shape
    S = y_preds.shape[0]
    
    mean = jnp.mean(y_preds, axis=0)
    std = jnp.std(y_preds, axis=0)
    
    # Standardize errors
    # (y_true - mean) / std should follow N(0, 1) if perfectly calibrated
    z_scores = (y_true - mean) / (std + 1e-6)
    
    # Compute cumulative probabilities of these z-scores
    cdfs = stats.norm.cdf(z_scores)
    
    # If perfectly calibrated, cdf values should be Uniform(0, 1)
    # Sort and check deviation from Uniform
    sorted_cdfs = np.sort(np.nan_to_num(cdfs.flatten(), nan=0.5))
    expected = np.linspace(0, 1, len(sorted_cdfs))
    
    rmsce = np.sqrt(np.mean((sorted_cdfs - expected)**2))
    return float(np.nan_to_num(rmsce, nan=1.0))
