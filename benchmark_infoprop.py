import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import traceback
import numpy as np
import gymnasium as gym
from flax import nnx
import optax
import os

from epitaph.infoprop import infoprop_step
from run_pjsvd_experiment_2 import (
    StandardEnsemble, PJSVDExperiment, collect_data, 
    positive_policy, train_standard_ensemble, TransitionModel
)

# Ensure JAX falls back gracefully if no GPU
os.environ['JAX_PLATFORMS'] = 'cpu,cuda'

def rollout_ts(ensemble, start_states, horizon, steps_per_model=1):
    """
    Trajectory Sampling Rollout.
    """
    batch_size, dim = start_states.shape
    current_states = start_states
    trajectory = [current_states]
    
    # Pre-generate random model indices
    # (Horizon, Batch)
    n_models = len(ensemble.models) if isinstance(ensemble, StandardEnsemble) else len(ensemble.perturbations)
    model_indices = np.random.randint(0, n_models, size=(horizon, batch_size))
    
    # Policy for rollout: Random actions in [0.5, 1.0] (matching positive_policy)
    # We need to replicate the action logic.
    # HalfCheetah-v5 Action Dim = 6.
    action_dim = 6 
    
    for t in range(horizon):
        # Generate Actions
        # JAX uniform random actions
        key = jax.random.PRNGKey(t) # Simple deterministic key for reproducibility across runs
        actions = jax.random.uniform(key, shape=(batch_size, action_dim), minval=0.5, maxval=1.0)
        
        # Prepare Input: [State, Action]
        # current_states: (Batch, StateDim)
        model_inputs = jnp.concatenate([current_states, actions], axis=-1)
        
        # Predict with ALL models to get distribution (N_Models, Batch, Dim)
        preds = ensemble.predict(model_inputs)
        
        # Select one model per batch item
        # Advanced indexing: (Batch, Dim)
        selected_indices = model_indices[t]
        next_states = preds[selected_indices, jnp.arange(batch_size), :]
        
        # Add noise? (Aleatoric)
        # For this experiment, we assume deterministic transition + epistemic uncertainty focus
        # or we treat the model output as mean and add fixed variance.
        # The paper implies sampling from N(mu, Sigma).
        # Our TransitionModel outputs only mean.
        # We will assume a fixed small aleatoric noise for the "Sampling" part or rely on ensemble variance.
        # Actually, standard TS samples from the predicted distribution.
        # Since our models are deterministic points, TS just picks one.
        
        current_states = next_states
        trajectory.append(current_states)
        
    return jnp.stack(trajectory) # (Horizon+1, Batch, Dim)

def rollout_infoprop(ensemble, start_states, horizon):
    """
    Infoprop Rollout.
    """
    batch_size, dim = start_states.shape
    current_states = start_states
    trajectory = [current_states]
    
    # Metric tracking
    entropies = []
    
    action_dim = 6
    
    for t in range(horizon):
        # Generate Actions
        key = jax.random.PRNGKey(t + 1000)
        actions = jax.random.uniform(key, shape=(batch_size, action_dim), minval=0.5, maxval=1.0)
        
        # Prepare Input
        model_inputs = jnp.concatenate([current_states, actions], axis=-1)
        
        # 1. Get Ensemble Predictions
        # Means: (E, Batch, Dim)
        means = ensemble.predict(model_inputs)
        
        # Vars: (E, Batch, Dim)
        # Our models don't predict variance. We assume a fixed aleatoric variance
        # or learn it. The paper uses learned variance.
        # For this purpose, we'll assume a fixed variance for the base models.
        # This is a simplification but allows Alg 1 to work.
        fixed_var = 1e-3
        vars = jnp.ones_like(means) * fixed_var
        
        # 2. Raw Sample (TS Step)
        # Sample one model index randomly
        n_models = means.shape[0]
        idx = np.random.randint(0, n_models, size=(batch_size,))
        raw_sample = means[idx, jnp.arange(batch_size), :]
        
        # 3. Infoprop Step
        info = infoprop_step(means, vars, raw_sample)
        
        # 4. Next State
        # Sample from the refined distribution N(tilde_mu, tilde_Sigma)
        # tilde_mu = info['mean']
        # tilde_Sigma = info['var'] (diagonal)
        
        mu = info['mean']
        sigma = jnp.sqrt(info['var'])
        noise = np.random.normal(0, 1, size=mu.shape)
        next_states = mu + sigma * noise
        
        current_states = next_states
        trajectory.append(current_states)
        entropies.append(info['entropy'])
        
    return jnp.stack(trajectory), jnp.stack(entropies)

def compute_spread(trajectory):
    """
    Compute spread (log det of covariance) of the particle cloud at each step.
    Trajectory: (Horizon, Batch, Dim)
    """
    # spread = log det(Cov(S_t))
    # We compute covariance over the Batch dimension
    H, B, D = trajectory.shape
    spreads = []
    
    for t in range(H):
        data = trajectory[t] # (B, D)
        cov = jnp.cov(data, rowvar=False) + jnp.eye(D) * 1e-6
        # sign, logdet = jnp.linalg.slogdet(cov)
        # Using trace for variance spread if determinant is unstable or near zero
        # The paper mentions "Entropy" of the Infoprop state, but for the PLOT (Fig 4a),
        # it compares "Standard" (diverging) vs "Infoprop".
        # We'll use sum of variances (Trace) or Generalized Variance (Det).
        # Let's use Trace (Total Variance) for stability.
        
        spread = jnp.trace(cov)
        spreads.append(spread)
        
    return jnp.array(spreads)

def run_benchmark():
    # 1. Setup Data
    env_name = "HalfCheetah-v5"
    steps = 10000
    print(f"Collecting {steps} steps of data from {env_name}...")
    # Use positive policy for training data
    inputs, targets = collect_data(env_name, steps, positive_policy)
    
    # 2. Train Models
    print("Training Standard Ensemble...")
    std_ensemble = train_standard_ensemble(inputs, targets, n_models=5, steps=2000)
    
    print("Training PJSVD Ensemble...")
    pjsvd_exp = PJSVDExperiment(env_name=env_name, steps=steps)
    # Hack: Inject already collected data
    pjsvd_exp.inputs_id = inputs
    pjsvd_exp.targets_id = targets
    # Train base model
    rngs = nnx.Rngs(0)
    pjsvd_exp.model = TransitionModel(inputs.shape[1], targets.shape[1], rngs)
    # train_model is a standalone function in run_pjsvd_2
    from run_pjsvd_experiment_2 import train_model
    pjsvd_exp.model = train_model(pjsvd_exp.model, inputs, targets, steps=2000)
    
    # Generate PJSVD Perturbations
    pjsvd_ensemble = pjsvd_exp.run_pjsvd(n_directions=20, n_perturbations=50) # Smaller for speed
    
    # 3. Rollout comparison
    horizon = 50
    # Start states: take last batch from data
    start_states = inputs[-64:, :17] # HalfCheetah Obs dim is 17
    
    print(f"Running Rollouts (Horizon {horizon})...")
    
    results = {}
    
    # Standard + TS
    print("  Standard + TS...")
    traj_std_ts = rollout_ts(std_ensemble, start_states, horizon)
    spread_std_ts = compute_spread(traj_std_ts)
    results['Standard + TS'] = spread_std_ts
    
    # Standard + Infoprop
    print("  Standard + Infoprop...")
    traj_std_info, _ = rollout_infoprop(std_ensemble, start_states, horizon)
    spread_std_info = compute_spread(traj_std_info)
    results['Standard + Infoprop'] = spread_std_info
    
    # PJSVD + TS
    print("  PJSVD + TS...")
    traj_pjsvd_ts = rollout_ts(pjsvd_ensemble, start_states, horizon)
    spread_pjsvd_ts = compute_spread(traj_pjsvd_ts)
    results['PJSVD + TS'] = spread_pjsvd_ts
    
    # PJSVD + Infoprop
    print("  PJSVD + Infoprop...")
    traj_pjsvd_info, _ = rollout_infoprop(pjsvd_ensemble, start_states, horizon)
    spread_pjsvd_info = compute_spread(traj_pjsvd_info)
    results['PJSVD + Infoprop'] = spread_pjsvd_info
    
    # 4. Plot
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    t = np.arange(horizon + 1)
    
    plt.plot(t, spread_std_ts, label='Standard + TS', linestyle='--', color='red')
    plt.plot(t, spread_std_info, label='Standard + Infoprop', linestyle='-', color='red')
    
    plt.plot(t, spread_pjsvd_ts, label='PJSVD + TS', linestyle='--', color='blue')
    plt.plot(t, spread_pjsvd_info, label='PJSVD + Infoprop', linestyle='-', color='blue')
    
    plt.xlabel("Horizon Step")
    plt.ylabel("Trajectory Spread (Trace of Covariance)")
    plt.title("Rollout Divergence: TS vs Infoprop")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.abspath('infoprop_figure_4a.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    if os.path.exists(save_path):
        print("File verification: EXISTS")
    else:
        print("File verification: MISSING")
        
    with open('success_marker.txt', 'w') as f:
        f.write(f"Saved plot to {save_path}\n")
        f.write(f"Final Spread (Standard+TS): {spread_std_ts[-1]:.4f}\n")
        f.write(f"Final Spread (Standard+Infoprop): {spread_std_info[-1]:.4f}\n")
        f.write(f"Final Spread (PJSVD+TS): {spread_pjsvd_ts[-1]:.4f}\n")
        f.write(f"Final Spread (PJSVD+Infoprop): {spread_pjsvd_info[-1]:.4f}\n")

    print(f"Final Spread (Standard+TS): {spread_std_ts[-1]:.4f}")
    print(f"Final Spread (Standard+Infoprop): {spread_std_info[-1]:.4f}")
    print(f"Final Spread (PJSVD+TS): {spread_pjsvd_ts[-1]:.4f}")
    print(f"Final Spread (PJSVD+Infoprop): {spread_pjsvd_info[-1]:.4f}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        with open('error_log.txt', 'w') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
