import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np
import os
from typing import Tuple, Callable, Any, List

from epitaph.pjsvd import find_optimal_perturbation, apply_correction
from epitaph.models import TransitionModel

# Ensure JAX falls back gracefully if no GPU
os.environ['JAX_PLATFORMS'] = 'cpu,cuda'

# =====================================================
# PART 1: Data Collection & Policies
# =====================================================

def positive_policy(env, obs):
    action_dim = env.action_space.shape[0]
    return np.random.uniform(0.5, 1.0, size=(action_dim,)).astype(np.float32)

def negative_policy(env, obs):
    action_dim = env.action_space.shape[0]
    return np.random.uniform(-1.0, -0.5, size=(action_dim,)).astype(np.float32)

def collect_data(env_name, steps, policy_fn, seed=0):
    print(f"Collecting {steps} steps ({policy_fn.__name__})...")
    env = gym.make(env_name)
        
    inputs, targets = [], []
    obs, _ = env.reset(seed=seed)
    
    for _ in range(steps):
        action = policy_fn(env, obs)
        # Fix for Gym API quirks
        if np.isscalar(action): action = np.array([action], dtype=np.float32)
        if action.ndim == 0: action = action[None]
            
        next_obs, _, term, trunc, _ = env.step(action)
        
        inputs.append(np.concatenate([obs, action])) # State + Action
        targets.append(next_obs) # Predict Next State
        
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    
    env.close()
    return jnp.array(np.stack(inputs)), jnp.array(np.stack(targets))

# =====================================================
# PART 2: Training
# =====================================================

def train_model(model, inputs, targets, steps=2000, batch_size=64):
    print(f"Training on {len(inputs)} samples...")
    optimizer = optax.adam(1e-3)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    indices = np.arange(len(inputs))
    for i in range(steps):
        batch = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch], targets[batch])
        if i % 500 == 0: print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f}")
    return model

# =====================================================
# PART 3: PJSVD Experiment Class
# =====================================================

class Ensemble:
    def __init__(self, base_model: TransitionModel, perturbations: List[Tuple]):
        self.base_model = base_model
        self.perturbations = perturbations # List of (w1, b1, w2, b2)

    def manual_forward(self, x, w1, b1, w2, b2):
        h1 = nnx.relu(x @ w1 + b1)
        h2 = nnx.relu(h1 @ w2 + b2)
        out = h2 @ self.base_model.l3.kernel.get_value() + self.base_model.l3.bias.get_value()
        return out

    def predict(self, x):
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            y = self.manual_forward(x, w1, b1, w2, b2)
            ys.append(y)
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x, idx):
        w1, b1, w2, b2 = self.perturbations[idx]
        return self.manual_forward(x, w1, b1, w2, b2)

class StandardEnsemble:
    def __init__(self, models: List[TransitionModel]):
        self.models = models

    def predict(self, x):
        ys = []
        for model in self.models:
            ys.append(model(x))
        return jnp.stack(ys, axis=0)
    
    def predict_one(self, x, idx):
        return self.models[idx](x)

def train_standard_ensemble(inputs, targets, n_models=5, steps=2000):
    print(f"\nTraining Standard Ensemble of {n_models} models...")
    models = []
    
    for i in range(n_models):
        rngs = nnx.Rngs(i) # Different seed for initialization
        model = TransitionModel(inputs.shape[1], targets.shape[1], rngs)
        
        # Train
        print(f"  Training Model {i+1}/{n_models}...")
        model = train_model(model, inputs, targets, steps=steps, batch_size=64)
        models.append(model)
        
    return StandardEnsemble(models)

def compute_nll(mean, variance, targets):
    """
    Computes Negative Log Likelihood (Gaussian).
    """
    variance = variance + 1e-6 # Stability
    nll = 0.5 * (jnp.log(variance) + (targets - mean)**2 / variance + jnp.log(2 * jnp.pi))
    return jnp.mean(nll)

def compute_calibration(mean, variance, targets, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for regression by checking quantiles.
    """
    std = jnp.sqrt(variance + 1e-6)
    # Calculate CDF values for targets under predicted Gaussian
    # Using error function for standard normal CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    z = (targets - mean) / (std * jnp.sqrt(2))
    cdfs = 0.5 * (1 + jax.scipy.special.erf(z))
    
    # ECE: Check if CDFs are uniform [0, 1]
    expected = jnp.linspace(0, 1, n_bins + 1)
    observed = []
    for p in expected:
        observed.append(jnp.mean(cdfs <= p))
    
    # Mean absolute difference between observed proportion and expected proportion
    return jnp.mean(jnp.abs(jnp.array(observed) - expected))

class PJSVDExperiment:
    def __init__(self, env_name="HalfCheetah-v5", steps=10000, subset_size=4096):
        self.env_name = env_name
        self.steps = steps
        self.subset_size = subset_size
        self.model = None
        self.inputs_id = None
        self.targets_id = None
        self.inputs_id_eval = None
        self.targets_id_eval = None
        self.inputs_ood = None
        self.targets_ood = None
        self.ensemble = None

    def setup(self):
        # Data Collection
        self.inputs_id, self.targets_id = collect_data(self.env_name, self.steps, positive_policy, seed=0)
        self.inputs_id_eval, self.targets_id_eval = collect_data(self.env_name, self.steps, positive_policy, seed=42)
        self.inputs_ood, self.targets_ood = collect_data(self.env_name, self.steps, negative_policy, seed=99)
        
        # Model Training
        rngs = nnx.Rngs(0)
        self.model = TransitionModel(self.inputs_id.shape[1], self.targets_id.shape[1], rngs)
        self.model = train_model(self.model, self.inputs_id, self.targets_id)
    
    def run_pjsvd(self, n_directions=50, n_perturbations=50):
        print("\n--- Running PJSVD ---")
        
        # Subset for analysis
        # Fix: Ensure we don't sample more than available if dataset is small
        actual_subset_size = min(len(self.inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(self.inputs_id), actual_subset_size, replace=False)
        X_subset = self.inputs_id[subset_idx]
        
        # Capture current weights
        W1_curr = self.model.l1.kernel.get_value()
        b1_curr = self.model.l1.bias.get_value()
        
        # Define function to analyze: Output of L1
        def model_fn_l1(w):
            return nnx.relu(X_subset @ w + b1_curr)
        
        # Find Null Space Directions
        print(f"Finding {n_directions} Orthogonal Null Space Directions...")
        directions = []
        sigmas = []
        
        # Fix 1: Directions Loop
        # We must stack the found directions into a JAX array for the next iteration
        # because 'find_optimal_perturbation' expects a JAX array, not a list.
        
        for k in range(n_directions):
            # Prepare orthogonal constraint
            if len(directions) > 0:
                orth_constraint = jnp.stack(directions)
            else:
                orth_constraint = None
                
            v_opt, sigma = find_optimal_perturbation(
                model_fn_l1, 
                W1_curr, 
                max_iter=500, 
                orthogonal_directions=orth_constraint
            )
            
            directions.append(v_opt)
            sigmas.append(sigma)
            print(f"  Direction {k+1}: Residual Sigma = {sigma:.6f}")
            
        # Convert to array for easy indexing
        v_opts = jnp.stack(directions) # (n_directions, n_in, n_out)
        
        # Generate Perturbed Models (Ensemble)
        print(f"Generating {n_perturbations} ensemble members...")
        perturbations = []
        
        # Cache original stats ONCE
        h_old = model_fn_l1(W1_curr)
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        
        W2_curr = self.model.l2.kernel.get_value()
        b2_curr = self.model.l2.bias.get_value()

        for i in range(n_perturbations):
            # Fix 2: Ellipsoid Sampling
            # We want to sample coefficients based on the sigma (smaller sigma = bigger coeff)
            # sigma is the "badness", so we divide by it.
            
            # Sample standard normal z
            z = np.random.normal(0, 1, size=n_directions)
            
            # Scale by inverse sigma (Ellipsoid Logic)
            # Add epsilon to sigma to avoid division by zero
            safe_sigmas = jnp.array(sigmas) + 1e-6
            coeffs = z / safe_sigmas
            
            # Normalize global size (e.g. to length 2.0 or 5.0)
            perturbation_size = 5.5
            coeffs = coeffs / np.linalg.norm(coeffs) * perturbation_size
            
            # Linear combination: sum_k (coeff_k * v_k)
            # reshape coeffs to (n_dir, 1, 1) to broadcast against v_opts (n_dir, in, out)
            weighted_vs = jnp.reshape(coeffs, (-1, 1, 1)) * v_opts
            total_perturbation = jnp.sum(weighted_vs, axis=0)
            
            W1_new = W1_curr + total_perturbation

            # Correction
            h_new = model_fn_l1(W1_new)
            
            W2_new, b2_new = apply_correction(
                (W2_curr, b2_curr), 
                (mu_old, std_old), 
                h_new
            )
            perturbations.append((W1_new, b1_curr, W2_new, b2_new))
            

        self.ensemble = Ensemble(self.model, perturbations)
        return self.ensemble

    def run_baseline(self, n_models=5, steps=2000):
        print(f"\n--- Running Standard Ensemble Baseline ({n_models} models) ---")
        self.ensemble = train_standard_ensemble(self.inputs_id, self.targets_id, n_models, steps)
        return self.ensemble

    def evaluate(self):
        print("\n--- Results ---")
        
        def compute_metrics(name, inputs, targets):
            # Get Ensemble Predictions: (N_Models, Batch, Dim)
            preds = self.ensemble.predict(inputs)
            
            # Compute Mean and Variance
            mean = jnp.mean(preds, axis=0)
            var = jnp.var(preds, axis=0)
            
            # 1. Variance (Epistemic Uncertainty)
            avg_var = jnp.mean(var)
            
            # 2. NLL
            nll = compute_nll(mean, var, targets)
            
            # 3. Calibration
            cal_err = compute_calibration(mean, var, targets)
            
            # 4. RMSE (Accuracy)
            rmse = jnp.sqrt(jnp.mean((mean - targets)**2))
            
            print(f"[{name}] RMSE: {rmse:.5f} | Var: {avg_var:.5f} | NLL: {nll:.5f} | CalibErr: {cal_err:.5f}")
            return rmse, avg_var, nll

        rmse_id, var_id, nll_id = compute_metrics("ID", self.inputs_id_eval, self.targets_id_eval)
        rmse_ood, var_ood, nll_ood = compute_metrics("OOD", self.inputs_ood, self.targets_ood)
        
        rmse_ratio = rmse_ood / (rmse_id + 1e-6)
        print(f"\nRMSE Ratio (OOD / ID): {rmse_ratio:.2f}x")
        
        var_ratio = var_ood / (var_id + 1e-9)
        print(f"Variance Ratio (OOD / ID): {var_ratio:.2f}x")

        if rmse_ratio > 2.0 and var_ratio > 2.0:
            print("SUCCESS: Perturbation is hidden on ID but visible on OOD.")
        else:
            print("FAILURE: Perturbation is either too small or leaked into ID.")

if __name__ == "__main__":
    
    # 1. Run PJSVD
    print("=== EXPERIMENT 1: PJSVD ===")
    experiment = PJSVDExperiment(env_name="HalfCheetah-v5")
    experiment.setup()
    experiment.run_pjsvd(n_directions=40, n_perturbations=1000)
    experiment.evaluate()

    # 2. Run Standard Baseline
    print("\n\n=== EXPERIMENT 2: Standard Baseline (Deep Ensemble) ===")
    # Reuse setup (same data)
    experiment.run_baseline(n_models=5, steps=2000)
    experiment.evaluate()