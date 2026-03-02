import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from epitaph.pjsvd import find_optimal_perturbation, apply_correction

# Configuration
ACTIVATION = 'gelu' # 'crelu' or 'gelu'
MODE = 'deep_ensemble' # 'pjsvd' or 'deep_ensemble'
ENSEMBLE_SIZE = 5 # for deep_ensemble

class ToyModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs, activation=ACTIVATION):
        self.activation = activation
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        if self.activation == 'crelu':
            self.l2 = nnx.Linear(128, 64, rngs=rngs) # Takes doubled features
        else:
            self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x):
        if self.activation == 'crelu':
            z1 = self.l1(x)
            h1 = jnp.concatenate([nnx.relu(z1), jnp.minimum(0.0, z1)], axis=-1)
            h2 = nnx.relu(self.l2(h1))
        elif self.activation == 'gelu':
            h1 = nnx.gelu(self.l1(x))
            h2 = nnx.gelu(self.l2(h1))
        out = self.l3(h2)
        return out

# Ensure JAX falls back gracefully if no GPU
os.environ['JAX_PLATFORMS'] = 'cpu,cuda'

def generate_moons_data(n_samples=200, noise=0.1):
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return x.astype(np.float32), y.astype(np.int32)

def get_eval_data(grid_size=60):
    # x ~ uniform grid [-4, 4]^2
    grid = np.linspace(-4, 4, grid_size).astype(np.float32)
    x1, x2 = np.meshgrid(grid, grid)
    x = np.stack([x1.flatten(), x2.flatten()], axis=1)
    return x, x1, x2

def train_model(model, inputs, targets, steps=3000):
    print(f"Training on {len(inputs)} samples...")
    optimizer = optax.adam(1e-2)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            logits = model(x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        
        grads = nnx.grad(loss_fn)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss_fn(model), new_opt

    for i in range(steps):
        # Full batch 
        loss, opt_state = train_step(model, opt_state, inputs, targets)
        if i % 1000 == 0: 
            print(f"Step {i}: Loss {loss:.5f}")
            
    print(f"Final Loss: {loss:.5f}")
    return model

class Ensemble:
    def __init__(self, base_model, perturbations):
        self.base_model = base_model
        self.perturbations = perturbations # List of (w1, b1, w2, b2)

    def manual_forward(self, x, w1, b1, w2, b2):
        if self.base_model.activation == 'crelu':
            z1 = x @ w1 + b1
            h1 = jnp.concatenate([nnx.relu(z1), jnp.minimum(0.0, z1)], axis=-1)
            h2 = nnx.relu(h1 @ w2 + b2)
        elif self.base_model.activation == 'gelu':
            h1 = nnx.gelu(x @ w1 + b1)
            h2 = nnx.gelu(h1 @ w2 + b2)
        # L3 remains untouched
        out = h2 @ self.base_model.l3.kernel.get_value() + self.base_model.l3.bias.get_value()
        return out

    def predict_proba(self, x):
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            logits = self.manual_forward(x, w1, b1, w2, b2)
            probs = nnx.softmax(logits, axis=-1)
            ys.append(probs)
        return jnp.stack(ys, axis=0)

class DeepEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, x):
        ys = []
        for model in self.models:
            logits = model(x)
            probs = nnx.softmax(logits, axis=-1)
            ys.append(probs)
        return jnp.stack(ys, axis=0)

def train_deep_ensemble(x_train, y_train, n_models=5, steps=5000):
    print(f"\n--- Training Deep Ensemble of {n_models} models ---")
    models = []
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")
        rngs = nnx.Rngs(i + 42) # Different seed for each model
        model = ToyModel(in_features=2, out_features=2, rngs=rngs, activation=ACTIVATION)
        model = train_model(model, x_train, y_train, steps=steps)
        models.append(model)
    return DeepEnsemble(models)

def run_pjsvd(model, x_train, n_directions=40, n_perturbations=100, perturbation_size=2.0):
    print("\n--- Running PJSVD ---")
    
    W1_curr = model.l1.kernel.get_value()
    b1_curr = model.l1.bias.get_value()
    
    # Define function to analyze: Output of L1
    def model_fn_l1(w):
        if model.activation == 'crelu':
            z1 = x_train @ w + b1_curr
            return jnp.concatenate([nnx.relu(z1), jnp.minimum(0.0, z1)], axis=-1)
        elif model.activation == 'gelu':
            return nnx.gelu(x_train @ w + b1_curr)
    
    print(f"Finding {n_directions} Orthogonal Null Space Directions...")
    directions = []
    sigmas = []
    
    for k in range(n_directions):
        orth_constraint = jnp.stack(directions) if len(directions) > 0 else None
            
        v_opt, sigma = find_optimal_perturbation(
            model_fn_l1, 
            W1_curr, 
            max_iter=500, 
            orthogonal_directions=orth_constraint
        )
        directions.append(v_opt)
        sigmas.append(sigma)
        print(f"  Direction {k+1}: Residual Sigma = {sigma:.6f}")
        
    v_opts = jnp.stack(directions)
    
    h_old = model_fn_l1(W1_curr)
    mu_old = jnp.mean(h_old, axis=0)
    std_old = jnp.std(h_old, axis=0)
    
    W2_curr = model.l2.kernel.get_value()
    b2_curr = model.l2.bias.get_value()

    print(f"Generating {n_perturbations} ensemble members (size={perturbation_size})...")
    perturbations = []
    
    n_pairs = n_perturbations // 2
    
    for i in range(n_pairs):
        z = np.random.normal(0, 1, size=n_directions)
        safe_sigmas = jnp.array(sigmas) + 1e-6
        coeffs = z / safe_sigmas
        
        # Rescale strictly to perturbation_size
        coeffs = coeffs / np.linalg.norm(coeffs) * perturbation_size
        
        weighted_vs = jnp.reshape(coeffs, (-1, 1, 1)) * v_opts
        total_perturbation = jnp.sum(weighted_vs, axis=0)
        
        # --- Boundary-Aware Scaling ---
        if model.activation == 'crelu':
            # Ensure no training point crosses the ReLU boundary to preserve exact mean.
            z_curr = x_train @ W1_curr + b1_curr
            delta_z = x_train @ total_perturbation
            
            # We only care when (z_curr * delta_z) < 0
            mask = (z_curr * delta_z) < 0
            safe_alphas = jnp.where(mask, jnp.abs(z_curr / (delta_z + 1e-9)), jnp.inf)
            max_safe_alpha = jnp.min(safe_alphas)
            
            # Scale to 99% of the boundary hit to be perfectly safe
            scale_factor = jnp.minimum(1.0, 0.99 * max_safe_alpha)
            total_perturbation = total_perturbation * scale_factor
            
            # Positive perturbation
            W1_new_pos = W1_curr + total_perturbation
            z_new_pos = x_train @ W1_new_pos + b1_curr
            h_new_pos = jnp.concatenate([nnx.relu(z_new_pos), jnp.minimum(0.0, z_new_pos)], axis=-1)
            W2_new_pos, b2_new_pos = apply_correction((W2_curr, b2_curr), (mu_old, std_old), h_new_pos)
            perturbations.append((W1_new_pos, b1_curr, W2_new_pos, b2_new_pos))
            
            # Negative perturbation (-z)
            W1_new_neg = W1_curr - total_perturbation
            z_new_neg = x_train @ W1_new_neg + b1_curr
            h_new_neg = jnp.concatenate([nnx.relu(z_new_neg), jnp.minimum(0.0, z_new_neg)], axis=-1)
            W2_new_neg, b2_new_neg = apply_correction((W2_curr, b2_curr), (mu_old, std_old), h_new_neg)
            perturbations.append((W1_new_neg, b1_curr, W2_new_neg, b2_new_neg))
        elif model.activation == 'gelu':
            # No boundary scaling for smooth GELU
            W1_new_pos = W1_curr + total_perturbation
            h_new_pos = nnx.gelu(x_train @ W1_new_pos + b1_curr)
            W2_new_pos, b2_new_pos = apply_correction((W2_curr, b2_curr), (mu_old, std_old), h_new_pos)
            perturbations.append((W1_new_pos, b1_curr, W2_new_pos, b2_new_pos))
            
            W1_new_neg = W1_curr - total_perturbation
            h_new_neg = nnx.gelu(x_train @ W1_new_neg + b1_curr)
            W2_new_neg, b2_new_neg = apply_correction((W2_curr, b2_curr), (mu_old, std_old), h_new_neg)
            perturbations.append((W1_new_neg, b1_curr, W2_new_neg, b2_new_neg))
        # ------------------------------
        
    return Ensemble(model, perturbations)

def plot_results(model, ensemble, x_train, y_train, x_eval, x1, x2, save_path, title_prefix="PJSVD Ensemble"):
    print("Evaluating models over test grid...")
    
    # Base model decision boundary
    base_logits = model(x_eval)
    base_preds = jnp.argmax(base_logits, axis=-1).reshape(x1.shape)
    
    # Ensemble uncertainty
    ensemble_probs = ensemble.predict_proba(x_eval)
    mean_probs = jnp.mean(ensemble_probs, axis=0) 
    
    # Entropy: H(y|x) = -sum p(y|x) log2 p(y|x)
    entropy = -jnp.sum(mean_probs * jnp.log2(mean_probs + 1e-9), axis=-1).reshape(x1.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    c0 = (y_train == 0)
    c1 = (y_train == 1)
    
    # Plot 1: Decision Boundary
    ax1 = axes[0]
    ax1.contourf(x1, x2, base_preds, alpha=0.3, cmap='bwr')
    ax1.scatter(x_train[c0, 0], x_train[c0, 1], c='blue', s=40, edgecolors='k', label="Class 0")
    ax1.scatter(x_train[c1, 0], x_train[c1, 1], c='red', s=40, edgecolors='k', label="Class 1")
    ax1.set_title(f"Base Model Decision Boundary ({model.activation})")
    ax1.legend()
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    
    # Plot 2: Predictive Entropy
    ax2 = axes[1]
    contour = ax2.contourf(x1, x2, entropy, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label="Predictive Entropy (bits)")
    ax2.scatter(x_train[c0, 0], x_train[c0, 1], c='blue', s=40, edgecolors='w', marker='o')
    ax2.scatter(x_train[c1, 0], x_train[c1, 1], c='red', s=40, edgecolors='w', marker='^')
    ax2.set_title(f"{title_prefix} Predictive Entropy")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print(f"=== EXPERIMENT: Two Moons Classification with {MODE.upper()} ({ACTIVATION}) ===")
    
    x_train, y_train = generate_moons_data(n_samples=200, noise=0.1)
    x_eval, x1, x2 = get_eval_data(grid_size=80)
        
    if MODE == 'pjsvd':
        rngs = nnx.Rngs(0)
        model = ToyModel(in_features=2, out_features=2, rngs=rngs, activation=ACTIVATION)
        
        model = train_model(model, x_train, y_train, steps=5000)
        
        # For a 2D layer with 64 units, we have 128 weights. Max directions = 60.
        ensemble = run_pjsvd(model, x_train, n_directions=60, n_perturbations=1000, perturbation_size=6.0)
        
        plot_path = f"pjsvd_two_moons_{ACTIVATION}.png"
        plot_title_prefix = "PJSVD Ensemble"
    elif MODE == 'deep_ensemble':
        ensemble = train_deep_ensemble(x_train, y_train, n_models=ENSEMBLE_SIZE, steps=5000)
        model = ensemble.models[0] # Use first model as 'base' for plotting
        
        plot_path = f"deep_ensemble_two_moons_{ACTIVATION}.png"
        plot_title_prefix = "Deep Ensemble"
    else:
        raise ValueError(f"Unknown mode: {MODE}")
        
    plot_results(model, ensemble, x_train, y_train, x_eval, x1, x2, save_path=plot_path, title_prefix=plot_title_prefix)
