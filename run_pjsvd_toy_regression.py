import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import os
import matplotlib.pyplot as plt

from epitaph.pjsvd import find_optimal_perturbation, apply_correction

# Configuration
DIMENSIONS = 1  # 1 or 2
ACTIVATION = 'crelu' # 'crelu' or 'gelu'

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

def generate_toy_data(n_samples=50, dims=1):
    np.random.seed(42)
    if dims == 1:
        x = np.random.uniform(-4, 4, size=(n_samples, 1)).astype(np.float32)
        y = (x**3 + np.random.normal(0, 3, size=(n_samples, 1))).astype(np.float32)
        return x, y
    else:
        # x ~ U(-4, 4)^2
        x = np.random.uniform(-4, 4, size=(n_samples, 2)).astype(np.float32)
        # y = x1^3 + x2^3 + e
        y = (x[:, 0]**3 + x[:, 1]**3 + np.random.normal(0, 3, size=n_samples)).astype(np.float32).reshape(-1, 1)
        return x, y

def get_eval_data(grid_size=40, dims=1):
    if dims == 1:
        x = np.linspace(-8, 8, 200).reshape(-1, 1).astype(np.float32)
        y_true = (x**3).astype(np.float32)
        return x, y_true, None, None
    else:
        # x ~ uniform grid [-8, 8]^2
        grid = np.linspace(-8, 8, grid_size).astype(np.float32)
        x1, x2 = np.meshgrid(grid, grid)
        x = np.stack([x1.flatten(), x2.flatten()], axis=1)
        # noiseless y for reference
        y_true = (x[:, 0]**3 + x[:, 1]**3).astype(np.float32).reshape(-1, 1)
        return x, y_true, x1, x2

def train_model(model, inputs, targets, steps=3000, batch_size=20):
    print(f"Training on {len(inputs)} samples...")
    optimizer = optax.adam(1e-2)
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

    for i in range(steps):
        # Full batch for toy data
        loss, opt_state = train_step(model, opt_state, inputs, targets)
        if i % 500 == 0: 
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

    def predict(self, x):
        ys = []
        for w1, b1, w2, b2 in self.perturbations:
            y = self.manual_forward(x, w1, b1, w2, b2)
            ys.append(y)
        return jnp.stack(ys, axis=0)

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
    
    # We want the expected perturbation to be strictly zero so the ensemble mean
    # aligns with the base model. To do this perfectly, we can generate symmetric
    # pairs of perturbations (+z and -z).
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

def plot_results_1d(model, ensemble, x_train, y_train, x_eval, y_eval_true, save_path):
    print("Evaluating models over test grid...")
    base_preds = model(x_eval)
    ensemble_preds = ensemble.predict(x_eval)
    
    mean_preds = jnp.mean(ensemble_preds, axis=0).flatten()
    std_preds = jnp.std(ensemble_preds, axis=0).flatten()
    
    x_eval_flat = x_eval.flatten()
    base_preds_flat = base_preds.flatten()
    y_eval_true_flat = y_eval_true.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval_flat, y_eval_true_flat, 'k--', label="True Function ($y = x^3$)")
    plt.fill_between(x_eval_flat, 
                     mean_preds - 3*std_preds, 
                     mean_preds + 3*std_preds, 
                     color='blue', alpha=0.3, label="PJSVD Uncertainty ($\\pm 3\\sigma$)")
    plt.plot(x_eval_flat, mean_preds, 'b-', label="PJSVD Mean")
    plt.plot(x_eval_flat, base_preds_flat, 'g-', label="Base Model (MAP)")
    plt.scatter(x_train, y_train, c='red', s=50, label="Training Data", zorder=5)
    plt.axvline(-4, color='gray', linestyle=':')
    plt.axvline(4, color='gray', linestyle=':')
    plt.title(f"1D Toy Regression: PJSVD Uncertainty ({model.activation})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-400, 400)
    plt.xlim(-8, 8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def plot_results_2d(model, ensemble, x_train, y_train, x_eval, y_eval_true, x1, x2, save_path):
    print("Evaluating models over test grid...")
    ensemble_preds = ensemble.predict(x_eval)
    std_preds = jnp.std(ensemble_preds, axis=0).flatten()
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(x1, x2, std_preds.reshape(x1.shape), levels=30, cmap='viridis')
    plt.colorbar(contour, label="PJSVD Uncertainty (Standard Deviation)")
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', s=50, edgecolors='k', label="Training Data", zorder=5)
    plt.plot([-4, 4, 4, -4, -4], [-4, -4, 4, 4, -4], 'w--', linewidth=2, label="ID Boundary")
    plt.title(f"2D Toy Regression: PJSVD Uncertainty ({model.activation})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def plot_results(model, ensemble, x_train, y_train, x_eval, y_eval_true, x1, x2, save_path, dims=1):
    if dims == 1:
        plot_results_1d(model, ensemble, x_train, y_train, x_eval, y_eval_true, save_path)
    else:
        plot_results_2d(model, ensemble, x_train, y_train, x_eval, y_eval_true, x1, x2, save_path)

if __name__ == "__main__":
    print(f"=== EXPERIMENT: Toy Regression with PJSVD ({DIMENSIONS}D, {ACTIVATION}) ===")
    
    if DIMENSIONS == 1:
        x_train, y_train = generate_toy_data(n_samples=20, dims=DIMENSIONS)
        x_eval, y_eval_true, x1, x2 = get_eval_data(dims=DIMENSIONS)
        n_directions = 60
        perturbation_size = 5.0
    else:
        x_train, y_train = generate_toy_data(n_samples=50, dims=DIMENSIONS)
        x_eval, y_eval_true, x1, x2 = get_eval_data(grid_size=40, dims=DIMENSIONS)
        n_directions = 60
        perturbation_size = 5.0
        
    rngs = nnx.Rngs(0)
    model = ToyModel(in_features=DIMENSIONS, out_features=1, rngs=rngs, activation=ACTIVATION)
    
    model = train_model(model, x_train, y_train, steps=5000)
    
    ensemble = run_pjsvd(model, x_train, n_directions=n_directions, n_perturbations=100, perturbation_size=perturbation_size)
    
    plot_path = f"pjsvd_toy_regression_{DIMENSIONS}d_{ACTIVATION}.png"
    plot_results(model, ensemble, x_train, y_train, x_eval, y_eval_true, x1, x2, save_path=plot_path, dims=DIMENSIONS)
