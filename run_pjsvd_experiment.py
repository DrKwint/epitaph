import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np
import sys
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from epitaph import pjsvd
from typing import Tuple

# --- 1. Data Collection ---
# --- 1. Data Collection ---
def positive_policy(env, obs):
    # Sample positive actions [0, 2]
    return np.array([np.random.uniform(0.0, 1.0)], dtype=np.float32)

def negative_policy(env, obs):
    # Sample negative actions [-2, 0]
    return np.array([np.random.uniform(-1.0, 0.0)], dtype=np.float32)

def zero_policy(env, obs):
    # Sample zero actions [0, 0]
    return np.array([0.0], dtype=np.float32)

def random_policy(env, obs):
    # Sample random actions [-1, 1]
    return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

def collect_data(env_name: str, steps: int, policy_fn=positive_policy, seed: int = 0):
    print(f"Collecting {steps} steps from {env_name} with {policy_fn.__name__}...", flush=True)
    try:
        env = gym.make(env_name)
    except gym.error.NameNotFound:
        print(f"Warning: {env_name} not found. Trying Pendulum-v1")
        env = gym.make("Pendulum-v1")
        
    inputs = []
    targets = []
    
    obs, _ = env.reset(seed=seed)
    for i in range(steps):
        if i % 1000 == 0:
            print(f"Step {i}", flush=True)
            
        action = policy_fn(env, obs)
        # Ensure action is correct shape/type for Env
        if isinstance(action, (float, int)):
             action = np.array([action], dtype=np.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Transition Model: (s, a) -> s'
        # Input: [s, a]
        # Target: s' (or delta s)
        
        inputs.append(np.concatenate([obs, action]))
        targets.append(next_obs)
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    return jnp.array(np.stack(inputs)), jnp.array(np.stack(targets))

# --- 2. Simple Transition Model ---
class TransitionModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        # 2 Hidden Layers as requested
        self.l1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, out_features, rngs=rngs)
        self.relu = nnx.relu

    def __call__(self, x):
        h1 = self.relu(self.l1(x))
        h2 = self.relu(self.l2(h1))
        out = self.l3(h2)
        return out

def train_model(model, inputs, targets, steps=1000, batch_size=32, lr=1e-3):
    print(f"Training model for {steps} steps...")
    optimizer = optax.adam(lr)
    # Use standard nnx training pattern
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, batch_in, batch_out):
        def loss_fn(model):
            pred = model(batch_in)
            return jnp.mean((pred - batch_out) ** 2)
            
        grads = nnx.grad(loss_fn)(model)
        
        # Update
        updates, new_opt_state = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        
        return loss_fn(model), new_opt_state

    # Simple loop
    indices = np.arange(inputs.shape[0])
    losses = []
    
    for i in range(steps):
        batch_idx = np.random.choice(indices, batch_size)
        loss, opt_state = train_step(model, opt_state, inputs[batch_idx], targets[batch_idx])
        if i % 100 == 0:
            losses.append(loss)
            
    print(f"Final Loss: {losses[-1]}")
    return model

def run_experiment():
    key = jax.random.PRNGKey(42)
    
    # A. Collect Data
    # 2000 steps for Pendulum-v1 (Positive Policy)
    inputs, targets = collect_data("Pendulum-v1", 10000, policy_fn=random_policy)
    
    # B. Train Model
    rngs = nnx.Rngs(0)
    model = TransitionModel(inputs.shape[1], targets.shape[1], rngs)
    model = train_model(model, inputs, targets)
    
    # C. Apply PJSVD
    print("\n--- Applying PJSVD ---", flush=True)
    
    # We will perturb Layer 1 (model.l1)
    # Note: PJSVD needs redundant degrees of freedom.
    # InvertedPendulum: State=4, Action=1 -> Input=5.
    # Layer 1 is 5 -> 64. 
    # Batch size for PJSVD: Let's use a subset of data as "ID" data.
    # If we use *all* data, optimization might find it hard to be redundant?
    # Actually, 5 -> 64 is "expanding".
    # W is [5, 64]. 
    # We want v (perturbation of W) such that X @ v is small?
    # W is [In, Out]. v is [In, Out].
    # X @ v approx 0.
    
    # Let's use a subset of 100 data points as our "ID Constraints"
    id_subset_idx = np.random.choice(len(inputs), 1000, replace=False)
    X_id = inputs[id_subset_idx]
    
    # Define model function for perturbation finding
    # We need access to just the linear part of L1? 
    # Or the whole model? PJSVD minimizes residuals of "model_fn".
    # Typically this is the L1 output (pre-activation or post-activation).
    # Since we correct L2, let's use L1 output (post-Relu).
    
    def model_fn_l1(w):
        # We need to manually inject w into the layer logic
        # model.l1.kernel is what we perturb
        
        # Recreate L1 logic with explicit w
        # L1 is Linear(5, 64). Bias is fixed (not perturbed here, though PJSVD can handle it)
        # To reuse model structure easily is hard with functional style unless we do surgery.
        # But we can just replicate the math of L1:
        
        # But for 'find_optimal_perturbation', we return the output of the layer.
        pre = X_id @ w + model.l1.bias.get_value()
        return nnx.relu(pre)

    print("Finding optimal perturbation for L1...", flush=True)
    W1_current = model.l1.kernel.get_value()
    print(f"W1 shape: {W1_current.shape}, X_id shape: {X_id.shape}", flush=True)
    print(f"W1 shape: {W1_current.shape}, X_id shape: {X_id.shape}", flush=True)
    
    # Test model_fn_l1 once
    try:
        dummy_out = model_fn_l1(W1_current)
        print(f"model_fn_l1 check - Output shape: {dummy_out.shape}", flush=True)
    except Exception as e:
        print(f"model_fn_l1 failed: {e}", flush=True)
    
    v, sigma = pjsvd.find_optimal_perturbation(model_fn_l1, W1_current, iterations=500)
    print("Optimization returned.", flush=True)
    
    v = v / jnp.linalg.norm(v)
    print(f"Singular Value (Residual Norm): {sigma}")
    
    # Perturb
    epsilon = 0.5 # Larger perturbation to verify robustness
    W1_new = W1_current + epsilon * v
    print(f"Applied perturbation with epsilon={epsilon}")
    
    # 4. Apply Correction to Next Layer (L2)
    # L2 takes L1 output as input.
    # We need statistics of L1 output on the ID data.
    h_old = model_fn_l1(W1_current)
    mu_old = jnp.mean(h_old, axis=0)
    std_old = jnp.std(h_old, axis=0)
    
    # New output
    h_new = model_fn_l1(W1_new)
    
    h_new = model_fn_l1(W1_new)
    
    # Next layer params: model.l2.kernel, model.l2.bias
    W2_current = model.l2.kernel.get_value()
    b2_current = model.l2.bias.get_value()
    
    print("Applying correction to L2...")
    W2_new, b2_new = pjsvd.apply_correction(
        (W2_current, b2_current),
        (mu_old, std_old),
        h_new
    )
    
    # Update model with new weights for verification
    # We'll create a "perturbed" model copy conceptually or just measure
    
    # 5. Measure Residuals / Performance
    
    # Helper to run full forward pass with patched weights
    def forward_manual(x, w1, b1, w2, b2, w3, b3):
        h1 = nnx.relu(x @ w1 + b1)
        h2 = nnx.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3
        return out
        
    # ID Fidelity
    y_id_orig = model(X_id)
    y_id_new = forward_manual(
        X_id, 
        W1_new, model.l1.bias.get_value(), 
        W2_new, b2_new, 
        model.l3.kernel.get_value(), model.l3.bias.get_value()
    )
    
    diff_id = jnp.linalg.norm(y_id_new - y_id_orig) / len(X_id)
    print(f"ID Output Difference (per sample approx): {diff_id}")
    
    # Uncorrected residual (if we didn't correct L2)
    y_id_uncorrected = forward_manual(
        X_id,
        W1_new, model.l1.bias.get_value(),
        W2_current, b2_current, # Old L2
        model.l3.kernel.get_value(), model.l3.bias.get_value()
    )
    diff_uncorrected = jnp.linalg.norm(y_id_uncorrected - y_id_orig) / len(X_id)
    print(f"ID Uncorrected Difference: {diff_uncorrected}")
    
    print(f"Correction Improvement Factor: {diff_uncorrected / diff_id:.2f}x")
    
    # 6. Similar data Sensitivity
    # Generate OOD data by using a similar policy
    print("\ncollecting similar data (random policy)...", flush=true)
    inputs_ood, _ = collect_data("pendulum-v1", 10000, policy_fn=random_policy, seed=999)
    x_ood = inputs_ood

    y_ood_orig = model(x_ood)
    y_ood_new = forward_manual(
        x_ood, 
        w1_new, model.l1.bias.get_value(), 
        w2_new, b2_new, 
        model.l3.kernel.get_value(), model.l3.bias.get_value()
    )
    
    diff_ood = jnp.linalg.norm(y_ood_new - y_ood_orig) / len(x_ood)
    print(f"similar output difference: {diff_ood}")
    
    print(f"ratio similar/id sensitivity: {diff_ood / diff_id:.2f}x", flush=true)

    # 6. OOD Sensitivity
    # Generate OOD data by using a DIFFERENT policy (Negative policy)
    print("\ncollecting ood data (zero policy)...", flush=true)
    inputs_ood, _ = collect_data("pendulum-v1", 10000, policy_fn=zero, seed=999)
    x_ood = inputs_ood

    y_ood_orig = model(x_ood)
    y_ood_new = forward_manual(
        x_ood, 
        w1_new, model.l1.bias.get_value(), 
        w2_new, b2_new, 
        model.l3.kernel.get_value(), model.l3.bias.get_value()
    )
    
    diff_ood = jnp.linalg.norm(y_ood_new - y_ood_orig) / len(x_ood)
    print(f"ood output difference: {diff_ood}")
    
    print(f"ratio ood/id sensitivity: {diff_ood / diff_id:.2f}x", flush=true)


if __name__ == "__main__":
    run_experiment()
