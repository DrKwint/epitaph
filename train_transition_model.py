import gymnasium as gym
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax
import numpy as np
from typing import Tuple

# --- 1. Model Definition ---

class TransitionModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, width: int = 64, rngs: nnx.Rngs = None):
        self.layers = nnx.Sequential(
            nnx.Linear(in_features, width, rngs=rngs),
            nnx.relu,
            nnx.Linear(width, width, rngs=rngs),
            nnx.relu,
            nnx.Linear(width, out_features, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layers(x)
        
    def get_activations(self, x: jax.Array, binary=True) -> list[jax.Array]:
        patterns = []
        for layer in self.layers.layers:
            x = layer(x)
            if layer == nnx.relu:
                # x is post-relu. Pattern is x > 0.
                # Strictly speaking, if x was 0 pre-relu, it is 0 post-relu.
                # But we can just check x > 0.
                if binary:
                    patterns.append(x > 0)
                else:
                    patterns.append(x)
        return patterns

# --- 2. Data Collection ---

def collect_data(env_name: str, num_steps: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = gym.make(env_name)
    obs, _ = env.reset()
    
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        states.append(obs)
        actions.append(action)
        next_states.append(next_obs)
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    return np.array(states), np.array(actions), np.array(next_states)

def preprocess_data(states, actions, next_states):
    # Predict state difference (delta)
    deltas = next_states - states
    
    # Concatenate state and action for input
    inputs = np.concatenate([states, actions], axis=-1)
    
    # Basic normalization (mean/std)
    in_mean = inputs.mean(axis=0)
    in_std = inputs.std(axis=0) + 1e-6
    out_mean = deltas.mean(axis=0)
    out_std = deltas.std(axis=0) + 1e-6
    
    norm_inputs = (inputs - in_mean) / in_std
    norm_targets = (deltas - out_mean) / out_std
    
    return (
        jnp.array(norm_inputs), 
        jnp.array(norm_targets), 
        (in_mean, in_std, out_mean, out_std)
    )

# --- 3. Serialization Utils ---

def save_model(model, filename):
    state = nnx.state(model)
    # Convert State to pure dict for serialization
    state_dict = state.to_pure_dict()
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(state_dict))

def load_model(model, filename):
    with open(filename, 'rb') as f:
        state_bytes = f.read()
    
    # We need the structure/template
    current_state = nnx.state(model)
    template = current_state.to_pure_dict()
    
    new_state_dict = flax.serialization.from_bytes(template, state_bytes)
    # Update model with loaded state
    nnx.update(model, new_state_dict)

def save_data(inputs, targets, stats, filename):
    # stats is a tuple (in_mean, in_std, out_mean, out_std)
    np.savez(filename, inputs=inputs, targets=targets, 
             in_mean=stats[0], in_std=stats[1], 
             out_mean=stats[2], out_std=stats[3])

def load_data(filename):
    data = np.load(filename)
    inputs = data['inputs']
    targets = data['targets']
    stats = (data['in_mean'], data['in_std'], data['out_mean'], data['out_std'])
    return inputs, targets, stats

def normalize_data(states, actions, stats):
    # stats is (in_mean, in_std, out_mean, out_std)
    in_mean, in_std, _, _ = stats
    
    # Concatenate state and action for input
    inputs = np.concatenate([states, actions], axis=-1)
    
    # Optimize normalization
    norm_inputs = (inputs - in_mean) / in_std
    
    return jnp.array(norm_inputs)

# --- 4. Training Logic ---

def train():
    # Setup
    env_name = "InvertedPendulum-v5"
    num_steps = 10000
    batch_size = 64
    epochs = 50
    lr = 1e-3
    
    print(f"Collecting {num_steps} steps from {env_name}...")
    states, actions, next_states = collect_data(env_name, num_steps)
    inputs, targets, stats = preprocess_data(states, actions, next_states)
    
    print(f"Initializing model...")
    rngs = nnx.Rngs(0)
    model = TransitionModel(
        in_features=inputs.shape[-1], 
        out_features=targets.shape[-1], 
        rngs=rngs
    )
    
    # Optimizer Setup
    # Muon for 2D weights in hidden layers, AdamW for everything else.
    # Output layer is the last linear layer (index 4 in Sequential).
    
    def map_params(path, param):
        # path is a tuple of strings/ints/dicts keys.
        # nnx.State creates a nested structure.
        # We want Muon for: 
        #   - ndim == 2
        #   - NOT the output layer.
        
        is_2d = param.ndim == 2
        
        # Path analysis for nnx.Sequential with nnx.Linear
        # Typical path: ('layers', 'layers', 0, 'kernel')
        # Output layer is index 4.
        
        # Convert path elements to string to check for '4'
        path_str = str(path)
        is_output = '4' in path_str
        
        if is_2d and not is_output:
            return 'muon'
        return 'adam'

    # Create param labels
    params = nnx.state(model, nnx.Param)
    param_labels = jax.tree_util.tree_map_with_path(map_params, params)

    # Define optimizers
    muon_opt = optax.contrib.muon(learning_rate=lr, ns_steps=5)
    adam_opt = optax.adamw(learning_rate=lr)
    
    tx = optax.multi_transform(
        {
            'muon': muon_opt,
            'adam': adam_opt
        },
        param_labels
    )
    
    opt_state = tx.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)
        
        grads = nnx.grad(loss_fn)(model)
        
        params = nnx.state(model, nnx.Param)
        updates, opt_state = tx.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        
        return loss_fn(model), opt_state

    print("Starting training...")
    num_batches = len(inputs) // batch_size
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(inputs))
        inputs_shuffled = inputs[idx]
        targets_shuffled = targets[idx]
        
        epoch_loss = 0
        for i in range(num_batches):
            batch_x = inputs_shuffled[i*batch_size : (i+1)*batch_size]
            batch_y = targets_shuffled[i*batch_size : (i+1)*batch_size]
            loss, opt_state = train_step(model, opt_state, batch_x, batch_y)
            epoch_loss += loss
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.6f}")

    print("Training complete.")
    
    # Simple evaluation
    test_idx = 0
    test_input = inputs[test_idx:test_idx+1]
    test_target = targets[test_idx:test_idx+1]
    prediction = model(test_input)
    print(f"\nFinal Test Sample:")
    print(f"Target (norm): {test_target}")
    print(f"Pred   (norm): {prediction}")

    print(f"Pred   (norm): {prediction}")

    # --- Serialization Verification ---
    print("\n--- Testing Serialization ---")
    
    # Save
    save_model(model, "transition_model.msgpack")
    save_data(inputs, targets, stats, "transition_data.npz")
    print("Model and data saved.")
    
    # Load
    print("Loading model and data...")
    loaded_inputs, loaded_targets, loaded_stats = load_data("transition_data.npz")
    
    # Verify Data
    assert np.allclose(inputs, loaded_inputs), "Inputs mismatch!"
    assert np.allclose(targets, loaded_targets), "Targets mismatch!"
    print("Data verification passed.")
    
    # Load Model
    # Create a new model instance to load into
    new_model = TransitionModel(
        in_features=loaded_inputs.shape[-1], 
        out_features=loaded_targets.shape[-1], 
        rngs=nnx.Rngs(0)
    )
    load_model(new_model, "transition_model.msgpack")
    
    # Verify Model Predictions
    new_pred = new_model(test_input)
    assert np.allclose(prediction, new_pred), "Model prediction mismatch!"
    print("Model verification passed.")
    
    # --- Verify normalize_data ---
    print("\n--- Testing normalization data ---")
    states_test, actions_test = loaded_inputs[:, :loaded_inputs.shape[-1]-1], loaded_inputs[:, loaded_inputs.shape[-1]-1:]
    # Because loaded_inputs is already normalized, we want to check if normalize_data produces same output *given original states*
    # However we didn't save original states.
    # We can check with `test_idx` sample from memory if we haven't overwritten them.
    
    # Let's use the memory variables from collect_data
    # `states`, `actions` are the raw numpy arrays from collect_data
    
    # We normalized `inputs` using `stats`. 
    # normalize_data(states, actions, stats) should equal `inputs` (approximately due to numpy/jax precision)
    
    recalc_inputs = normalize_data(states, actions, stats)
    assert np.allclose(inputs, recalc_inputs, atol=1e-5), "normalize_data mismatch!"
    print("normalize_data verification passed.")
    
    # --- Verify get_activation_patterns ---
    print("\n--- Testing Activation Patterns ---")
    patterns = new_model.get_activations(test_input)
    # Model has 2 hidden layers with ReLU
    assert len(patterns) == 2, f"Expected 2 patterns, got {len(patterns)}"
    for i, p in enumerate(patterns):
        print(f"Layer {i} pattern shape: {p.shape}, sparsity: {1.0 - p.mean():.2f}")
    print("Activation patterns verification passed.")

if __name__ == "__main__":
    train()

