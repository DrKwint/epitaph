"""
Main training script for Epinet model.
"""
import jax
import jax.numpy as jnp
from flax import nnx

from epitaph.models import Epinet
from epitaph.train import train_epinet, collect_gym_data
from epitaph.replay_buffer import ReplayBuffer
from epitaph.calibration import compute_gaussian_metrics, compute_coverage, compute_rmsce


import traceback

def main():
    """Main training pipeline with replay buffer."""
    print("=" * 60)
    print("Epinet Training Pipeline")
    print("=" * 60)
    
    # Configuration
    env_name = "InvertedPendulum-v5"
    buffer_capacity = 200
    train_split = 0.8
    
    # Model hyperparameters
    z_dim = 4
    base_width = 64
    base_depth = 2
    enn_width = 64
    enn_depth = 1
    
    # Training hyperparameters
    batch_size = 32
    patience = 100
    learning_rate = 3e-4
    weight_decay = 1e-6
    seed = 999
    
    # Step 1: Collect data
    print(f"\n1. Collecting {buffer_capacity} transitions from {env_name}...")
    s_all, a_all, r_all, sn_all, dones_all = collect_gym_data(buffer_capacity, env_name=env_name)
    
    n_total = s_all.shape[0]
    n_state = s_all.shape[1]
    n_action = a_all.shape[1]
    
    print(f"   State dim: {n_state}, Action dim: {n_action}")
    print(f"   Collected: {n_total} transitions")
    print(f"   Episode terminations: {jnp.sum(dones_all)}")
    
    # Data Validation
    for name, arr in [("states", s_all), ("actions", a_all), ("next_states", sn_all)]:
        nan_count = jnp.isnan(arr).sum()
        inf_count = jnp.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   [WARNING] CRITICAL: Found {nan_count} NaNs and {inf_count} Infs in {name}!")
            # Basic cleanup if necessary, or just fail early
        else:
            print(f"   [OK] {name} data is clean (no NaNs/Infs)")
    
    # Step 2: Create replay buffer and populate
    print(f"\n2. Creating replay buffer (capacity: {buffer_capacity})...")
    buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=n_state, action_dim=n_action)
    buffer.add_batch(s_all, a_all, r_all, sn_all, dones_all)
    
    print(f"   Buffer size: {len(buffer)}/{buffer.capacity}")
    print(f"   Buffer full: {buffer.is_full()}")
    
    # Step 3: Split into train/val using buffer.split()
    print(f"\n3. Splitting data (train: {train_split*100:.0f}%, val: {(1-train_split)*100:.0f}%)...")
    train_buffer, val_buffer = buffer.split(split_ratio=train_split, shuffle=True, seed=seed)
    
    print(f"   Train: {len(train_buffer)} samples")
    print(f"   Val: {len(val_buffer)} samples")
    
    # Step 4: Create model
    print(f"\n4. Creating Epinet model...")
    n_in_features = n_state + n_action
    n_out = n_state
    
    model = Epinet(
        n_in_features,
        n_out,
        z_dim=z_dim,
        base_width=base_width,
        base_depth=base_depth,
        enn_width=enn_width,
        enn_depth=enn_depth,
        rngs=nnx.Rngs(params=seed)
    )
    
    print(f"   Input: {n_in_features}, Output: {n_out}, Z-dim: {z_dim}")
    print(f"   Base: {base_depth} layers x {base_width} units")
    print(f"   ENN: {enn_depth} layers x {enn_width} units")
    
    # Step 5: Train with early stopping
    print(f"\n5. Training model (epochs)...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: 50")
    print(f"   Patience: 10")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    
    trained_model, history = train_epinet(
        model=model,
        train_buffer=train_buffer,
        val_buffer=val_buffer,
        batch_size=batch_size,
        max_epochs=1000,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        log_dir="runs/epinet_training_2h"
    )
    print(list(history.keys()))
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  - Validation checkpoints: {len(history['val_loss'])}")
    print(f"  - Best epoch: {history['best_epoch']}")
    print(f"  - Early stopping: {history['stopped_early']}")
    print(f"  - Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  - Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Step 7: Final Calibration Check
    print(f"\n7. Performing final uncertainty calibration check...")
    # Sample more z for the final check
    n_z_final = 100
    s_val, a_val, r_val, sn_val, done_val = val_buffer.get_all()
    x_val = jnp.concatenate([s_val, a_val], axis=-1)
    
    val_preds = []
    print(f"   Sampling {n_z_final} epistemic indices...")
    for i in range(n_z_final):
        z_val_sample = jax.random.normal(jax.random.PRNGKey(i + 10000), (x_val.shape[0], z_dim))
        pred = trained_model(x_val, z_val_sample)
        val_preds.append(pred)
    
    val_preds = jnp.stack(val_preds)
    
    gauss_metrics = compute_gaussian_metrics(sn_val, val_preds)
    coverage = compute_coverage(sn_val, val_preds, quantiles=[0.50, 0.90, 0.95])
    rmsce = compute_rmsce(sn_val, val_preds)
    
    print(f"   Final results:")
    print(f"     - RMSE: {gauss_metrics['rmse']:.6f}")
    print(f"     - NLL:  {gauss_metrics['nll']:.6f}")
    print(f"     - Coverage (95% CI): {coverage['coverage_95']*100:.1f}%")
    print(f"     - RMSCE: {rmsce:.6f}")
    
    print("\nCalibration check complete. Use TensorBoard to see epoch-wise progress.")
    print("Done!")
    return trained_model, history, buffer


if __name__ == "__main__":
    try:
        trained_model, history, buffer = main()
    except Exception:
        traceback.print_exc()
