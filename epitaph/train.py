import jax
import jax.numpy as jnp
from flax import nnx
import optax  # type: ignore
import gymnasium as gym  # type: ignore
import numpy as np
from typing import Tuple, Any, Optional
from .models import Epinet
from .replay_buffer import ReplayBuffer
from .calibration import compute_gaussian_metrics, compute_coverage, compute_rmsce, plot_calibration_curve
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def collect_gym_data(
    steps: int, 
    env_name: str = "Pendulum-v1"
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Collects a batch of transitions from a Gymnasium environment.
    
    Returns:
        (states, actions, rewards, next_states, dones)
    """
    env = gym.make(env_name)
    s_list = []
    a_list = []
    r_list = []
    sn_list = []
    done_list = []

    steps_collected = 0
    obs, _ = env.reset()

    while steps_collected < steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        s_list.append(obs)
        a_list.append(action)
        r_list.append(reward)
        sn_list.append(next_obs)
        done_list.append(terminated or truncated)

        obs = next_obs
        steps_collected += 1

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    s = np.stack(s_list)
    a = np.stack(a_list)
    r = np.array(r_list)
    sn = np.stack(sn_list)
    dones = np.array(done_list)

    return jnp.array(s), jnp.array(a), jnp.array(r), jnp.array(sn), jnp.array(dones)


def train_epinet(
    model: Epinet,
    train_buffer: ReplayBuffer,
    val_buffer: ReplayBuffer,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    patience: int = 20,
    seed: int = 0,
    log_dir: Optional[str] = None,
    n_z_calibration: int = 100,
) -> Tuple[Epinet, dict[str, Any]]:
    """
    Trains the Epinet model using epoch-based training with early stopping.
    
    Args:
        model: Epinet instance to train
        train_buffer: ReplayBuffer with training data
        val_buffer: ReplayBuffer with validation data
        batch_size: Batch size for training
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        patience: Early stopping patience (epochs without improvement)
        seed: Random seed for reproducibility
        log_dir: Directory for TensorBoard logs (if None, no logging)
        
    Returns:
        (trained_model, training_history)
    """
    key = jax.random.PRNGKey(seed)
    z_dim = model.z_dim
    
    # Setup TensorBoard writer
    writer = None
    if log_dir is not None:
        if SummaryWriter is None:
            print("Warning: tensorboardX not installed, logging disabled")
        else:
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
    
    # Setup Optimizer (AdamW with weight decay)
    # CRITICAL: Exclude prior_enn parameters to keep it frozen
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Get all parameters except prior_enn
    all_params = nnx.state(model, nnx.Param)
    prior_params = nnx.state(model.prior_enn, nnx.Param)
    
    # Create a mask for trainable parameters
    trainable_params = nnx.State({
        k: v for k, v in all_params.items() 
        if not k.startswith('prior_enn')
    })
    
    opt_state = tx.init(trainable_params)

    @nnx.jit
    def train_step(model, opt_state, s, a, sn, z):
        def loss_fn(model):
            x = jnp.concatenate([s, a], axis=-1)
            pred = model(x, z)
            loss = jnp.mean((pred - sn) ** 2)
            return loss

        grad = nnx.grad(loss_fn)(model)
        
        # Only update trainable parameters (exclude prior_enn)
        trainable_grads = nnx.State({
            k: v for k, v in nnx.state(grad, nnx.Param).items()
            if not k.startswith('prior_enn')
        })
        
        trainable_params = nnx.State({
            k: v for k, v in nnx.state(model, nnx.Param).items()
            if not k.startswith('prior_enn')
        })
        
        updates, new_opt_state = tx.update(trainable_grads, opt_state, trainable_params)
        
        # Apply updates to parameters
        new_trainable_params = optax.apply_updates(trainable_params, updates)
        
        # Sanity check for NaNs in updates or new params
        def check_finite(tree):
            return jnp.all(jnp.isfinite(jax.tree_util.tree_leaves(tree)[0])) # Simple leaf check
        
        nnx.update(model, new_trainable_params)
        
        loss = loss_fn(model)
        return loss, new_opt_state
    
    @nnx.jit
    def eval_step(model, s, a, sn, z):
        """Compute validation loss without gradient updates."""
        x = jnp.concatenate([s, a], axis=-1)
        pred = model(x, z)
        loss = jnp.mean((pred - sn) ** 2)
        return loss

    # Training data setup
    n_train = len(train_buffer)
    n_val = len(val_buffer)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'stopped_early': False
    }
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    print("\nStarting epoch-based training...")
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}\n")

    for epoch in range(max_epochs):
        # Training epoch
        epoch_train_losses = []
        
        for s_batch, a_batch, r_batch, sn_batch, d_batch in train_buffer.iterate_batches(
            batch_size=batch_size, 
            shuffle=True, 
            seed=seed + epoch  # Different shuffle each epoch
        ):
            # Sample z for this batch
            key = jax.random.PRNGKey(seed + epoch * 1000 + len(epoch_train_losses))
            z_batch = jax.random.normal(key, (s_batch.shape[0], z_dim))
            
            # Training step
            train_loss, opt_state = train_step(model, opt_state, s_batch, a_batch, sn_batch, z_batch)
            epoch_train_losses.append(float(train_loss))
        
        # Average training loss for epoch
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        
        # Validation
        # Validation pass
        avg_val_loss = 0.0
        val_batches = 0
        val_key_base = seed + epoch * 2000 # Base key for validation z sampling
        for i, (s_b, a_b, r_b, sn_b, d_b) in enumerate(val_buffer.iterate_batches(
            batch_size=batch_size,
            shuffle=False
        )):
            z_b_key = jax.random.PRNGKey(val_key_base + i)
            z_b = jax.random.normal(z_b_key, (s_b.shape[0], z_dim))
            val_loss = eval_step(model, s_b, a_b, sn_b, z_b)
            if jnp.isnan(val_loss):
                print(f"   [WARNING] Validation loss is NaN in batch {i} at epoch {epoch+1}!")
            avg_val_loss += float(jnp.nan_to_num(val_loss, nan=1e6))
            val_batches += 1
        avg_val_loss /= max(1, val_batches)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
            writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
            writer.add_scalar('Loss/best_val', float(np.nan_to_num(best_val_loss, posinf=1e6)), epoch + 1)
            
            # --- Calibration Check ---
            s_val_all, a_val_all, _, sn_val_all, _ = val_buffer.get_all()
            x_val = jnp.concatenate([s_val_all, a_val_all], axis=-1)
            
            val_preds = []
            for i in range(n_z_calibration):
                z_val_sample = jax.random.normal(jax.random.PRNGKey(seed + epoch * 1000 + i), (x_val.shape[0], z_dim))
                pred = model(x_val, z_val_sample)
                val_preds.append(pred)
            
            val_preds = jnp.stack(val_preds)
            gauss_metrics = compute_gaussian_metrics(sn_val_all, val_preds)
            coverage = compute_coverage(sn_val_all, val_preds, quantiles=[0.95])
            rmsce = compute_rmsce(sn_val_all, val_preds)
            
            # Ensure we only log finite values to TB
            writer.add_scalar('Calibration/NLL', float(np.nan_to_num(gauss_metrics['nll'], nan=1e6)), epoch + 1)
            writer.add_scalar('Calibration/RMSE', float(np.nan_to_num(gauss_metrics['rmse'], nan=1e6)), epoch + 1)
            writer.add_scalar('Calibration/Avg_Var', float(np.nan_to_num(gauss_metrics['avg_var'], nan=0.0)), epoch + 1)
            writer.add_scalar('Calibration/PICP_95', float(np.nan_to_num(coverage['coverage_95'], nan=0.0)), epoch + 1)
            writer.add_scalar('Calibration/RMSCE', float(np.nan_to_num(rmsce, nan=1.0)), epoch + 1)
            
            # Log full calibration plot less frequently
            if (epoch + 1) % 50 == 0 or epoch == 0:
                fig = plot_calibration_curve(sn_val_all, val_preds)
                writer.add_figure('Calibration/Reliability_Diagram', fig, epoch + 1)
        
        if jnp.isnan(avg_train_loss) or jnp.isnan(avg_val_loss):
            print(f"   [CRITICAL] NaN detected in losses at epoch {epoch+1}. Aborting training.")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{max_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            history['best_epoch'] = epoch + 1
            # Save best model state
            best_model_state = nnx.state(model)
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")
            history['stopped_early'] = True
            # Restore best model
            if best_model_state is not None:
                nnx.update(model, best_model_state)
            break
    
    if not history['stopped_early']:
        print(f"\nTraining completed (max epochs reached)")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {history['best_epoch']}")
        # Restore best model
        if best_model_state is not None:
            nnx.update(model, best_model_state)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    return model, history
