from epitaph.safety import estimate_safety_mc, estimate_reward_mc
from epitaph.safety import collect_iterative_safety_constraints
from epitaph.control import TrialController
import jax
import jax.numpy as jnp
from flax import nnx
import gymnasium as gym
from epitaph.models import Epinet
from epitaph.control import MPPIController
from epitaph.train import train_epinet, collect_gym_data
from epitaph.replay_buffer import ReplayBuffer
import time
import os
import csv
from tensorboardX import SummaryWriter
from tqdm import trange

def main():
    print("=" * 60)
    print("Safe MPPI Control Evaluation")
    print("=" * 60)

    # 1. Environment and Constants
    env_name = "InvertedPendulum-v5"
    log_dir = f"logs/{env_name}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    csv_file = open(os.path.join(log_dir, "rewards.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "reward", "cumulative_reward", "episode_total"])
    
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset(seed=454)

    in_dim = obs.shape[0]
    out_dim = obs.shape[0] # state transition
    action_dim = env.action_space.shape[0]
    
    # 2. Setup Model (Assuming we train it first or use current state)
    # For evaluation, we'll train a quick model on random data first
    z_dim = 4
    model = Epinet(in_dim + action_dim, out_dim, z_dim, rngs=nnx.Rngs(432))
    
    print("1. Collecting initial transitions for quick training...")
    buffer = ReplayBuffer(capacity=1000, state_dim=in_dim, action_dim=action_dim)
    _s, _a, _r, _sn, _d = collect_gym_data(steps=200, env_name=env_name)
    buffer.add_batch(_s, _a, _r, _sn, _d)
    
    print("2. Training Epinet model...")
    # Small training for demo
    train_buffer, val_buffer = buffer.split(0.8)
    model, history = train_epinet(
        model, train_buffer, val_buffer, 
        max_epochs=1000, 
        batch_size=int(jnp.sqrt(len(train_buffer))),
        log_dir=os.path.join(log_dir, "init_train")
    )
    print(f"   Training complete. Best Val Loss: {min(history['val_loss']):.6f}")

    # 3. Setup Environment-Specific Configs
    def hopper_reward(s, u):
        z = s[0]
        angle = s[1]
        state = s[1:] 
        
        # 1. Healthy Reward
        is_healthy = (z > 0.7) & (jnp.abs(angle) < 0.2) & jnp.all(jnp.abs(state) < 100.0)
        healthy_reward = jnp.where(is_healthy, 1.0, 0.0)
        
        # 2. Forward Reward (x_velocity is at s[5])
        forward_reward = 1.0 * s[5] 
        
        # 3. Control Cost
        ctrl_cost = 0.001 * jnp.sum(u**2)
        
        return forward_reward + healthy_reward - ctrl_cost

    def inverted_pendulum_reward(s, u):
        is_healthy = jnp.logical_and(jnp.abs(s[1]) < 0.2, jnp.all(jnp.isfinite(s)))
        healthy_reward = jnp.where(is_healthy, 1.0, 0.0)
        return healthy_reward

    ENV_CONFIGS = {
        "InvertedPendulum-v5": {
            "H": jnp.array([
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0]
            ]),
            "d": jnp.array([0.2, 0.2]),
            "reward_fn": inverted_pendulum_reward
        },
        "Hopper-v5": {
            "H": jnp.zeros((3, in_dim)).at[0, 0].set(-1.0).at[1, 1].set(1.0).at[2, 1].set(-1.0),
            "d": jnp.array([-0.7, 0.2, 0.2]),
            "reward_fn": hopper_reward
        },
        "HalfCheetah-v5": {
            # Dummy safety: limit joint velocities/angles to some large range
            "H": jnp.zeros((0, in_dim)),
            "d": jnp.zeros((0,)),
            "reward_fn": lambda s, u: s[8] - 0.1 * jnp.sum(u**2)
        }
    }

    config = ENV_CONFIGS.get(env_name, {
        "H": jnp.zeros((0, in_dim)),
        "d": jnp.zeros((0,)),
        "reward_fn": lambda s, u: 0.0
    })
    
    H, d = config["H"], config["d"]
    reward_fn = config["reward_fn"]

    # controller = TrialController(
        # model=model,
        # action_space=env.action_space,
        # horizon=15,
        # n_samples=100,
        # temperature=0.1,
        # safety_weight=50.0, # High weight for safety
        # noise_sigma=0.5
    # )

    # 4. Control Loop
    print("3. Starting Control Loop...")
    main_key = jax.random.PRNGKey(123)
   
    horizon = 25
    resets = 0
    curr_episode_reward = 0
    total_reward = 0
    
    # Initialize action sequence for warm start
    action_seq = jnp.zeros((horizon, env.action_space.shape[0]))
    
    for step in range(2000):
        # Split key for this control step
        main_key, step_key, z_key = jax.random.split(main_key, 3)
        
        # MPPI Inner Loop
        for mppi_iter in range(5):
            # Sample noise with dynamic key
            iter_key = jax.random.fold_in(step_key, mppi_iter)
            noise = jax.random.normal(iter_key, (500, horizon, env.action_space.shape[0])) * 0.5
            trial_seqs = action_seq + noise

            # 1. Estimate Expected Cumulative Reward
            weights = jax.vmap(estimate_reward_mc, in_axes=(None, 0, None, None, None, None, None, None))(
                obs, trial_seqs, H, d, model, reward_fn, 500, z_key
            )

            # Normalize
            weight_sum = jnp.sum(weights) + 1e-10
            action_seq = jnp.sum(jnp.expand_dims(weights, [-1, -2]) * trial_seqs, axis=0) / weight_sum

        action = trial_seqs[jnp.argmax(weights)][0]
        obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, obs, terminated or truncated)
        env.render()
        
        curr_episode_reward += reward
        total_reward += reward
        
        # Logging
        writer.add_scalar("reward/step", reward, step)
        writer.add_scalar("reward/cumulative", total_reward, step)
        writer.flush()
        csv_writer.writerow([step, reward, total_reward, ""])
        csv_file.flush()
        
        if terminated or truncated:
            resets += 1
            writer.add_scalar("reward/episode_total", curr_episode_reward, resets)
            writer.flush()
            print(f"\nEpisode {resets} Reward: {curr_episode_reward:.3f}")
            
            # Update CSV with episode total
            csv_writer.writerow([step, reward, total_reward, curr_episode_reward])
            csv_file.flush()
            
            curr_episode_reward = 0
            obs, _ = env.reset()
            # Reset action sequence on environment reset
            action_seq = jnp.zeros((horizon, env.action_space.shape[0]))
        else:
            # MPPI Warm Start: Shift the sequence for the next step
            action_seq = jnp.roll(action_seq, -1, axis=0)
            action_seq = action_seq.at[-1].set(0.0)
        
        jnp.set_printoptions(precision=3)
        print(f"\nStep {step} | Resets: {resets} | Action: {action} | Horizon: {horizon} | Best Reward: {weights.max():.3f}")
    
        if (step + 1) % 20 == 0:
            train_buffer, val_buffer = buffer.split(0.8)
            # Diagnostic: Check for NaNs/Infs in buffer
            _s, _a, _r, _sn, _d = train_buffer.get_all()
            if not jnp.all(jnp.isfinite(_s)) or not jnp.all(jnp.isfinite(_sn)):
                print(f"\n[CRITICAL] NaN or Inf detected in buffer at step {step}!")
                # Find which dimensions are problematic
                bad_s = jnp.where(~jnp.isfinite(_s))[1]
                if len(bad_s) > 0:
                    print(f"Problematic state dims: {jnp.unique(bad_s)}")
            
            # Re-initialize model before training
            model_key = jax.random.fold_in(jax.random.PRNGKey(432), step)
            model = Epinet(in_dim + action_dim, out_dim, z_dim, rngs=nnx.Rngs(model_key))
            
            model, history = train_epinet(
                model, train_buffer, val_buffer, 
                max_epochs=1000, 
                batch_size=64,
                log_dir=os.path.join(log_dir, f"train_step_{step+1}")
            )
            print(f"Training complete. Best Val Loss: {min(history['val_loss']):.6f}")

    writer.close()
    csv_file.close()


if __name__ == "__main__":
    main()
