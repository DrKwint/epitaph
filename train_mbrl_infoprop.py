import jax
import jax.numpy as jnp
from flax import nnx
import optax
import gymnasium as gym
import numpy as np
import os
import time
from tqdm import trange

from epitaph.sac import SACAgent
from epitaph.infoprop import infoprop_step
from epitaph.replay_buffer import ReplayBuffer
from run_pjsvd_experiment_2 import (
    PJSVDExperiment, StandardEnsemble, train_model, 
    train_standard_ensemble, TransitionModel
)

# Configuration
ENV_NAME = "HalfCheetah-v5"
TOTAL_STEPS = 50000
MODEL_TRAIN_FREQ = 2500  # Train model every N steps
MODEL_ROLLOUT_LENGTH = 1 # Start short (MBPO style)
ROLLOUT_BATCH_SIZE = 10_000
AGENT_UPDATES_PER_STEP = 20
REAL_RATIO = 0.05 # Ratio of real data in agent batch

# Model Params
USE_PJSVD = True 
N_MODELS = 5
N_DIRECTIONS = 20
N_PERTURBATIONS = 100 # For PJSVD size

def get_reward(env, state, action, next_state):
    # Reward Function
    # We will use the exact reward logic from HalfCheetah-v5.
    # Reward = (x_t+1 - x_t)/dt - 0.1 * |a|^2
    # But since we don't have x_position, we rely on the velocity which is usually in the observation.
    # HalfCheetah-v5 Observation Space:
    # 0: rootz
    # 1: rooty
    # ...
    # 8: rootx velocity (forward velocity)
    
    forward_vel = next_state[..., 8]
    ctrl_cost = 0.1 * jnp.sum(action**2, axis=-1)
    reward = forward_vel - ctrl_cost
    return reward

def collect_real_step(env, agent, buffer, state, rng_key):
    # Select action
    action = agent.select_action(state, rng_key, deterministic=False)
    
    # Step Env
    next_state, reward, terminated, truncated, _ = env.step(np.array(action))
    done = terminated or truncated
    
    # Add to buffer
    buffer.add(state, action, reward, next_state, float(done))
    
    return next_state, done

def train_model_on_buffer(buffer, use_pjsvd=True):
    s, a, r, sn, d = buffer.get_all()
    inputs = jnp.concatenate([s, a], axis=-1)
    targets = sn
    
    print(f"Training model on {len(inputs)} samples...")
    
    if use_pjsvd:
        pjsvd_exp = PJSVDExperiment(env_name=ENV_NAME, steps=len(inputs))
        # Inject data
        pjsvd_exp.inputs_id = inputs
        pjsvd_exp.targets_id = targets
        
        # Train Base Model
        rngs = nnx.Rngs(int(time.time()))
        pjsvd_exp.model = TransitionModel(inputs.shape[1], targets.shape[1], rngs)
        pjsvd_exp.model = train_model(pjsvd_exp.model, inputs, targets, steps=200) # Reduced for debugging
        
        # Perturb
        ensemble = pjsvd_exp.run_pjsvd(n_directions=N_DIRECTIONS, n_perturbations=N_PERTURBATIONS)
        return ensemble
    else:
        # Standard Ensemble
        return train_standard_ensemble(inputs, targets, n_models=N_MODELS, steps=2000)

def rollout_model(ensemble, real_buffer, agent, rollout_length, batch_size, rng_key):
    # Sample start states
    # We need to sample with replacement if batch_size > buffer size
    # ReplayBuffer.sample might strict on size.
    # We can just sample indices manually.
    
    indices = jax.random.randint(rng_key, (batch_size,), 0, len(real_buffer))
    # Create batch manually from buffer arrays (assuming they are accessible)
    # buffer.states is numpy array.
    # We can use get_all() and then index? Or just access valid range.
    
    # Efficient way:
    # Just use ReplayBuffer.sample if it supports replacement? 
    # Current impl of ReplayBuffer.sample:
    # idx = np.random.choice(self.size, size=batch_size, replace=False) -> Defaults to False usually?
    # Let's check the code:
    # if key is not None: idx = jax.random.choice(..., replace=False)
    # So we CANNOT use sample() for large batches.
    
    # Manual indexing:
    states = jnp.array(real_buffer.states[indices])
    
    s = states
    
    rollouts = {'s': [], 'a': [], 'r': [], 'sn': [], 'd': []}
    
    curr_s = s
    
    # Infoprop Variance Assumption
    fixed_var = 1e-3
    
    for t in range(rollout_length):
        # Action
        rng_key, act_key = jax.random.split(rng_key)
        action = agent.select_action(curr_s, act_key) # (Batch, A_dim) - need batch support in select_action
        # Actually SAC select_action expects (Dim,) or (Batch, Dim).
        # We need to verify if agent.actor handles batch. Yes, standard NNX Linear should.
        
        # Next State via Infoprop
        # 1. Inputs [s, a]
        model_inputs = jnp.concatenate([curr_s, action], axis=-1)
        
        # 2. Ensemble Preds
        means = ensemble.predict(model_inputs) # (E, B, D)
        vars = jnp.ones_like(means) * fixed_var
        
        # 3. TS Sample (Raw)
        n_models = means.shape[0]
        idx = jax.random.randint(rng_key, (batch_size,), 0, n_models)
        raw_sample = means[idx, jnp.arange(batch_size), :]
        
        # 4. Infoprop
        info = infoprop_step(means, vars, raw_sample)
        
        mu = info['mean']
        sigma = jnp.sqrt(info['var'])
        rng_key, noise_key = jax.random.split(rng_key)
        noise = jax.random.normal(noise_key, mu.shape)
        next_s = mu + sigma * noise
        
        # Reward
        # reward = get_reward(None, curr_s, action, next_s)
        # Using vectorized reward
        forward_vel = next_s[..., 8]
        ctrl_cost = 0.1 * jnp.sum(action**2, axis=-1)
        reward = forward_vel - ctrl_cost
        
        # Terminals (assume False for HalfCheetah unless time limit, which we handle in wrapper)
        done = jnp.zeros((batch_size,), dtype=jnp.float32)
        
        # Store
        rollouts['s'].append(curr_s)
        rollouts['a'].append(action)
        rollouts['r'].append(reward)
        rollouts['sn'].append(next_s)
        rollouts['d'].append(done)
        
        curr_s = next_s
        
    # Stack and flatten
    def flatten(x):
        return jnp.concatenate(x, axis=0)
        
    # rollout_model needs to return a dictionary of arrays
    # Fix: Ensure keys match buffer structure
    return {
        's': flatten(rollouts['s']),
        'a': flatten(rollouts['a']),
        'r': flatten(rollouts['r']),
        'sn': flatten(rollouts['sn']),
        'd': flatten(rollouts['d'])
    }

def main():
    # 1. Init Env & Agent
    env = gym.make(ENV_NAME)
    state, _ = env.reset(seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # JAX Key
    rng = nnx.Rngs(42)
    key = jax.random.PRNGKey(42)
    
    agent = SACAgent(state_dim, action_dim, rngs=rng)
    opt_states = agent.init_state()
    
    # Buffers
    real_buffer = ReplayBuffer(100000, state_dim, action_dim)
    model_buffer = ReplayBuffer(400000, state_dim, action_dim)
    
    # Model
    ensemble = None
    
    # 2. Pre-collect Seed Data
    print("Collecting seed data...")
    for _ in range(1000):
        key, step_key = jax.random.split(key)
        # Random action for seed
        action = env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        real_buffer.add(state, action, reward, next_state, float(term or trunc))
        state = next_state
        if term or trunc:
            state, _ = env.reset()
            
    # 3. Training Loop
    ep_rewards = []
    curr_ep_reward = 0
    state, _ = env.reset()
    
    start_time = time.time()
    
    for step in trange(TOTAL_STEPS):
        # A. Collect Real Step
        key, step_key = jax.random.split(key)
        state, done = collect_real_step(env, agent, real_buffer, state, step_key)
        
        # Track reward is tricky here since collect_real_step adds to buffer but 
        # doesn't return reward easily.
        # Actually I need to fix collect_real_step to return reward for logging.
        # But for now, let's assume we read from buffer tail.
        last_r = real_buffer.rewards[(real_buffer.idx - 1) % real_buffer.capacity]
        curr_ep_reward += last_r
        
        if done:
            ep_rewards.append(curr_ep_reward)
            print(f"Step {step}: Episode Reward = {curr_ep_reward:.2f}")
            curr_ep_reward = 0
            state, _ = env.reset()
        
        # B. Train Model
        if step % MODEL_TRAIN_FREQ == 0:
            print(f"--- Training Model at Step {step} ---")
            ensemble = train_model_on_buffer(real_buffer, use_pjsvd=USE_PJSVD)
            
            # C. Generate Model Rollouts
            key, roll_key = jax.random.split(key)
            rollout_data = rollout_model(
                ensemble, real_buffer, agent, 
                MODEL_ROLLOUT_LENGTH, ROLLOUT_BATCH_SIZE, roll_key
            )
            
            # Add to Model Buffer
            model_buffer.add_batch(
                rollout_data['s'], rollout_data['a'], rollout_data['r'],
                rollout_data['sn'], rollout_data['d']
            )
            print(f"Added {len(rollout_data['s'])} samples to Model Buffer.")
            
        # D. Train Agent
        # Only if we have model data
        if ensemble is not None:
            for _ in range(AGENT_UPDATES_PER_STEP):
                key, batch_key = jax.random.split(key)
                # Sample Mixed Batch
                batch_size = 256
                real_batch_size = int(batch_size * REAL_RATIO)
                model_batch_size = batch_size - real_batch_size
                
                r_s, r_a, r_r, r_sn, r_d = real_buffer.sample(real_batch_size, batch_key)
                
                key, batch_key = jax.random.split(key)
                m_s, m_a, m_r, m_sn, m_d = model_buffer.sample(model_batch_size, batch_key)
                
                # Concatenate
                b_s = jnp.concatenate([r_s, m_s])
                b_a = jnp.concatenate([r_a, m_a])
                b_r = jnp.concatenate([r_r, m_r])
                b_sn = jnp.concatenate([r_sn, m_sn])
                b_d = jnp.concatenate([r_d, m_d])
                
                batch = (b_s, b_a, b_r, b_sn, b_d)
                
                # Update
                key, update_key = jax.random.split(key)
                info, opt_states = agent.train_step(batch, update_key, opt_states)

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
