import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Tuple

class Scalar(nnx.Module):
    def __init__(self, value: float, rngs: nnx.Rngs = None):
        self.value = nnx.Param(jnp.array(value, dtype=jnp.float32))

    def __call__(self):
        return self.value

class Critic(nnx.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, rngs: nnx.Rngs = None):
        self.l1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.l3 = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.relu = nnx.relu

    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

class DoubleCritic(nnx.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, rngs: nnx.Rngs = None):
        self.q1 = Critic(state_dim + action_dim, hidden_dim, rngs=rngs)
        self.q2 = Critic(state_dim + action_dim, hidden_dim, rngs=rngs)

    def __call__(self, s, a):
        return self.q1(s, a), self.q2(s, a)

class Actor(nnx.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, rngs: nnx.Rngs = None):
        self.l1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mu_layer = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std_layer = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.relu = nnx.relu
        self.action_scale = 1.0 # Assuming normalized actions [-1, 1]
        self.action_bias = 0.0

    def __call__(self, s):
        x = self.relu(self.l1(s))
        x = self.relu(self.l2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mu, log_std

    def sample(self, s, rng_key):
        mu, log_std = self(s)
        std = jnp.exp(log_std)
        
        # Reparameterization trick
        eps = jax.random.normal(rng_key, shape=mu.shape)
        x_t = mu + std * eps
        y_t = jnp.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Enforcing Action Bound
        log_prob = jax.scipy.stats.norm.logpdf(x_t, mu, std)
        # Correction for Tanh squashing
        log_prob -= 2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t))
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
        
        return action, log_prob

class SACAgent(nnx.Module):
    def __init__(self, state_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.actor = Actor(state_dim, action_dim, rngs=rngs)
        self.critic = DoubleCritic(state_dim, action_dim, rngs=rngs)
        # Target entropy usually -dim(A)
        self.target_entropy = -float(action_dim)
        # Alpha (Temperature) stored as log_alpha for stability
        self.log_alpha = Scalar(0.0, rngs=rngs) # Initial alpha = exp(0) = 1.0
        
        # Optimizers
        self.actor_optimizer = optax.adam(3e-4)
        self.critic_optimizer = optax.adam(3e-4)
        self.alpha_optimizer = optax.adam(3e-4)
        
        # Target Critic (Polyaks updated manually)
        self.target_critic = DoubleCritic(state_dim, action_dim, rngs=rngs)
        
        # Initialize target to same weights
        self.update_target(tau=1.0) # Hard copy

    def init_state(self):
        actor_params = nnx.state(self.actor, nnx.Param)
        critic_params = nnx.state(self.critic, nnx.Param)
        alpha_params = nnx.state(self.log_alpha, nnx.Param)
        
        return {
            'actor_opt': self.actor_optimizer.init(actor_params),
            'critic_opt': self.critic_optimizer.init(critic_params),
            'alpha_opt': self.alpha_optimizer.init(alpha_params)
        }

    def update_target(self, tau=0.005):
        # Soft update: target = tau * source + (1 - tau) * target
        source_state = nnx.state(self.critic, nnx.Param)
        target_state = nnx.state(self.target_critic, nnx.Param)
        
        new_target_state = jax.tree_util.tree_map(
            lambda s, t: tau * s + (1 - tau) * t,
            source_state, target_state
        )
        nnx.update(self.target_critic, new_target_state)

    def select_action(self, obs, rng_key, deterministic=False):
        # obs: (Batch, Dim) or (Dim,)
        is_batch = obs.ndim > 1
        if not is_batch:
            obs = obs[None, :]
            
        if deterministic:
            mu, _ = self.actor(obs)
            action = jnp.tanh(mu)
        else:
            action, _ = self.actor.sample(obs, rng_key)
        
        if is_batch:
            return action
        else:
            return action[0]

    @nnx.jit
    def train_step(self, batch, rng_key, opt_states):
        """
        Batch: (s, a, r, sn, d)
        opt_states: dict of optimizer states
        """
        states, actions, rewards, next_states, dones = batch
        rewards = rewards[:, None]
        dones = dones[:, None]
        
        # Split key for actor sampling
        rng_key, actor_key = jax.random.split(rng_key)
        
        # --- 1. Critic Update ---
        current_alpha = jnp.exp(self.log_alpha())
        
        # Target Q calculation
        # Sample next action
        next_actions, next_log_probs = self.actor.sample(next_states, actor_key)
        
        q1_target, q2_target = self.target_critic(next_states, next_actions)
        min_q_target = jnp.minimum(q1_target, q2_target)
        
        # Soft Q target
        q_target = rewards + (1.0 - dones) * 0.99 * (min_q_target - current_alpha * next_log_probs)
        
        def critic_loss_fn(critic):
            q1, q2 = critic(states, actions)
            loss1 = jnp.mean((q1 - q_target)**2)
            loss2 = jnp.mean((q2 - q_target)**2)
            return loss1 + loss2
            
        critic_grads = nnx.grad(critic_loss_fn)(self.critic)
        critic_updates, new_critic_opt_state = self.critic_optimizer.update(
            critic_grads, 
            opt_states['critic_opt'], 
            nnx.state(self.critic, nnx.Param)
        )
        nnx.update(self.critic, optax.apply_updates(nnx.state(self.critic, nnx.Param), critic_updates))
        
        # --- 2. Actor Update ---
        # Reuse current_alpha (detached)
        
        def actor_loss_fn(actor):
            # Re-sample actions for current states
            # We need a new key? No, in SGD we usually resample.
            # But inside JIT function we need fixed graph unless we pass key.
            # We use the SAME actor_key as above? No, independent sample.
            # Let's verify: In SAC, we update Critic, then Actor.
            # Actor loss depends on Q(s, pi(s)).
            
            # Note: We need to pass rng_key to sample() but sample() is method of actor.
            # Handling randomness in Functional API inside loss:
            # We should pass the key or make it explicit.
            # Actor.sample takes rng_key.
            
            new_actions, log_probs = actor.sample(states, actor_key)
            
            # Query Critic (frozen)
            # We use self.critic (which was just updated)
            q1, q2 = self.critic(states, new_actions)
            min_q = jnp.minimum(q1, q2)
            
            # Maximize: min_Q - alpha * log_prob
            # Minimize: alpha * log_prob - min_Q
            return jnp.mean(current_alpha * log_probs - min_q)
            
        actor_grads = nnx.grad(actor_loss_fn)(self.actor)
        actor_updates, new_actor_opt_state = self.actor_optimizer.update(
            actor_grads,
            opt_states['actor_opt'],
            nnx.state(self.actor, nnx.Param)
        )
        nnx.update(self.actor, optax.apply_updates(nnx.state(self.actor, nnx.Param), actor_updates))
        
        # --- 3. Alpha Update ---
        # Minimize: -alpha * (log_prob + target_entropy).detach()
        # But wait, we need log_prob from the CURRENT policy (after update? or before? usually before/same batch)
        # Using the log_probs computed during actor loss is efficient and standard.
        
        log_probs_detached = jax.lax.stop_gradient(next_log_probs) # Using 'next' or 'current'? 
        # Actually usually re-computed or reused from Actor step.
        # Let's reuse 'next_log_probs' from the Target Q step for simplicity (approx) 
        # OR re-run. Let's re-run to be safe (Actor changed? yes slightly).
        # Actually, standard impl uses log_prob from actor update.
        # But we need it outside the actor_loss_fn.
        
        # We can re-call actor.sample specific for alpha loss to get gradients for Alpha?
        # Alpha loss depends on alpha parameter.
        
        # Re-calc log_prob with *current* actor (updated)
        _, current_log_probs = self.actor.sample(states, actor_key)
        current_log_probs = jax.lax.stop_gradient(current_log_probs)
        
        def alpha_loss_fn(log_alpha_module):
            alpha = jnp.exp(log_alpha_module())
            return -jnp.mean(alpha * (current_log_probs + self.target_entropy))
            
        alpha_grads = nnx.grad(alpha_loss_fn)(self.log_alpha)
        alpha_updates, new_alpha_opt_state = self.alpha_optimizer.update(
            alpha_grads,
            opt_states['alpha_opt'],
            nnx.state(self.log_alpha, nnx.Param)
        )
        nnx.update(self.log_alpha, optax.apply_updates(nnx.state(self.log_alpha, nnx.Param), alpha_updates))

        # Update Target Critic
        self.update_target()
        
        new_opt_states = {
            'actor_opt': new_actor_opt_state,
            'critic_opt': new_critic_opt_state,
            'alpha_opt': new_alpha_opt_state
        }
        
        return {
            'critic_loss': 0.0, # Placeholder
            'actor_loss': 0.0,
            'alpha': current_alpha
        }, new_opt_states

