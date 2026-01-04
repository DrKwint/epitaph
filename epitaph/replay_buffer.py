"""
Replay buffer for storing and sampling transitions.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """Replay buffer for storing (s, a, r, s', done) transitions."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        
        self.idx = 0
        self.size = 0
    
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a single transition to the buffer."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self, 
        states: np.ndarray, 
        actions: np.ndarray,
        rewards: np.ndarray, 
        next_states: np.ndarray,
        dones: np.ndarray
    ):
        """Add a batch of transitions to the buffer."""
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(
        self, 
        batch_size: int, 
        key: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            key: JAX random key (if None, uses numpy random)
            
        Returns:
            (states, actions, rewards, next_states, dones) as JAX arrays
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")
        
        if key is not None:
            # Use JAX random
            idx = jax.random.choice(key, self.size, shape=(batch_size,), replace=False)
            idx = np.array(idx)  # Convert to numpy for indexing
        else:
            # Use numpy random
            idx = np.random.choice(self.size, size=batch_size, replace=False)
        
        return (
            jnp.array(self.states[idx]),
            jnp.array(self.actions[idx]),
            jnp.array(self.rewards[idx]),
            jnp.array(self.next_states[idx]),
            jnp.array(self.dones[idx])
        )
    
    def get_all(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Get all stored transitions as JAX arrays."""
        return (
            jnp.array(self.states[:self.size]),
            jnp.array(self.actions[:self.size]),
            jnp.array(self.rewards[:self.size]),
            jnp.array(self.next_states[:self.size]),
            jnp.array(self.dones[:self.size])
        )
    
    def iterate_batches(
        self, 
        batch_size: int, 
        shuffle: bool = True, 
        seed: Optional[int] = None
    ):
        """
        Iterate over the buffer one batch at a time without replacement.
        Useful for epoch-based training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before iterating (default: True)
            seed: Random seed for shuffling (if None, uses current numpy random state)
            
        Yields:
            (states, actions, rewards, next_states, dones) tuples for each batch
            
        Example:
            >>> for s, a, r, sn, d in buffer.iterate_batches(batch_size=32):
            >>>     # Train on batch
            >>>     loss = train_step(s, a, r, sn, d)
        """
        if self.size == 0:
            return
        
        # Get indices
        indices = np.arange(self.size)
        
        # Shuffle if requested
        if shuffle:
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random
            rng.shuffle(indices)
        
        # Iterate in batches
        num_batches = (self.size + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                jnp.array(self.states[batch_indices]),
                jnp.array(self.actions[batch_indices]),
                jnp.array(self.rewards[batch_indices]),
                jnp.array(self.next_states[batch_indices]),
                jnp.array(self.dones[batch_indices])
            )
    
    def split(self, split_ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None) -> Tuple['ReplayBuffer', 'ReplayBuffer']:
        """
        Split the buffer into two new buffers (e.g., for train/val).
        
        Args:
            split_ratio: Fraction of data to put in first buffer (default: 0.8)
            shuffle: Whether to shuffle data before splitting (default: True)
            seed: Random seed for shuffling (if None, uses current numpy random state)
            
        Returns:
            (buffer1, buffer2): Two new ReplayBuffer instances
        """
        if self.size == 0:
            raise ValueError("Cannot split empty buffer")
        
        # Get all data
        states, actions, rewards, next_states, dones = self.get_all()
        
        # Convert to numpy for manipulation
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Shuffle if requested
        if shuffle:
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random
            
            indices = rng.permutation(self.size)
            states = states[indices]
            actions = actions[indices]
            rewards = rewards[indices]
            next_states = next_states[indices]
            dones = dones[indices]
        
        # Split
        split_idx = int(self.size * split_ratio)
        
        # Create first buffer
        buffer1 = ReplayBuffer(split_idx, self.state_dim, self.action_dim)
        buffer1.add_batch(
            states[:split_idx],
            actions[:split_idx],
            rewards[:split_idx],
            next_states[:split_idx],
            dones[:split_idx]
        )
        
        # Create second buffer
        buffer2_size = self.size - split_idx
        buffer2 = ReplayBuffer(buffer2_size, self.state_dim, self.action_dim)
        buffer2.add_batch(
            states[split_idx:],
            actions[split_idx:],
            rewards[split_idx:],
            next_states[split_idx:],
            dones[split_idx:]
        )
        
        return buffer1, buffer2
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.size == self.capacity
    
    def clear(self):
        """Clear the buffer."""
        self.idx = 0
        self.size = 0
