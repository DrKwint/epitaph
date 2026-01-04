"""
Unit and property-based tests for ReplayBuffer.
Tests storage, sampling, and splitting functionality for (s, a, r, s', done) transitions.
"""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st, settings

from epitaph.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    
    def test_basic_add_and_sample(self):
        """Test basic add and sample operations."""
        buffer = ReplayBuffer(capacity=10, state_dim=2, action_dim=1)
        
        # Add a single transition
        state = np.array([1.0, 2.0])
        action = np.array([0.5])
        reward = 1.5
        next_state = np.array([1.1, 2.1])
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 1)
        self.assertFalse(buffer.is_full())
        
        # Sample it back
        key = jax.random.PRNGKey(0)
        s, a, r, sn, d = buffer.sample(1, key=key)
        
        self.assertEqual(s.shape, (1, 2))
        self.assertEqual(a.shape, (1, 1))
        self.assertEqual(r.shape, (1,))
        self.assertEqual(sn.shape, (1, 2))
        self.assertEqual(d.shape, (1,))
        
        # Check values match
        self.assertTrue(jnp.allclose(s[0], state))
        self.assertTrue(jnp.allclose(a[0], action))
        self.assertAlmostEqual(float(r[0]), reward)
        self.assertTrue(jnp.allclose(sn[0], next_state))
        self.assertEqual(bool(d[0]), done)
    
    def test_batch_add(self):
        """Test adding multiple transitions at once."""
        buffer = ReplayBuffer(capacity=100, state_dim=3, action_dim=2)
        
        n_samples = 50
        states = np.random.randn(n_samples, 3).astype(np.float32)
        actions = np.random.randn(n_samples, 2).astype(np.float32)
        rewards = np.random.randn(n_samples).astype(np.float32)
        next_states = np.random.randn(n_samples, 3).astype(np.float32)
        dones = np.random.rand(n_samples) > 0.9
        
        buffer.add_batch(states, actions, rewards, next_states, dones)
        
        self.assertEqual(len(buffer), n_samples)
        self.assertFalse(buffer.is_full())
        
        # Sample and verify shapes
        key = jax.random.PRNGKey(42)
        s, a, r, sn, d = buffer.sample(10, key=key)
        
        self.assertEqual(s.shape, (10, 3))
        self.assertEqual(a.shape, (10, 2))
        self.assertEqual(r.shape, (10,))
        self.assertEqual(sn.shape, (10, 3))
        self.assertEqual(d.shape, (10,))
    
    def test_circular_buffer_overflow(self):
        """Test that buffer correctly wraps around when capacity is exceeded."""
        buffer = ReplayBuffer(capacity=5, state_dim=1, action_dim=1)
        
        # Add 10 transitions (twice the capacity)
        for i in range(10):
            buffer.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([float(i+1)]),
                False
            )
        
        # Buffer should be full with size 5
        self.assertEqual(len(buffer), 5)
        self.assertTrue(buffer.is_full())
        
        # Should contain last 5 transitions (5-9)
        s, a, r, sn, d = buffer.get_all()
        rewards = np.array(r)
        
        # Rewards should be from the last 5 added (5, 6, 7, 8, 9)
        self.assertTrue(np.all(rewards >= 5))
        self.assertTrue(np.all(rewards <= 9))
    
    def test_split_shuffled(self):
        """Test splitting buffer with shuffling."""
        buffer = ReplayBuffer(capacity=100, state_dim=1, action_dim=1)
        
        # Add 100 transitions with sequential rewards for tracking
        for i in range(100):
            buffer.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([float(i+1)]),
                (i % 10 == 0)
            )
        
        # Split 80/20 with shuffle
        train_buf, val_buf = buffer.split(split_ratio=0.8, shuffle=True, seed=42)
        
        self.assertEqual(len(train_buf), 80)
        self.assertEqual(len(val_buf), 20)
        
        # Get all data
        train_s, train_a, train_r, train_sn, train_d = train_buf.get_all()
        val_s, val_a, val_r, val_sn, val_d = val_buf.get_all()
        
        # Verify we have all original data (union of train and val)
        all_rewards = np.concatenate([np.array(train_r), np.array(val_r)])
        self.assertEqual(len(np.unique(all_rewards)), 100)
        
        # Train rewards should NOT be perfectly sequential (shuffled)
        train_rewards_arr = np.array(train_r)
        is_sequential = np.allclose(np.sort(train_rewards_arr), np.arange(80))
        self.assertFalse(is_sequential)
    
    def test_split_sequential(self):
        """Test splitting buffer without shuffling."""
        buffer = ReplayBuffer(capacity=20, state_dim=1, action_dim=1)
        
        for i in range(20):
            buffer.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([float(i+1)]),
                False
            )
        
        # Split 75/25 without shuffle
        train_buf, val_buf = buffer.split(split_ratio=0.75, shuffle=False)
        
        self.assertEqual(len(train_buf), 15)
        self.assertEqual(len(val_buf), 5)
        
        train_r = np.array(train_buf.get_all()[2])
        val_r = np.array(val_buf.get_all()[2])
        
        # Should be sequential split
        self.assertTrue(np.allclose(train_r, np.arange(15)))
        self.assertTrue(np.allclose(val_r, np.arange(15, 20)))
    
    def test_get_all(self):
        """Test get_all returns all stored transitions."""
        buffer = ReplayBuffer(capacity=50, state_dim=4, action_dim=2)
        
        n = 30
        for i in range(n):
            buffer.add(
                np.random.randn(4).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                np.random.randn(),
                np.random.randn(4).astype(np.float32),
                bool(np.random.rand() > 0.9)
            )
        
        s, a, r, sn, d = buffer.get_all()
        
        self.assertEqual(s.shape, (30, 4))
        self.assertEqual(a.shape, (30, 2))
        self.assertEqual(r.shape, (30,))
        self.assertEqual(sn.shape, (30, 4))
        self.assertEqual(d.shape, (30,))
    
    @settings(deadline=None, max_examples=10)
    @given(
        capacity=st.integers(min_value=10, max_value=100),
        n_samples=st.integers(min_value=5, max_value=50),
        state_dim=st.integers(min_value=1, max_value=5),
        action_dim=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_sample_property(self, capacity, n_samples, state_dim, action_dim, seed):
        """
        Property: Sampling from buffer always returns valid transitions with correct shapes.
        """
        # Ensure we can sample
        n_add = min(n_samples * 2, capacity)
        batch_size = min(n_samples, n_add)
        
        buffer = ReplayBuffer(capacity=capacity, state_dim=state_dim, action_dim=action_dim)
        
        # Add random data
        rng = np.random.RandomState(seed)
        for _ in range(n_add):
            buffer.add(
                rng.randn(state_dim).astype(np.float32),
                rng.randn(action_dim).astype(np.float32),
                rng.randn(),
                rng.randn(state_dim).astype(np.float32),
                bool(rng.rand() > 0.8)
            )
        
        # Sample
        key = jax.random.PRNGKey(seed)
        s, a, r, sn, d = buffer.sample(batch_size, key=key)
        
        # Verify shapes
        self.assertEqual(s.shape, (batch_size, state_dim))
        self.assertEqual(a.shape, (batch_size, action_dim))
        self.assertEqual(r.shape, (batch_size,))
        self.assertEqual(sn.shape, (batch_size, state_dim))
        self.assertEqual(d.shape, (batch_size,))
        
        # Verify all are valid JAX arrays
        self.assertTrue(isinstance(s, jax.Array))
        self.assertTrue(isinstance(a, jax.Array))
        self.assertTrue(isinstance(r, jax.Array))
        self.assertTrue(isinstance(sn, jax.Array))
        self.assertTrue(isinstance(d, jax.Array))
    
    @settings(deadline=None, max_examples=10)
    @given(
        total_size=st.integers(min_value=10, max_value=100),
        split_ratio=st.floats(min_value=0.1, max_value=0.9),
        state_dim=st.integers(min_value=1, max_value=4),
        action_dim=st.integers(min_value=1, max_value=2),
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_split_property(self, total_size, split_ratio, state_dim, action_dim, seed):
        """
        Property: Split always preserves all data and creates buffers of correct sizes.
        """
        buffer = ReplayBuffer(capacity=total_size, state_dim=state_dim, action_dim=action_dim)
        
        # Add data
        rng = np.random.RandomState(seed)
        for i in range(total_size):
            buffer.add(
                rng.randn(state_dim).astype(np.float32),
                rng.randn(action_dim).astype(np.float32),
                float(i),  # Use index as reward for tracking
                rng.randn(state_dim).astype(np.float32),
                False
            )
        
        # Split
        buf1, buf2 = buffer.split(split_ratio=split_ratio, shuffle=True, seed=seed)
        
        # Check sizes
        expected_size1 = int(total_size * split_ratio)
        expected_size2 = total_size - expected_size1
        
        self.assertEqual(len(buf1), expected_size1)
        self.assertEqual(len(buf2), expected_size2)
        
        # Check that we preserved all data (no duplicates, no missing)
        r1 = np.array(buf1.get_all()[2])
        r2 = np.array(buf2.get_all()[2])
        all_rewards = np.concatenate([r1, r2])
        
        # All original rewards should be present exactly once
        self.assertEqual(len(np.unique(all_rewards)), total_size)
        self.assertTrue(np.all(np.sort(all_rewards) == np.arange(total_size)))
    
    def test_iterate_batches_coverage(self):
        """Test that iterate_batches covers all data exactly once."""
        buffer = ReplayBuffer(capacity=50, state_dim=1, action_dim=1)
        
        # Add 50 transitions with unique rewards for tracking
        for i in range(50):
            buffer.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([float(i+1)]),
                False
            )
        
        # Iterate with batch size 7 (not a divisor of 50)
        collected_rewards = []
        for s, a, r, sn, d in buffer.iterate_batches(batch_size=7, shuffle=False):
            collected_rewards.extend(np.array(r).tolist())
        
        # Should have all 50 rewards
        self.assertEqual(len(collected_rewards), 50)
        self.assertTrue(np.allclose(np.sort(collected_rewards), np.arange(50)))
    
    def test_iterate_batches_shapes(self):
        """Test that iterate_batches yields correct shapes."""
        buffer = ReplayBuffer(capacity=100, state_dim=3, action_dim=2)
        
        # Add 37 transitions (irregular number)
        for i in range(37):
            buffer.add(
                np.random.randn(3).astype(np.float32),
                np.random.randn(2).astype(np.float32),
                float(i),
                np.random.randn(3).astype(np.float32),
                False
            )
        
        batch_size = 10
        batch_count = 0
        total_samples = 0
        
        for s, a, r, sn, d in buffer.iterate_batches(batch_size=batch_size, shuffle=True, seed=42):
            batch_count += 1
            batch_len = s.shape[0]
            total_samples += batch_len
            
            # Check shapes
            self.assertEqual(s.shape, (batch_len, 3))
            self.assertEqual(a.shape, (batch_len, 2))
            self.assertEqual(r.shape, (batch_len,))
            self.assertEqual(sn.shape, (batch_len, 3))
            self.assertEqual(d.shape, (batch_len,))
            
            # All batches except possibly last should be full
            if batch_count < 4:  # 37 / 10 = 3.7, so first 3 batches full
                self.assertEqual(batch_len, batch_size)
        
        # Should have iterated over all samples
        self.assertEqual(total_samples, 37)
        self.assertEqual(batch_count, 4)  # ceil(37/10)
    
    def test_iterate_batches_shuffle(self):
        """Test that shuffle in iterate_batches actually shuffles."""
        buffer = ReplayBuffer(capacity=30, state_dim=1, action_dim=1)
        
        for i in range(30):
            buffer.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([0.0]),
                False
            )
        
        # Iterate without shuffle
        rewards_no_shuffle = []
        for s, a, r, sn, d in buffer.iterate_batches(batch_size=30, shuffle=False):
            rewards_no_shuffle.extend(np.array(r).tolist())
        
        # Should be sequential
        self.assertTrue(np.allclose(rewards_no_shuffle, np.arange(30)))
        
        # Iterate with shuffle
        rewards_shuffle = []
        for s, a, r, sn, d in buffer.iterate_batches(batch_size=30, shuffle=True, seed=42):
            rewards_shuffle.extend(np.array(r).tolist())
        
        # Should NOT be sequential (with high probability)
        is_sequential = np.allclose(rewards_shuffle, np.arange(30))
        self.assertFalse(is_sequential)
        
        # But should still have all values
        self.assertTrue(np.allclose(np.sort(rewards_shuffle), np.arange(30)))


if __name__ == "__main__":
    unittest.main()
