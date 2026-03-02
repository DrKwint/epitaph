
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

from epitaph.sac import SACAgent

def test_sac():
    print("Initializing SAC Agent...")
    rng = nnx.Rngs(0)
    state_dim = 17
    action_dim = 6
    agent = SACAgent(state_dim, action_dim, rngs=rng)
    opt_states = agent.init_state()
    
    print("Creating Dummy Batch...")
    batch_size = 32
    key = jax.random.PRNGKey(0)
    
    s = jax.random.normal(key, (batch_size, state_dim))
    a = jax.random.normal(key, (batch_size, action_dim))
    r = jax.random.normal(key, (batch_size,))
    sn = jax.random.normal(key, (batch_size, state_dim))
    d = jnp.zeros((batch_size,))
    
    batch = (s, a, r, sn, d)
    
    print("Running Train Step...")
    try:
        info, new_opt_states = agent.train_step(batch, key, opt_states)
        print("Success!")
        print(info)
    except Exception:
        import traceback
        with open('traceback.txt', 'w') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_sac()
    except Exception as e:
        import traceback
        traceback.print_exc()
