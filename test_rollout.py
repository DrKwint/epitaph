
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from epitaph.sac import SACAgent
from epitaph.infoprop import infoprop_step
from train_mbrl_infoprop import rollout_model
from epitaph.replay_buffer import ReplayBuffer

class DummyModel:
    def predict(self, x):
        # Return dummy means (E, B, D)
        # x is (B, D_in)
        batch = x.shape[0]
        dim = 17 # HalfCheetah state dim
        E = 5
        return jnp.ones((E, batch, dim))

def test_rollout():
    print("Initializing components...")
    rng = nnx.Rngs(0)
    state_dim = 17
    action_dim = 6
    agent = SACAgent(state_dim, action_dim, rngs=rng)
    # Init state (not needed for select_action usually but let's be safe)
    agent.init_state()
    
    ensemble = DummyModel()
    
    buffer = ReplayBuffer(1000, state_dim, action_dim)
    # Add one dummy transition
    buffer.add(np.zeros(state_dim), np.zeros(action_dim), 0.0, np.zeros(state_dim), False)
    
    print("Running Rollout...")
    key = jax.random.PRNGKey(0)
    rollout_data = rollout_model(
        ensemble, buffer, agent, 
        rollout_length=5, 
        batch_size=10, 
        rng_key=key
    )
    
    print("Rollout Success!")
    print("Keys:", rollout_data.keys())
    print("Shapes:", {k: v.shape for k, v in rollout_data.items()})

if __name__ == "__main__":
    try:
        test_rollout()
    except Exception:
        import traceback
        with open('traceback.txt', 'w') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
