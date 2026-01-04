import jax
import jax.numpy as jnp
from epitaph.models import Epinet
from epitaph.control import get_safe_z_constraints
from flax import nnx
import sys

def test_get_safe_z_constraints_shapes():
    print("Testing get_safe_z_constraints shapes...")
    z_dim = 2
    in_features = 2 # [s, a]
    out_features = 1 # [s']
    
    model = Epinet(in_features, out_features, z_dim, rngs=nnx.Rngs(0))
    
    s0 = jnp.zeros((1,))
    action_seq = jnp.zeros((5, 1)) # (Horizon, Action_Dim)
    
    # Safe region: s <= 0.5
    unsafe_A = jnp.array([[1.0]])
    unsafe_b = jnp.array([0.5])
    
    # This should fail with several errors
    C, d = get_safe_z_constraints(model, s0, action_seq, unsafe_A, unsafe_b)
    print(f"Success! C shape: {C.shape}, d shape: {d.shape}")

if __name__ == "__main__":
    try:
        test_get_safe_z_constraints_shapes()
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
