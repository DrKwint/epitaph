# Topic: Probabilistic Safety via Parameter Space Volume

**Context**: Deriving the "Confidence of Safety" metric for MPPI.

## 1. Augmented Dynamics

We define the system state as an augmented vector containing the physical state $s \in \mathbb{R}^n$ and the epistemic parameter $z \in \mathbb{R}^k$. The epistemic parameter represents the specific realization of the learned dynamics model and is invariant over the planning horizon.

$$
\begin{bmatrix} s_{t+1} \\ z_{t+1} \end{bmatrix} = \begin{bmatrix} f_\theta(s_t, a_t, z_t) \\ I(z_t) \end{bmatrix}
$$

*   $f_\theta$: A Piecewise Linear (PWL) neural network (Epinet).
*   $z_0$: Initialized as a set (e.g., a hypercube) representing the support of the epistemic prior distribution $P(z)$.

## 2. Constrained Zonotope Propagation

We propagate the joint set $\mathcal{Z}_t \subset \mathbb{R}^{n+k}$ using Constrained Zonotopes.

A Constrained Zonotope is defined as:

$$
\mathcal{Z} = \{ c + G\xi \mid ||\xi||_\infty \leq 1, A\xi = b \}
$$

where $\xi$ contains generators for both state uncertainty and the epistemic parameters $z$.

When the set encounters a linear safety constraint $H s_t \leq d$:

1.  Since $f_\theta$ is PWL, the reachable set is (approximated as) a Constrained Zonotope.
2.  The safety constraint imposes a linear cut on the generator space $\xi$.
3.  We append this cut to the constraint matrix $A$, effectively slicing the valid domain of $z$.

## 3. The Safety Lower Bound

Let $\Omega_{valid} \subseteq \mathbb{R}^k$ be the subset of initial epistemic parameters $z$ that result in a safe trajectory $a_{0:H}$.

Since our Zonotope propagation is an over-approximation of the reachable set:

$$
\text{Intersection}(\mathcal{R}_{approx}, \mathcal{X}_{unsafe}) = \emptyset \implies \text{Intersection}(\mathcal{R}_{true}, \mathcal{X}_{unsafe}) = \emptyset
$$

Therefore, the volume of the surviving parameter set is a rigorous lower bound on the probability of safety:

$$
P(\text{Safe} \mid a_{0:H}) \geq \frac{\text{Vol}(\text{Proj}_z(\mathcal{Z}_H))}{\text{Vol}(Z_{prior})}
$$

## 4. MPPI Objective

We replace the standard reward-weighting in MPPI with a safety-weighted objective.

For trajectory $i$, the weight $w_i$ is:

$$
w_i \propto \underbrace{P_{LB}(\text{Safe} \mid \tau_i)}_{\text{Epistemic Confidence}} \cdot \exp\left( \frac{1}{\lambda} \sum_{t=0}^H R(s_t, a_t) \right)
$$

This ensures that high-reward trajectories are penalized proportionally to the volume of hypotheses under which they fail.

## 5. System Architecture & Implementation

**Stack**: JAX, Flax NNX
**Key Libraries**: jax.vmap, jax.lax.scan

### The Verification Kernel (JAX)

This is the core compute unit. It must be JIT-compilable.

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Pseudo-definition for Constrained Zonotope
# CZ = (center, generators, A_constraints, b_constraints)

@jax.jit
def rollout_kernel(model, initial_cz, action_sequence):
    """
    Args:
        model: Flax NNX Epinet (PWL)
        initial_cz: Augmented Constrained Zonotope (s, z)
        action_sequence: Shape [H, Action_Dim]
    Returns:
        accumulated_reward: Scalar
        safety_prob_bound: Scalar (0.0 to 1.0)
    """

    def step_fn(carry_cz, action):
        # 1. Propagate CZ through PWL network
        #    - Linear layers: Matmul on center/generators
        #    - ReLU layers: Analytic approximation or choice of slope
        next_cz = model.propagate(carry_cz, action)
        
        # 2. Apply Safety Constraints
        #    - Check intersection with Unsafe Set (Linear constraints)
        #    - Update CZ.A and CZ.b to slice invalid 'z' regions
        constrained_cz = apply_constraints(next_cz, unsafe_set)
        
        # 3. Compute Step Reward (using CZ center)
        reward = compute_reward(constrained_cz.center, action)
        
        return constrained_cz, reward

    # Scan over horizon
    final_cz, rewards = jax.lax.scan(step_fn, initial_cz, action_sequence)
    
    # Calculate Lower Bound Probability
    # Project final_cz onto 'z' dimensions and compute polytope volume
    vol = compute_z_volume(final_cz)
    prob_bound = vol / initial_z_volume
    
    return jnp.sum(rewards), prob_bound
```

### The MPPI Controller Loop

This replaces the standard Cross-Entropy Method.

1.  **Sample**: Generate $N=1000$ action sequences from Gaussian $\mathcal{N}(\mu, \Sigma)$.
2.  **Vectorize**: Use `jax.vmap(rollout_kernel, in_axes=(None, None, 0))` to process all sequences in parallel.
3.  **Weight**: Compute weights $w_i = \text{prob\_bound}_i \cdot \exp(\gamma \cdot \text{total\_reward}_i)$.
4.  **Aggregate**: Compute the weighted average of the action sequences to update $\mu$.
    $$
    \mu_{new} = \frac{\sum w_i \cdot a_i}{\sum w_i}
    $$
5.  **Act**: Execute the first action of $\mu_{new}$.

## 6. Development Roadmap

*   [ ] **Phase 1 (Primitives)**: Implement `ConstrainedZonotope` class in JAX. Implement `propagate_linear` and `propagate_relu` (using Lambda-over-approximation or similar).
*   [ ] **Phase 2 (Epinet)**: Train a simple Epinet on a toy environment (e.g., Pendulum/Cartpole).
*   [ ] **Phase 3 (Volume)**: Implement the `compute_z_volume` function. (Note: Exact volume is hard; approximate using bounding box of the $\xi$ constraints or specialized polytope volume libraries wrapped in JAX).
*   [ ] **Phase 4 (Integration)**: Build the MPPI loop and benchmark against standard PETS.