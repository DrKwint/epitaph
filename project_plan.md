# Project Plan: Safe Model-Based Control via Epistemic Set Propagation

Based on `concept.md` and `formulation.md`, this plan outlines the implementation roadmap and experimentation strategy for the Hybrid Epinet-Zonotope Controller.

## 1. Implementation Plan

The system will be built using **JAX** and **Flax NNX**, prioritizing JIT-compilation for the core verification kernel.

### Phase 1: Core Primitives (The "Math Library")
**Goal**: Build the differentiable set-propagation logic.
-   **`ConstrainedZonotope` Dataclass**: Define the JAX pytree structure `(center, generators, A, b)`.
-   **Propagate Linear**: Implement $Z' = WZ + b$ logic for linear layers.
-   **Propagate ReLU**: Implement over-approximations for non-linearities (e.g., DeepPoly or Lambda-Zonotopes).
-   **Constraint Slicing**: Implement the logic to append safety constraints $Hs \le d$ to the zonotope's $A, b$ matrices.

### Phase 2: Epistemic Dynamics (The "Model")
**Goal**: Train a model that separates physics form uncertainty.
-   **Architecture**: Implement a Piecewise Linear (PWL) Epinet in Flax.
    -   Base network for mean prediction.
    -   Epistemic head taking parameter $z$.
-   **Training Loop**: Standard MLE or variants suited for Epinets to ensure $z$ captures epistemic conceptual uncertainty.

### Phase 3: The Safety Verify (The "Signal")
**Goal**: Convert set geometry into a scalar safety probability.
-   **Volume Computation**: Implement `compute_z_volume(cz)`.
    -   *Strategy*: Since exact polytope volume is #P-hard helping, start with a differentiable approximation (e.g., bounding box volume of the valid $\xi_z$ space) or Monte-Carlo estimation within the kernel if feasible.
-   **Differentiability Check**: Ensure gradients flow from the volume output back to the action inputs.

### Phase 4: Controller Integration (The "Agent")
**Goal**: Close the loop with MPPI.
-   **`rollout_kernel`**: JIT-compiled function combining Model + Primitives + Safety Verifier.
-   **MPPI Loop**:
    -   Sampling $N$ trajectories.
    -   `vmap` rollout.
    -   Safety-weighted aggregation: $w \propto P_{safe} \cdot e^{\gamma R}$.

## 2. Experimentation Plan

### Environments
1.  **Inverted Pendulum (Toy)**:
    -   *Constraint*: max angular velocity or angle limits.
    -   *Purpose*: Sanity check for set propagation and gradient flow.
2.  **Cartpole with Walls (Classic)**:
    -   *Constraint*: $x_{min} \le x \le x_{max}$.
    -   *Purpose*: Test "Cliff" avoidance. Hitting the wall is terminal/catastrophic.
3.  **2D Quadrotor / Point Mass Navigation**:
    -   *Constraint*: Static obstacles.
    -   *Purpose*: Multi-dimensional state constraints.

### Baselines
1.  **PETS (Probabilistic Ensembles with Trajectory Sampling)**:
    -   The primary rival. Uses particle ensembles for risk.
    -   *Expectation*: PETS will fail to distinguish low-margin safety from high-margin safety.
2.  **Standard MPPI**:
    -   Uses soft penalty for constraint violations observed in particles.
3.  **Oracle MPPI (Upper Bound)**:
    -   MPPI with access to the true dynamics and perfect safety checks.

### Metrics & Hypotheses to Validate
1.  **Safety Violation Rate**:
    -   *Hypothesis*: Our method approaches 0 violations faster than PETS in low-data regimes.
2.  **Sample Efficiency**:
    -   *Hypothesis*: We need fewer interactions to learn a "safe enough" policy because the consistent epistemic signal guides exploration safely.
3.  **"Cliff" Detection**:
    -   *Specific Test*: Initialize agent near a boundary. Assert that our controller assigns $P_{safe} < 1.0$ even if all sampled mean-trajectories are safe, whereas PETS might assign Risk=0 if no particle hits the wall.
