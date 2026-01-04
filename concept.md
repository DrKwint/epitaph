# Project Title: Safe Model-Based Control via Epistemic Set Propagation

**Objective**: Demonstrate that hybridizing particle-based sampling (PETS) with set-based verification (Zonotopes) using Epistemic Neural Networks (Epinets) provides superior safety guarantees and sample efficiency compared to pure sampling baselines.

## 1. The Core Problem

Standard Model-Based RL methods (like PETS) rely on particle sampling to estimate risk. This has two critical flaws in safety-critical settings:

*   **The Zero-Shot Blindness**: A trajectory that misses a cliff edge by 1mm and one that misses by 1km both have a 100% survival rate in a finite particle batch. The optimizer cannot distinguish between "barely safe" and "robustly safe."
*   **Inconsistent Physics**: Resampling model ensembles at every step implies the laws of physics change randomly over time. This prevents the agent from committing to safe corridors that rely on consistent (though unknown) dynamics.

## 2. The Solution: Epistemic Set Propagation

We propose a Hybrid Epinet-Zonotope Controller.

*   **Epistemic Consistency**: We use an Epinet to separate aleatoric noise from epistemic uncertainty. We sample a "physics hypothesis" (epistemic index $z$) and hold it constant for the trajectory horizon.
*   **Set-Based Safety**: Instead of propagating points, we propagate Constrained Zonotopes of the joint state-parameter space $(s, z)$.
*   **Continuous Safety Signal**: When a trajectory approaches a constraint, we do not discard it. We calculate the volume of the remaining valid hypothesis space for $z$. This provides a differentiable lower bound on the Probability of Safety, allowing the optimizer to maximize safety margins even when no collision occurs.

## 3. The "Pitch" vs. Baselines

*   **Baseline (PETS)**: Optimizes for Survival Rate (discrete, high variance).
*   **Ours**: Optimizes for Confidence of Safety (continuous, differentiable).
*   **Hypothesis**: Our method will demonstrate fewer constraint violations during early training (low data regimes) because it detects and avoids "cliffs" in the epistemic landscape that particle sampling misses.