Implementing an Osband et al. Epistemic Neural Network (Epinet)Reference Paper: Epistemic Neural Networks (Osband et al., 2021) & Randomized Prior Functions for Deep Reinforcement Learning (Osband et al., 2018).Purpose: Defines the architecture for the transition dynamics model $f(s, a, z) \rightarrow s'$. This network separates the "mean" prediction from the "epistemic" variation, allowing us to sample consistent physical hypotheses by fixing the epistemic index $z$.1. Conceptual ArchitectureThe Epinet is not a simple ensemble. It is a single architecture that takes an epistemic index $z$ as an auxiliary input to output a specific "hypothesis" of the dynamics.We follow the "Matched Prior" design pattern, which is crucial for good uncertainty estimates in sparse data regimes.$$f_{\text{total}}(x, z) = \underbrace{\mu_\theta(x)}_{\text{Base Network}} + \underbrace{\lambda \cdot \big( \eta_\phi(x, z) + \pi_{\text{fixed}}(x, z) \big)}_{\text{Epistemic Head}}$$$\mu_\theta(x)$ (Base Mean): A standard MLP that learns the average data trend. It does not see $z$.$\eta_\phi(x, z)$ (Learnable Epistemic): A trainable network that takes $x$ and $z$. It learns to shape the uncertainty based on data.$\pi_{\text{fixed}}(x, z)$ (Prior Network): A frozen, randomly initialized network. It never updates. Its job is to provide "noise" in regions where the agent has no data. As data comes in, $\eta_\phi$ learns to cancel out $\pi_{\text{fixed}}$ to reduce uncertainty, or amplify it where needed.$\lambda$ (Scaling): A hyperparameter controlling the magnitude of the prior/uncertainty.2. Architecture Specification (JAX/Flax NNX)Input Dimensions:$x$: Concatenation of $[s_t, a_t]$.$z$: Epistemic Index.1 Usually a vector drawn from $\mathcal{N}(0, I)$ of dimension $K$ (e.g., $K=10$ or $K=20$).Component Breakdown:Base Network (base_net):Input: $x$Layers: Linear -> ReLU -> Linear -> ... -> Linear (Output Dim = State Dim)Note: This should be Piecewise Linear (ReLU) to support Zonotope verification.Epistemic Network (enn_net):Input: Concatenation $[x, z]$ (or inject $z$ into hidden layers).Recommendation: A "rank-1" modulation is often used. The last layer outputs a weight matrix and bias offset predicted from $z$, or simply acts as an additive offset to the Base output.Simple Implementation: MLP taking $[x, z]$ -> Output Dim = State Dim.Prior Network (prior_net):Identical architecture to the Epistemic Network.Crucial: Its parameters are initialized once and never updated (stop_gradient).3. Implementation Details for Flax NNXIn Flax NNX, we handle the stateful separation of "Trainable" vs. "Frozen" parameters explicitly.Pythonimport jax
import jax.numpy as jnp
from flax import nnx

class Epinet(nnx.Module):
    def __init__(self, in_features, out_features, z_dim, rngs):
        self.z_dim = z_dim
        
        # 1. Base Network (The Mean)
        # Standard MLP: s,a -> s'
        self.base_net = nnx.Sequential(
            nnx.Linear(in_features, 64, rngs=rngs),
            nnx.relu,
            nnx.Linear(64, 64, rngs=rngs),
            nnx.relu,
            nnx.Linear(64, out_features, rngs=rngs)
        )

        # 2. Learnable Epistemic Network (The Correction)
        # Takes [s, a, z] -> s' offset
        self.learnable_enn = nnx.Sequential(
            nnx.Linear(in_features + z_dim, 64, rngs=rngs),
            nnx.relu,
            nnx.Linear(64, out_features, rngs=rngs)
        )

        # 3. Prior Network (The Anchor)
        # Same architecture as Learnable, but we will treating it as static
        # Note: We create a separate RNG stream to ensure it differs from learnable_enn
        prior_rng = nnx.RngStream(rngs.params.key + 1) # simple offset for demo
        self.prior_enn = nnx.Sequential(
            nnx.Linear(in_features + z_dim, 64, rngs=prior_rng),
            nnx.relu,
            nnx.Linear(64, out_features, rngs=prior_rng)
        )
        
        # Prior Scaling Factor (Lambda)
        self.prior_scale = 5.0 

    def __call__(self, x, z):
        """
        Forward pass for a specific epistemic index z.
        """
        # 1. Base prediction
        mu = self.base_net(x)
        
        # 2. Epistemic Inputs
        # Concatenate state/action with the epistemic index
        enn_input = jnp.concatenate([x, z], axis=-1)
        
        # 3. Calculate Epistemic Offset
        # We stop_gradient on the prior because it is a fixed anchor
        prior_out = jax.lax.stop_gradient(self.prior_enn(enn_input))
        learnable_out = self.learnable_enn(enn_input)
        
        # 4. Combine
        # The 'net' uncertainty is (Learnable + Prior)
        # Some formulations use (Learnable - Prior) or just (Learnable + Prior).
        # Osband's "Randomized Prior" usually implies: prediction = Base + (Learnable + Prior)
        # The Training Loss will force Learnable to approximate (Target - Base - Prior).
        epistemic_offset = self.prior_scale * (learnable_out + prior_out)
        
        return mu + epistemic_offset
4. The Training Loss (EnnLoss)The magic of the Epinet happens in the loss function. We do not just minimize MSE on the mean. We minimize the error of the sampled hypothesis.$$\mathcal{L}(\theta, \phi) = \frac{1}{B} \sum_{i=1}^B \mathbb{E}_{z \sim \mathcal{N}(0, I)} \left[ || y_i - f(x_i, z) ||^2_2 \right] + \text{Regularization}$$Key Training Logic:Sample a batch of data $(x, y)$.Sample a batch of indices $z$ (same size as data batch).Forward pass: $\hat{y} = \text{Epinet}(x, z)$.Backprop the MSE between $\hat{y}$ and $y$.Effect: This forces the network to learn that for a random z, it should produce a valid prediction. Because the Prior Network is random and added, the Learnable Network must "work around" the Prior to fit the data.In regions with lots of data, $\eta_\phi$ learns to cancel out the variance of $\pi_{\text{fixed}}$, collapsing the uncertainty.In regions with no data, gradients don't flow, so $\eta_\phi$ stays near initialization. The output is dominated by the un-cancelled $\pi_{\text{fixed}}$, resulting in high variance (uncertainty) as desired.5. Verification Considerations (Zonotopes)To support your specific research method (Zonotope propagation):Piecewise Linear: Ensure all activations are ReLUs. Avoid Tanh/Sigmoid.Zonotope Input: The function signature for verification will differ slightly.Standard: (x_tensor, z_tensor) -> y_tensorVerification: (x_zonotope, z_set) -> y_zonotopeZ-Space: In verification, $z$ is not a sampled vector. It is treated as a set of generators (noise terms) added to the input zonotope. The identity function propagates $z$ forward.Propagating Z in Verification:You effectively treat the input to the network as a Zonotope over the joint space $\mathbb{R}^{\text{dim}(x) + \text{dim}(z)}$.The base_net only sees the $x$ dimensions.The enn_net sees the full joint zonotope.Linear layers work natively on Zonotopes.ReLU layers require the abstract domain transformation (e.g., DeepPoly or Lambda domain).