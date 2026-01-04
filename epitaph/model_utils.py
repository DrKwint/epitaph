import jax
import jax.numpy as jnp
from typing import List, Any
from .structures import ConstrainedZonotope
from . import ops
from .models import Epinet
from flax import nnx


def lift_sequential(layers: List[Any], cz: ConstrainedZonotope) -> ConstrainedZonotope:
    """
    Lifts a sequence of NNX layers to work on Constrained Zonotopes.
    """
    for layer in layers:
        if isinstance(layer, nnx.Linear):
            w = layer.kernel[...]
            b = layer.bias[...] if layer.bias is not None else None
            cz = ops.propagate_linear(cz, w, b)
        elif layer == nnx.relu or (hasattr(layer, "func") and layer.func == nnx.relu):
            cz = ops.propagate_relu(cz)
        elif hasattr(layer, "__call__") and layer.__name__ == "relu":
            cz = ops.propagate_relu(cz)
    return cz


def lift_epinet_propagation(
    model: Epinet, cz: ConstrainedZonotope
) -> ConstrainedZonotope:
    """
    Propagates a Constrained Zonotope through the Base+Prior Epinet.

    Args:
        model: Epinet instance.
        cz: Input zonotope over [x, z].
            x dim = model.base_net.layers[0].in_features
            z dim = model.z_dim
    """

    # 1. Split Input
    # Get first layer from base_repr to determine input dimensions
    base_layers = model.base_repr.layers
    
    if hasattr(base_layers[0], "in_features"):
        in_features = base_layers[0].in_features
    else:
        in_features = base_layers[0].kernel[...].shape[0]

    # Slice cz for Base Net
    cz_x_center = cz.center[..., :in_features]
    cz_x_gens = cz.generators[..., :in_features]

    cz_x = ConstrainedZonotope(
        center=cz_x_center,
        generators=cz_x_gens,
        constraints_eq_A=cz.constraints_eq_A,
        constraints_eq_b=cz.constraints_eq_b,
        constraints_ineq_A=cz.constraints_ineq_A,
        constraints_ineq_b=cz.constraints_ineq_b,
    )

    # 2. Lift Base Representation Network (all layers of base_repr)
    base_latent_cz = lift_sequential(base_layers, cz_x)
    
    # Apply base_head to get mu_cz
    w_head = model.base_head.kernel[...]
    b_head = model.base_head.bias[...] if model.base_head.bias is not None else None
    mu_cz = ops.propagate_linear(base_latent_cz, w_head, b_head)
    
    # 3. Concatenate [x, z, base_latent] for epistemic networks
    # Concatenate centers
    enn_center = jnp.concatenate([cz.center, base_latent_cz.center], axis=-1)
    
    # Concatenate generators
    # cz.generators: (batch, n_gen, in_features + z_dim)
    # base_latent_cz.generators: (batch, n_gen_base, base_width)
    # Need to align generators
    n_in_gens = cz.n_gen
    n_base_gens = base_latent_cz.n_gen
    
    # Pad cz generators to match total generator count
    if n_base_gens > n_in_gens:
        # Need to add zero generators to cz for the new base generators
        pad_gens = jnp.zeros(
            (cz.generators.shape[0], n_base_gens - n_in_gens, cz.generators.shape[2])
        )
        cz_gens_padded = jnp.concatenate([cz.generators, pad_gens], axis=1)
    else:
        cz_gens_padded = cz.generators
    
    # Pad base_latent generators to match
    if n_in_gens > n_base_gens:
        pad_base = jnp.zeros(
            (base_latent_cz.generators.shape[0], n_in_gens - n_base_gens, base_latent_cz.generators.shape[2])
        )
        base_latent_gens_padded = jnp.concatenate([base_latent_cz.generators, pad_base], axis=1)
    else:
        base_latent_gens_padded = base_latent_cz.generators
    
    # Concatenate along feature dimension
    enn_generators = jnp.concatenate([cz_gens_padded, base_latent_gens_padded], axis=-1)
    
    # Create combined zonotope for epistemic networks
    # Constraints: use the more restrictive set (from base_latent_cz which has gone through more layers)
    # Pad constraints to match new generator count
    total_gens = max(n_in_gens, n_base_gens)
    
    # Use cz constraints (input constraints) and pad them
    if cz.constraints_eq_A.shape[-1] < total_gens:
        pad_eq = jnp.zeros(
            (cz.constraints_eq_A.shape[0], cz.constraints_eq_A.shape[1], total_gens - cz.constraints_eq_A.shape[-1])
        )
        enn_eq_A = jnp.concatenate([cz.constraints_eq_A, pad_eq], axis=-1)
    else:
        enn_eq_A = cz.constraints_eq_A
    
    if cz.constraints_ineq_A.shape[-1] < total_gens:
        pad_ineq = jnp.zeros(
            (cz.constraints_ineq_A.shape[0], cz.constraints_ineq_A.shape[1], total_gens - cz.constraints_ineq_A.shape[-1])
        )
        enn_ineq_A = jnp.concatenate([cz.constraints_ineq_A, pad_ineq], axis=-1)
    else:
        enn_ineq_A = cz.constraints_ineq_A
    
    cz_enn_input = ConstrainedZonotope(
        center=enn_center,
        generators=enn_generators,
        constraints_eq_A=enn_eq_A,
        constraints_eq_b=cz.constraints_eq_b,
        constraints_ineq_A=enn_ineq_A,
        constraints_ineq_b=cz.constraints_ineq_b,
    )
    
    # 4. Lift Learnable Epistemic Net
    learnable_cz = lift_sequential(model.learnable_enn.layers, cz_enn_input)

    # 5. Lift Prior Net
    prior_cz = lift_sequential(model.prior_enn.layers, cz_enn_input)

    # 5. Combine and Unify Generators
    # We must unify the generators from the three branches (Base, Learnable, Prior).
    # Shared generators (input noise) coincide.
    # Independent generators (ReLU noise) are concatenated.

    n_in_gens = cz.n_gen

    # Extract Shared Generators
    G_base_pure = mu_cz.generators[..., :n_in_gens, :]
    G_enn_pure = learnable_cz.generators[..., :n_in_gens, :]
    G_prior_pure = prior_cz.generators[..., :n_in_gens, :]

    # Global Shared Generators
    G_total_shared = G_base_pure + model.prior_scale * (G_enn_pure + G_prior_pure)

    # Extract Independent Generators (ReLU noise)
    G_base_noise = mu_cz.generators[..., n_in_gens:, :]
    G_enn_noise = learnable_cz.generators[..., n_in_gens:, :]
    G_prior_noise = prior_cz.generators[..., n_in_gens:, :]

    # Scale Independent Generators
    G_enn_noise_scaled = model.prior_scale * G_enn_noise
    G_prior_noise_scaled = model.prior_scale * G_prior_noise

    # Concatenate: [Shared, BaseNoise, EnnNoise, PriorNoise]
    new_generators = jnp.concatenate(
        [G_total_shared, G_base_noise, G_enn_noise_scaled, G_prior_noise_scaled], axis=1
    )

    # Calculate Center
    new_center = mu_cz.center + model.prior_scale * (
        learnable_cz.center + prior_cz.center
    )

    # Combine Constraints
    # We create a large constraint matrix that includes constraints from all branches.
    # Shared constraints are taken from the Base branch.
    # Independent constraints (on new generators) are padded and concatenated.

    def pad_constraints(
        A: jax.Array,
        b: jax.Array,
        offset: int,
        total_gens: int,
        current_gens_count: int,
        n_in_gens: int,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Pads constraint matrix A to map local generators to the global generator vector.
        Global [0...K-1] is shared (input noise).
        Global [K+offset ... K+offset+N_local] is the local noise.
        """
        A_shared = A[..., :n_in_gens]
        A_local = A[..., n_in_gens:]

        batch_size = A.shape[0]
        n_cons = A.shape[1]
        A_new = jnp.zeros((batch_size, n_cons, total_gens))

        # Set shared constraints (columns 0 to K)
        A_new = A_new.at[..., :n_in_gens].set(A_shared)

        # Set local constraints (columns K+offset to K+offset+N_local)
        n_local = A.shape[-1] - n_in_gens
        start = n_in_gens + offset
        end = start + n_local
        A_new = A_new.at[..., start:end].set(A_local)

        return A_new, b

    # Count sizes
    N_base = G_base_noise.shape[1]
    N_enn = G_enn_noise.shape[1]
    N_prior = G_prior_noise.shape[1]

    total_gens = n_in_gens + N_base + N_enn + N_prior

    # Process EQ Constraints
    # We assume input constraints are correctly handled by taking them from each branch
    # (function pad_constraints copies the shared part from each input A).
    # Since they are identical, they will provide redundant constraints on the shared variables.
    # This is safe for solvers (just duplicated rows).
    A_base_new, b_base_new = pad_constraints(
        mu_cz.constraints_eq_A,
        mu_cz.constraints_eq_b,
        0,
        total_gens,
        0,
        n_in_gens,
    )
    A_enn_new, b_enn_new = pad_constraints(
        learnable_cz.constraints_eq_A,
        learnable_cz.constraints_eq_b,
        N_base,
        total_gens,
        0,
        n_in_gens,
    )
    A_prior_new, b_prior_new = pad_constraints(
        prior_cz.constraints_eq_A,
        prior_cz.constraints_eq_b,
        N_base + N_enn,
        total_gens,
        0,
        n_in_gens,
    )

    new_eq_A = jnp.concatenate([A_base_new, A_enn_new, A_prior_new], axis=1)
    new_eq_b = jnp.concatenate([b_base_new, b_enn_new, b_prior_new], axis=1)

    # Process INEQ Constraints
    A_base_in, b_base_in = pad_constraints(
        mu_cz.constraints_ineq_A, mu_cz.constraints_ineq_b, 0, total_gens, 0, n_in_gens
    )
    A_enn_in, b_enn_in = pad_constraints(
        learnable_cz.constraints_ineq_A,
        learnable_cz.constraints_ineq_b,
        N_base,
        total_gens,
        0,
        n_in_gens,
    )
    A_prior_in, b_prior_in = pad_constraints(
        prior_cz.constraints_ineq_A,
        prior_cz.constraints_ineq_b,
        N_base + N_enn,
        total_gens,
        0,
        n_in_gens,
    )

    new_ineq_A = jnp.concatenate([A_base_in, A_enn_in, A_prior_in], axis=1)
    new_ineq_b = jnp.concatenate([b_base_in, b_enn_in, b_prior_in], axis=1)

    return ConstrainedZonotope(
        center=new_center,
        generators=new_generators,
        constraints_eq_A=new_eq_A,
        constraints_eq_b=new_eq_b,
        constraints_ineq_A=new_ineq_A,
        constraints_ineq_b=new_ineq_b,
    )
