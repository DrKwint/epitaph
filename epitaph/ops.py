import jax
import jax.numpy as jnp
from typing import Optional, List
from .structures import ConstrainedZonotope


def propagate_linear(
    cz: ConstrainedZonotope, weight: jax.Array, bias: Optional[jax.Array] = None
) -> ConstrainedZonotope:
    """
    Propagates a constrained zonotope through a linear layer: x' = Ax + b.

    Args:
        cz: Input ConstrainedZonotope.
        weight: Weight matrix (out_dim, in_dim).
        bias: Bias vector (out_dim,). Optional.

    Returns:
        Transformed ConstrainedZonotope.
    """
    new_center = cz.center @ weight
    new_generators = cz.generators @ weight

    if bias is not None:
        new_center = new_center + bias

    return cz.replace(center=new_center, generators=new_generators)


def compute_bounds(cz: ConstrainedZonotope) -> tuple[jax.Array, jax.Array]:
    """
    Computes component-wise lower and upper bounds of the zonotope.

    Approximation:
    lb = center - sum(|generators|, axis=1)
    ub = center + sum(|generators|, axis=1)

    This is a sound over-approximation (bounding box of the unconstrained zonotope).
    It ignores the constraints, so it might be loose, but it is fast.
    """
    delta = jnp.sum(jnp.abs(cz.generators), axis=1)  # Sum over generators (axis 1)
    lb = cz.center - delta
    ub = cz.center + delta
    return lb, ub


def apply_constraints(
    cz: ConstrainedZonotope, H: jax.Array, d: jax.Array
) -> ConstrainedZonotope:
    """
    Slices the zonotope with linear constraints: Hx <= d.

    The constraint in x-space:
    H(c + G*xi) <= d
    (HG)xi <= d - Hc

    We appends these rows to strictly inequality constraints.
    """
    # H: (n_cons_new, n_dim)
    # d: (n_cons_new)

    # In batch context:
    # H could be (batch, n_cons_new, n_dim) or (n_cons_new, n_dim) broadcasted.
    # d could be (batch, n_cons_new) or (n_cons_new) broadcasted.
    # We assume H, d are broadcastable or already batched.
    # For simplicitly, assume H and d are constant across batch or matched.

    # Calculate A_new = H @ G^T.
    # G is (batch, n_gen, n_dim).
    # We want A_new to be (batch, n_cons_new, n_gen).

    # Expand H if needed
    if H.ndim == 2:
        H_batch = jnp.expand_dims(H, 0)  # (1, n_cons, n_dim)
    else:
        H_batch = H

    if d.ndim == 1:
        d_batch = jnp.expand_dims(d, 0)  # (1, n_cons)
    else:
        d_batch = d

    A_new = jnp.einsum("bij,bkj->bik", H_batch, cz.generators)

    # b_new = d - Hc
    Hc = jnp.einsum("bij,bj->bi", H_batch, cz.center)
    b_new = d_batch - Hc

    # Concatenate with existing inequalities
    new_ineq_A = jnp.concatenate([cz.constraints_ineq_A, A_new], axis=1)
    new_ineq_b = jnp.concatenate([cz.constraints_ineq_b, b_new], axis=1)

    return cz.replace(constraints_ineq_A=new_ineq_A, constraints_ineq_b=new_ineq_b)


def propagate_relu(cz: ConstrainedZonotope) -> ConstrainedZonotope:
    """
    Propagates CZ through ReLU using Lambda-Zonotope relaxation (DeepZ).
    """
    lb, ub = compute_bounds(cz)

    # Prevent division by zero
    diff = ub - lb
    safe_diff = jnp.where(diff == 0, 1.0, diff)

    # Slopes and offsets
    # Case 1: u <= 0 -> slope 0, offset 0
    # Case 2: l >= 0 -> slope 1, offset 0
    # Case 3: l < 0 < u -> slope u/(u-l), offset ...

    is_blocking = ub <= 0
    is_passing = lb >= 0
    is_unstable = ~(is_blocking | is_passing)

    # Calculate lambda (slope)
    slope = jnp.where(is_passing, 1.0, 0.0)
    slope = jnp.where(is_unstable, ub / safe_diff, slope)

    # Calculate mu (offset)
    # DeepZ: mu = - l * u / (2 * (u - l))
    mu = -lb * ub / (2 * safe_diff)
    mu = jnp.where(is_unstable, mu, 0.0)

    # Update Center
    new_center = cz.center * slope + mu

    # Update Existing Generators
    # Broadcast slope: (batch, 1, n_dim)
    new_generators = cz.generators * jnp.expand_dims(slope, 1)

    # Add New Generators (Error terms)
    # Each row i has mu[i] at col i, 0 elsewhere.
    new_error_gens = jax.vmap(jnp.diag)(mu)

    # Concatenate
    combined_generators = jnp.concatenate([new_generators, new_error_gens], axis=1)

    # Constraints A, b must be padded with zeros for the new generators
    # A: (batch, n_cons, n_gen) -> (batch, n_cons, n_gen + n_dim)
    # combined_generators: (batch, n_gen_new, n_dim). We want n_gen_new.
    pad_eq_A = jnp.zeros(
        (
            cz.constraints_eq_A.shape[0],
            cz.constraints_eq_A.shape[1],
            combined_generators.shape[1] - cz.constraints_eq_A.shape[-1],
        )
    )
    new_eq_A = jnp.concatenate([cz.constraints_eq_A, pad_eq_A], axis=2)

    pad_ineq_A = jnp.zeros(
        (
            cz.constraints_ineq_A.shape[0],
            cz.constraints_ineq_A.shape[1],
            combined_generators.shape[1] - cz.constraints_ineq_A.shape[-1],
        )
    )
    new_ineq_A = jnp.concatenate([cz.constraints_ineq_A, pad_ineq_A], axis=2)

    return cz.replace(
        center=new_center,
        generators=combined_generators,
        constraints_eq_A=new_eq_A,
        constraints_ineq_A=new_ineq_A,
    )


def add_zonotopes(cz1: ConstrainedZonotope, cz2: ConstrainedZonotope) -> ConstrainedZonotope:
    """
    Computes the Minkowski sum of two constrained zonotopes: Z = Z1 + Z2.
    
    If both operate on the same xi dimension, they are concatenated in generator space.
    """
    new_center = cz1.center + cz2.center
    
    # Concatenate generators
    new_generators = jnp.concatenate([cz1.generators, cz2.generators], axis=1)
    
    # Pad constraints
    def pad_A(A1, A2):
        # A1: (B, C1, G1), A2: (B, C2, G2)
        # We want to combine them such that xi1 and xi2 are independent.
        # But wait, Minkowski sum means we have xi1 and xi2.
        # So the new xi is [xi1, xi2].
        B, C1, G1 = A1.shape
        _, C2, G2 = A2.shape
        
        # New A should be (B, C1 + C2, G1 + G2)
        top_left = A1
        top_right = jnp.zeros((B, C1, G2))
        bottom_left = jnp.zeros((B, C2, G1))
        bottom_right = A2
        
        row1 = jnp.concatenate([top_left, top_right], axis=2)
        row2 = jnp.concatenate([bottom_left, bottom_right], axis=2)
        return jnp.concatenate([row1, row2], axis=1)

    new_eq_A = pad_A(cz1.constraints_eq_A, cz2.constraints_eq_A)
    new_eq_b = jnp.concatenate([cz1.constraints_eq_b, cz2.constraints_eq_b], axis=1)
    
    new_ineq_A = pad_A(cz1.constraints_ineq_A, cz2.constraints_ineq_A)
    new_ineq_b = jnp.concatenate([cz1.constraints_ineq_b, cz2.constraints_ineq_b], axis=1)
    
    return ConstrainedZonotope(
        center=new_center,
        generators=new_generators,
        constraints_eq_A=new_eq_A,
        constraints_eq_b=new_eq_b,
        constraints_ineq_A=new_ineq_A,
        constraints_ineq_b=new_ineq_b
    )


def concatenate_zonotopes(czs: List[ConstrainedZonotope]) -> ConstrainedZonotope:
    """
    Concatenates multiple zonotopes along the dimension axis.
    """
    if not czs:
        raise ValueError("Empty list of zonotopes")
    
    batch_size = czs[0].center.shape[0]
    
    new_center = jnp.concatenate([c.center for c in czs], axis=-1)
    
    # Generators: we need to handle different xi indices for each zonotope.
    # New generator matrix will be block-diagonal.
    total_gens = sum(c.n_gen for c in czs)
    total_dim = sum(c.n_dim for c in czs)
    
    new_generators = jnp.zeros((batch_size, total_gens, total_dim))
    
    curr_gen = 0
    curr_dim = 0
    for cz in czs:
        # Put generators of cz into the correct block
        # new_generators[:, curr_gen:curr_gen+n_gen, curr_dim:curr_dim+n_dim] = cz.generators
        # Use dynamic update since we are in JAX (actually this is construction)
        # For simplicity, let's use a list of slices and concatenate.
        pass
    
    # Better construction:
    gen_blocks = []
    curr_dim = 0
    for i, cz in enumerate(czs):
        # Create a block of shape (B, n_gen_i, total_dim)
        prefix = jnp.zeros((batch_size, cz.n_gen, curr_dim))
        suffix = jnp.zeros((batch_size, cz.n_gen, total_dim - curr_dim - cz.n_dim))
        block = jnp.concatenate([prefix, cz.generators, suffix], axis=2)
        gen_blocks.append(block)
        curr_dim += cz.n_dim
        
    new_generators = jnp.concatenate(gen_blocks, axis=1)
    
    # Constraints: also block diagonal
    def combine_constraints(alist, blist):
        B = batch_size
        total_c = sum(a.shape[1] for a in alist)
        total_g = sum(a.shape[2] for a in alist)
        
        new_A = jnp.zeros((B, total_c, total_g))
        a_blocks = []
        curr_g = 0
        for a in alist:
            p_g = jnp.zeros((B, a.shape[1], curr_g))
            s_g = jnp.zeros((B, a.shape[1], total_g - curr_g - a.shape[2]))
            block = jnp.concatenate([p_g, a, s_g], axis=2)
            a_blocks.append(block)
            curr_g += a.shape[2]
            
        new_A = jnp.concatenate(a_blocks, axis=1)
        new_b = jnp.concatenate(blist, axis=1)
        return new_A, new_b

    new_eq_A, new_eq_b = combine_constraints([c.constraints_eq_A for c in czs], [c.constraints_eq_b for c in czs])
    new_ineq_A, new_ineq_b = combine_constraints([c.constraints_ineq_A for c in czs], [c.constraints_ineq_b for c in czs])
    
    return ConstrainedZonotope(
        center=new_center,
        generators=new_generators,
        constraints_eq_A=new_eq_A,
        constraints_eq_b=new_eq_b,
        constraints_ineq_A=new_ineq_A,
        constraints_ineq_b=new_ineq_b
    )


def compute_cz_volume_approx(cz: ConstrainedZonotope) -> jax.Array:
    """
    Computes a differentiable approximation of the constrained zonotope volume.
    
    Heuristic: Volume of the bounding box of the valid latent space xi.
    Valid xi satisfy:
    -1 <= xi <= 1
    A_eq * xi = b_eq
    A_in * xi <= b_in
    
    Since we can't solve LPs efficiently inside JIT (without complex ops),
    we use a soft Log-Barrier / Slack approach.
    
    Signal = Product of (1 - soft_violation_i)
    """
    # 1. Inequality Constraints: A_in * xi <= b_in
    # Since xi is in [-1, 1], the max value of A_in * xi is sum(abs(A_in), axis=2)
    # We want to know how much "room" we have: b_in - (A_in * xi)
    # But xi is a SET. A common heuristic for volume of a polytope Ax <= b 
    # intersected with a cube is to look at the "slack" of the center (xi=0)
    # and the sensitivity to generators.
    
    # Slack at origin (xi=0):
    slack_in = cz.constraints_ineq_b # (B, C_in)
    
    # Sensitivity (how much xi can push against the constraint):
    sensitivity_in = jnp.sum(jnp.abs(cz.constraints_ineq_A), axis=2) # (B, C_in)
    
    # A constraint is "tight" if sensitivity > slack.
    # Effective volume fraction for each constraint row:
    # If slack >> sensitivity, the constraint doesn't cut the hypercube much.
    # If slack << -sensitivity, the volume is zero.
    
    # Soft approximation: sigmoid((slack + sensitivity) / sensitivity) ?
    # Let's use a simpler one: relu(slack + sensitivity) / (2 * sensitivity) clipped.
    
    # For xi in [-1, 1]^G, the range of A*xi is [-sens, sens].
    # The valid part is [-sens, min(sens, slack)].
    # Length of valid interval in A-space: max(0, min(sens, slack) - (-sens)) = max(0, min(2*sens, slack + sens))
    # Fraction of A-space preserved: max(0, min(1.0, (slack + sens) / (2 * sens)))
    
    eps = 1e-6
    fraction_in = jnp.clip((slack_in + sensitivity_in) / (2 * sensitivity_in + eps), 0.0, 1.0)
    
    # 2. Equality Constraints: A_eq * xi = b_eq
    # Equality constraints reduce dimension. 
    # Heuristic: exp(- |b_eq| / sensitivity_eq)
    slack_eq = jnp.abs(cz.constraints_eq_b)
    sensitivity_eq = jnp.sum(jnp.abs(cz.constraints_eq_A), axis=2)
    fraction_eq = jnp.exp(-slack_eq / (sensitivity_eq + eps))
    
    # Combine (Geometric mean or product)
    # We use log-sum-exp or just product for simplicity
    total_fraction = jnp.prod(fraction_in, axis=1) * jnp.prod(fraction_eq, axis=1)
    
    # Initial volume is 2^n_gen
    initial_log_vol = cz.n_gen * jnp.log(2.0)
    
    return jnp.exp(initial_log_vol + jnp.log(total_fraction + eps))


def calculate_safety_probability(cz: ConstrainedZonotope) -> jax.Array:
    """
    Computes the probability of safety P_safe as the fraction of valid epistemic space.
    
    P_safe = V_valid / V_initial
    """
    # Valid volume approximation
    volume = compute_cz_volume_approx(cz)
    
    # Initial volume (unit hypercube [-1, 1]^G has volume 2^G)
    initial_volume = 2.0 ** cz.n_gen
    
    return jnp.clip(volume / initial_volume, 0.0, 1.0)


def calculate_gaussian_safety_prob(cz: ConstrainedZonotope, n_epistemic_dims: int) -> jax.Array:
    """
    Computes the probability of safety P(z in SafeZone) where z ~ N(0, I).
    
    We assume the first n_epistemic_dims generators of the zonotope correspond 
    to the epistemic indices xi_z, and the prior is xi_z ~ N(0, I).
    
    Warning: This assumes linear constraints s(z) <= d.
    """
    from jax.scipy.special import erfc
    
    # 1. Inequality Constraints: A_in * xi <= b_in
    # Each row specifies a condition on the combination of generators.
    # In the Epinet context, the state s is an affine function of z:
    # s = c + G_z * xi_z + G_e * xi_e
    # Our constraints are H s <= d -> (H G_z) xi_z + (H G_e) xi_e <= d - H c
    
    # Let A = cz.constraints_ineq_A[:, :, :n_epistemic_dims] (epistemic part)
    # Let B = cz.constraints_ineq_b
    # We want P(A * xi_z <= B) where xi_z ~ N(0, I)
    
    A = cz.constraints_ineq_A[:, :, :n_epistemic_dims] # (Batch, N_cons, N_z)
    B = cz.constraints_ineq_b # (Batch, N_cons)
    
    # Each constraint row i: sum_j(A_ij * xi_zj) <= B_i
    # The sum u_i = sum_j(A_ij * xi_zj) is Gaussian.
    # u_i ~ N(0, sigma_i^2) where sigma_i^2 = sum_j(A_ij^2)
    
    sigma_sq = jnp.sum(jnp.square(A), axis=2) # (Batch, N_cons)
    sigma = jnp.sqrt(jnp.maximum(sigma_sq, 1e-10))
    
    # P(u_i <= B_i) = 0.5 * (1 + erf(B_i / (sigma * sqrt(2))))
    def normal_cdf(x):
        return 0.5 * erfc(-x / jnp.sqrt(2.0))

    probs = normal_cdf(B / sigma) # (Batch, N_cons)
    
    # Combine constraints (assuming independence as a heuristic, or using product)
    p_safe = jnp.prod(probs, axis=1)
    
    return p_safe
