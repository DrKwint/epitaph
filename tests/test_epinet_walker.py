import jax.numpy as jnp
from flax import nnx
from epitaph.models import Epinet
from epitaph.constellation import NetworkWalker, StarTree
from epitaph.star import Star
from epitaph.polytope import Polytope
from epitaph.affine import Affine


def test_epinet_walker_small_network():
    """
    Demonstrates how to use NetworkWalker to verify a sub-component of Epinet.
    """
    print("=== Testing Epinet with NetworkWalker ===")

    # 1. Initialize a small Epinet
    # We use small dimensions to keep the tree manageable and readable
    input_dim = 2
    z_dim = 1
    out_dim = 1

    # Initialize model with RNG
    rngs = nnx.Rngs(0)
    model = Epinet(
        in_features=input_dim,
        out_features=out_dim,
        z_dim=z_dim,
        base_width=4,
        base_depth=1,
        enn_width=4,
        enn_depth=1,
        rngs=rngs,
    )

    # 2. Select the 'learnable_enn' component to verify
    # NetworkWalker is designed for Sequential models (Linear -> ReLU -> ...)
    # The learnable_enn component of Epinet fits this structure perfectly.
    target_net = model.learnable_enn

    # Calculate input dimension for the ENN: [x, z, base_latent]
    # base_latent dimension is base_width (4)
    # x (2) + z (1) + base_latent (4) = 7
    enn_input_dim = input_dim + z_dim + 4

    print(f"Target Network: {target_net}")
    print(f"ENN Input Dimension: {enn_input_dim}")

    # 3. Initialize NetworkWalker
    # This parses the network structure into layers and neurons
    walker = NetworkWalker(target_net, input_shape=(enn_input_dim,))
    print(f"Walker initialized. Total neurons to split: {walker.total_neurons}")
    print(f"Layers detected: {len(walker.layers)}")

    # 4. Define Input Set (Star)
    # Create a small hypercube around the origin: -0.1 <= x <= 0.1
    lb = -0.1 * jnp.ones(enn_input_dim)
    ub = 0.1 * jnp.ones(enn_input_dim)

    input_poly = Polytope.box(enn_input_dim, (lb, ub))
    input_trans = Affine.identity(enn_input_dim)
    root_star = Star(input_poly, input_trans)

    # 5. Create StarTree
    tree = StarTree(root_star)

    # 6. Traverse the tree (DFS)
    print("\nStarting DFS Traversal of the reachability tree...")
    stack = [tree.root]
    leaves = []
    nodes_visited = 0

    while stack:
        node = stack.pop()
        nodes_visited += 1

        # Get successors (branches) from the walker
        # This applies the linear layers and splits on ReLUs
        children = walker.get_successors(node)

        if not children:
            # If no children, we check if we reached the full depth (end of network)
            if len(node.path) == walker.total_neurons:
                leaves.append(node)
        else:
            for child in children.values():
                stack.append(child)

    print(f"Traversal complete.")
    print(f"Nodes visited: {nodes_visited}")
    print(f"Leaf nodes found: {len(leaves)}")

    # 7. Inspect Results
    print("\n--- Leaf Node Analysis ---")
    for i, leaf in enumerate(leaves):
        # Apply the final output head (Linear layer) to get the final set
        output_star = walker.apply_final_head(leaf)

        # Check if the set is empty (infeasible path)
        is_empty = output_star.is_empty()

        status = "EMPTY" if is_empty else "FEASIBLE"
        print(f"Leaf {i}: Path='{leaf.path}' [{status}]")

        if not is_empty:
            # For feasible paths, we can inspect the output set
            # e.g., calculate the Chebyshev radius as a proxy for size
            radius = output_star.output_set_chebyshev_radius()
            print(f"  -> Output Set Radius: {radius:.6f}")


if __name__ == "__main__":
    test_epinet_walker_small_network()