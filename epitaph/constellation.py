from __future__ import annotations
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from typing import Iterator
from .star import Star
from .affine import Affine


class StarNode:
    """
    A node in the reachability tree representing a specific prefix of activations.

    Attributes:
        star (Star): The geometric star set at this node.
        path (str): The activation string (prefix) reaching this node (e.g., "101").
        parent (Optional[StarNode]): Pointer to parent for backtracking.
        children (Dict[str, StarNode]): Lazy dictionary of child nodes.
        depth (int): Depth in the tree.
    """

    def __init__(
        self,
        star: Star,
        path: str = "",
        parent: Optional[StarNode] = None,
        depth: int = 0,
    ):
        self.star = star
        self.path = path
        self.parent = parent
        self.children: Dict[str, StarNode] = {}
        self.depth = depth

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def apply_affine(self, transform: Affine) -> StarNode:
        """
        Extends the tree deterministically with an affine transformation
        (e.g., passing through a linear layer).
        Returns a new child node (this does not branch).
        """
        new_star = self.star.map_affine(transform)

        # In a prefix tree, affine transforms often don't add to the binary string,
        # but you could add a special token if needed. We'll keep the path same.
        child = StarNode(
            star=new_star, path=self.path, parent=self, depth=self.depth + 1
        )
        # We assume generic 'next' key for deterministic steps, or just return it
        # without attaching if you only want to track branching.
        # Here we attach it for tree completeness.
        self.children["affine"] = child
        return child

    def branch(self, dim: int) -> Dict[str, StarNode]:
        """
        Attempts to split this node on a specific dimension (ReLU neuron).

        Returns:
            Dict mapping '0' (inactive) and '1' (active) to valid StarNodes.
            If a branch is infeasible (empty star), it is excluded from the dict.
        """
        # If we already expanded this node, return existing children
        if self.children:
            return {k: v for k, v in self.children.items() if k in ["0", "1"]}

        # Use the underlying Star's split logic
        # map_steprelu returns a list of NON-EMPTY stars.
        resulting_stars = self.star.map_steprelu(dim)

        valid_branches = {}

        for s in resulting_stars:
            # We determine which branch this is by looking at the last char
            # of the activation_pattern appended by map_steprelu
            if s.activation_pattern is None:
                continue

            token = s.activation_pattern[-1]  # '0' or '1'

            child_node = StarNode(
                star=s, path=self.path + token, parent=self, depth=self.depth + 1
            )
            self.children[token] = child_node
            valid_branches[token] = child_node

        return valid_branches

    def __repr__(self):
        return f"<StarNode path='{self.path}' empty={self.star.is_empty()}>"


class StarTree:
    """
    Manager for the incremental Star search tree.
    Allows walking the tree via prefixes.
    """

    def __init__(self, root_star: Star):
        # Ensure the root star has an activation pattern init
        if root_star.activation_pattern is None:
            root_star.activation_pattern = ""

        self.root = StarNode(root_star, path="")

    def walk(self, prefix: str) -> Optional[StarNode]:
        """
        Attempts to walk a specific binary string prefix from the root.

        Args:
            prefix: A string of '0's and '1's.

        Returns:
            The StarNode at that prefix, or None if that path is infeasible/pruned.
        """
        current = self.root

        # Note: This assumes the tree represents a sequence of ReLU splits
        # corresponding exactly to the characters in `prefix`.
        # If your network has layers, you might need to interleave apply_affine calls.

        for i, token in enumerate(prefix):
            # We assume the user knows which dimension to split on at this depth.
            # If the dimensions correspond 1:1 with string index:
            branches = current.branch(dim=i)

            if token not in branches:
                return None  # Infeasible path

            current = branches[token]

        return current

    def traverse_dfs(self, node: Optional[StarNode] = None) -> Iterator[StarNode]:
        """
        Yields all feasible nodes in the tree using Depth First Search.
        Useful for exhaustive verification or finding bounds.
        """
        if node is None:
            node = self.root

        yield node

        # We need to know how to expand.
        # If this is a generic traverser, we need a strategy.
        # This is a placeholder for where you'd inject the network structure logic.
        pass


@dataclass
class LayerMeta:
    """Metadata for a single learnable layer followed by a ReLU."""

    weights: Affine
    width: int
    layer_idx: int

    # Range of depths in the tree that this layer covers
    # e.g., if global_start_idx is 0 and width is 64, this layer handles depths 0..63
    global_start_idx: int


class NetworkWalker:
    """
    Adapts a Flax NNX model to drive the expansion of a StarTree.
    It maps the flat 'depth' of the tree to the specific (Layer, Neuron) architecture.
    """

    def __init__(self, model: nnx.Module, input_shape: Tuple[int, ...]):
        self.layers: List[LayerMeta] = []
        self.total_neurons = 0
        self.final_head: Optional[Affine] = None

        # 1. Parse the Model Structure
        # We perform a dummy pass or inspect the sequential structure to flatten it.
        # This assumes a structure of [Linear, Relu, Linear, Relu, ..., Linear]
        self._parse_model(model, input_shape)

    def _parse_model(self, model: nnx.Module, input_shape: Tuple[int, ...]):
        """
        Extracts weights and shapes.
        Note: This is a simplified parser for Sequential models.
        """
        # Hacky introspection for NNX Sequential - assumes standard list of layers
        # In production, you might want to use graph traversal or just manual specification.
        if hasattr(model, "layers"):
            layers = model.layers
        else:
            # Fallback for single container or custom module
            layers = [model]

        current_affine = None

        for layer in layers:
            if isinstance(layer, nnx.Linear):
                # Extract JAX arrays from NNX variables
                # Accessing .get_value() directly if they are State/Param, or just the array
                kernel = (
                    layer.kernel.get_value()
                    if isinstance(layer.kernel, nnx.Param)
                    else layer.kernel
                )
                bias = layer.bias.get_value() if layer.bias is not None else layer.bias

                # Transpose kernel because nnx.Linear is x @ A
                # Our Affine class expects A @ x (standard math notation)
                A = jnp.transpose(kernel)
                b = bias
                assert b is not None
                current_affine = Affine(A, b)

            elif layer is nnx.relu or isinstance(layer, (nnx.relu.__class__,)):
                if current_affine is None:
                    raise ValueError(
                        "Found ReLU before Linear layer. Cannot construct star."
                    )

                # We found a Linear -> ReLU block
                width = current_affine.A.shape[0]

                meta = LayerMeta(
                    weights=current_affine,
                    width=width,
                    layer_idx=len(self.layers),
                    global_start_idx=self.total_neurons,
                )
                self.layers.append(meta)
                self.total_neurons += width
                current_affine = None  # Consumed

        # If there is a dangling affine at the end (the output head), store it
        if current_affine is not None:
            self.final_head = current_affine

    def get_successors(self, node: StarNode) -> Dict[str, StarNode]:
        """
        Given a node, applies the network logic to generate children.
        Automatically handles:
        1. Applying Linear Weights (if we are at the start of a layer)
        2. Splitting the ReLU (generating the branches)
        """
        current_depth = len(node.path)

        # 1. Check if we are done
        if current_depth >= self.total_neurons:
            # Reached leaf of the activation patterns.
            # We can optionally apply the final head here if it's not done yet.
            return {}

        # 2. Identify which layer and neuron we are at
        active_layer = None
        for layer in self.layers:
            if (
                layer.global_start_idx
                <= current_depth
                < (layer.global_start_idx + layer.width)
            ):
                active_layer = layer
                break

        if active_layer is None:
            return {}  # Should not happen unless tree depth exceeds model

        neuron_idx = current_depth - active_layer.global_start_idx

        # 3. Prepare the Star
        # If this is the FIRST neuron of the layer, we must apply the layer's affine transform first.
        # This moves the Star from the previous layer's post-activation space
        # to the current layer's pre-activation space.
        current_star = node.star

        if neuron_idx == 0:
            # Transition: Apply weights Wx + b
            current_star = current_star.map_affine(active_layer.weights)

            # Note: We do NOT create a new 'node' for this intermediate step
            # because 'node' represents a prefix in the activation string.
            # The affine transform is implicit in the transition between layers.

        # 4. Perform the Split (Branching)
        # map_steprelu splits on dimension 'neuron_idx' of the current star
        # It handles updating the constraints (input set) and the transform (zeroing rows).
        split_stars = current_star.map_steprelu(neuron_idx)

        children = {}
        for s in split_stars:
            # Extract the last char ('0' or '1') added by map_steprelu
            if s.activation_pattern is None:
                continue
            token = s.activation_pattern[-1]

            child_node = StarNode(
                star=s, path=node.path + token, parent=node, depth=node.depth + 1
            )
            children[token] = child_node

        # Update the parent's children registry
        node.children.update(children)

        return children

    def apply_final_head(self, node: StarNode) -> Star:
        """
        Helper to get the actual output set for a leaf node.
        Applies the final Linear layer (if it exists) to the Star.
        """
        if self.final_head:
            return node.star.map_affine(self.final_head)
        return node.star
