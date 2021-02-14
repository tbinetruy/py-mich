import ast
from typing import Any, List


# thanks http://dev.stephendiehl.com/numpile/ :)
def ast_to_tree(node, include_attrs=True):
    def _transform(node):
        if isinstance(node, ast.AST):
            fields = ((a, _transform(b)) for a, b in ast.iter_fields(node))
            if include_attrs:
                attrs = (
                    (a, _transform(getattr(node, a)))
                    for a in node._attributes
                    if hasattr(node, a)
                )
                return (node.__class__.__name__, dict(fields), dict(attrs))
            return (node.__class__.__name__, dict(fields))
        elif isinstance(node, list):
            return [_transform(x) for x in node]
        elif isinstance(node, str):
            return repr(node)
        return node

    if not isinstance(node, ast.AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)
    return _transform(node)


class Tree:
    def make_node(self, left, right):
        raise NotImplementedError

    def get_left(self, tree_node):
        raise NotImplementedError

    def get_right(self, tree_node):
        raise NotImplementedError

    def set_right(self, tree_node, value):
        raise NotImplementedError

    def get_leaf_from_element(self, element):
        return element

    def format_leaf(self, leaf):
        return leaf

    def list_to_tree(self, elements: List[Any]):
        tree_leaves = []
        for element in elements:
            leaf = self.get_leaf_from_element(element)
            if not len(tree_leaves):
                tree_leaves.append(self.make_node(left=leaf, right=None))
            elif not self.get_right(tree_leaves[-1]):
                self.set_right(tree_leaves[-1], leaf)
            else:
                tree_leaves.append(self.make_node(left=leaf, right=None))

        if not self.get_right(tree_leaves[-1]):
            tree_leaves[-1] = self.get_left(tree_leaves[-1])

        def construct_tree(tree_leaves):
            tree = []
            i = 0
            while i < len(tree_leaves):
                try:
                    tree.append(
                        self.make_node(
                            left=self.format_leaf(tree_leaves[i]),
                            right=self.format_leaf(tree_leaves[i + 1]),
                        )
                    )
                except IndexError:
                    tree.append(tree_leaves[i])
                i += 2
            if len(tree) == 1:
                return tree[0]
            else:
                return construct_tree(tree)

        return construct_tree(tree_leaves)
