import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional

import instr_types as t
from helpers import Tree


@dataclass
class Array:
    """Michelson list. Called array not to conflict with `typing.List`
    used in type annotations.

    Parameters
    ----------
    els: python list representing the Michelson list"""

    els: List[Any]


@dataclass
class Pair:
    """Michelson pair.

    Parameters
    ----------
    car: first element
    cdr: second element"""

    car: Any
    cdr: Any
    annotation: Optional[str] = None


@dataclass
class Some:
    """Michelson `option.some` type"""

    value: Any

    def __eq__(self, o):
        if type(o) == type(self):
            return self.value == o.value and self.value == o.value
        else:
            return False


@dataclass
class Or:
    """Michelson `or` type"""

    left: Any
    right: Any

    def __eq__(self, o):
        if type(o) == type(self):
            return self.left == o.left and self.right == o.right
        else:
            return False


@dataclass
class Left:
    """Michelson `LEFT` data type"""

    value: Any

    def __eq__(self, o):
        return type(o) == type(self) and self.value == o.value


@dataclass
class Right:
    """Michelson `RIGHT` data type"""

    value: Any

    def __eq__(self, o):
        return type(o) == type(self) and self.value == o.value


@dataclass
class FunctionPrototype:
    arg_type: t.Type
    return_type: t.Type





@dataclass
class Instr:
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]


@dataclass
class Entrypoint:
    prototype: FunctionPrototype
    instructions: List[Instr]


class ParameterTree(Tree):
    def make_node(self, left, right):
        return Or(left, right)

    def get_left(self, tree_node):
        return tree_node.left

    def get_right(self, tree_node):
        return tree_node.right

    def set_right(self, tree_node, value):
        tree_node.right = value

    def left_side_tree_height(self, tree, height=0):
        if type(tree) is not Or:
            return height
        else:
            return self.left_side_tree_height(self.get_left(tree), height + 1)

    def navigate_to_tree_leaf(self, tree, leaf_number, param):
        if type(tree) is not Or:
            return param

        left_max_leaf_number = 2 ** self.left_side_tree_height(tree.left)
        if leaf_number <= left_max_leaf_number:
            return Left(self.navigate_to_tree_leaf(tree.left, leaf_number, param))
        else:
            return Right(
                self.navigate_to_tree_leaf(
                    tree.right, leaf_number - left_max_leaf_number, param
                )
            )


class EntrypointTree(Tree):
    def make_node(self, left=None, right=None):
        if not left:
            left = []
        if not right:
            right = []
        return Instr("IF_LEFT", [left, right], {})

    def get_left(self, tree_node):
        return tree_node.args[0]

    def get_right(self, tree_node):
        return tree_node.args[1]

    def set_right(self, tree_node, value):
        tree_node.args[1] = value

    def get_leaf_from_element(self, element):
        return element.instructions

    def format_leaf(self, leaf):
        return leaf if type(leaf) == list else [leaf]


@dataclass
class Contract:
    storage: Any
    storage_type: t.Type
    entrypoints: Dict[str, Entrypoint]
    instructions: List[Instr]

    def add_entrypoint(self, name: str, entrypoint: Entrypoint):
        self.entrypoints[name] = entrypoint

    def make_contract_param(self, entrypoint_name, entrypoint_param):
        entrypoint_names = sorted(self.entrypoints.keys())
        if len(entrypoint_names) == 1:
            return entrypoint_param

        parameter_tree = ParameterTree()
        tree = parameter_tree.list_to_tree(entrypoint_names)
        entrypoint_index = entrypoint_names.index(entrypoint_name)
        return parameter_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )

    def get_storage_type(self):
        return self.storage_type

    def get_parameter_type(self):
        entrypoint_names = self.entrypoints.keys()
        if len(entrypoint_names) == 1:
            return self.entrypoints[entrypoint_names[0]].arg_type
        else:
            parameter_tree = ParameterTree()
            entrypoints = [
                    self.entrypoints[name].prototype.arg_type
                    for name in sorted(entrypoint_names)
                ]

            for i, entrypoint in enumerate(entrypoints):
                entrypoint.annotation = "%" + sorted(list(entrypoint_names))[i]

            return parameter_tree.list_to_tree(entrypoints)

    def get_contract_body(self):
        entrypoints = [
            self.entrypoints[name] for name in sorted(self.entrypoints.keys())
        ]
        entrypoint_tree = EntrypointTree()
        return entrypoint_tree.list_to_tree(entrypoints)


class TestContract(unittest.TestCase):
    def test_get_contract_body(self):
        contract = Contract(
            storage=10,
            storage_type=t.Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 1], {}),
                    ],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 2], {}),
                    ],
                ),
                "div": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 3], {}),
                    ],
                ),
            },
            instructions=[],
        )
        contract_body = contract.get_contract_body()
        sorted_entrypoints = [
            contract.entrypoints[name] for name in sorted(contract.entrypoints.keys())
        ]
        entrypoint_tree = EntrypointTree()
        expected_result = entrypoint_tree.list_to_tree(sorted_entrypoints)
        self.assertEqual(contract_body, expected_result)

    def test_entrypoints_to_tree(self):
        contract = Contract(
            storage=10,
            storage_type=t.Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 1], {}),
                    ],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 2], {}),
                    ],
                ),
                "div": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [
                        Instr("PUSH", [t.Int(), 3], {}),
                    ],
                ),
            },
            instructions=[],
        )
        entrypoints = [
            contract.entrypoints[name] for name in sorted(contract.entrypoints.keys())
        ]
        entrypoint_tree = EntrypointTree()
        contract_body = entrypoint_tree.list_to_tree(entrypoints)
        expected_result = Instr(
            "IF_LEFT",
            [
                [
                    Instr(
                        "IF_LEFT",
                        [
                            contract.entrypoints["add"].instructions,
                            contract.entrypoints["div"].instructions,
                        ],
                        {},
                    )
                ],
                contract.entrypoints["sub"].instructions,
            ],
            {},
        )
        self.assertEqual(contract_body, expected_result)

    def test_navigate_to_tree_leaf(self):
        entrypoint_list = ["add", "div", "mul", "sub", "modulo"]
        param_tree = ParameterTree()
        tree = param_tree.list_to_tree(entrypoint_list)
        entrypoint_param = 111

        entrypoint_index = entrypoint_list.index("add")
        contract_param = param_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Left(Left(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("div")
        contract_param = param_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Left(Right(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("mul")
        contract_param = param_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Right(Left(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("sub")
        contract_param = param_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Right(Right(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("modulo")
        contract_param = param_tree.navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Right(entrypoint_param))

    def test_left_side_tree_height(self):
        param_tree = ParameterTree()
        self.assertEqual(
            param_tree.left_side_tree_height(param_tree.list_to_tree([1, 2, 3, 4])), 2
        )
        self.assertEqual(
            param_tree.left_side_tree_height(param_tree.list_to_tree([1, 2, 3, 4, 5])),
            3,
        )
        self.assertEqual(
            param_tree.left_side_tree_height(
                param_tree.list_to_tree([1, 2, 3, 4, 5]).left
            ),
            2,
        )

    def test_list_to_tree(self):
        l = [1, 2]
        param_tree = ParameterTree()
        tree = param_tree.list_to_tree(l)
        self.assertEqual(Or(left=1, right=2), tree)

        l = [1, 2, 3]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(Or(left=Or(left=1, right=2), right=3), tree)

        l = [1, 2, 3, 4]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(Or(left=Or(left=1, right=2), right=Or(left=3, right=4)), tree)

        l = [1, 2, 3, 4, 5]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(
            Or(left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)), right=5),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=5, right=6),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=Or(left=5, right=6), right=7),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7, 8]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=Or(left=5, right=6), right=Or(left=7, right=8)),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tree = param_tree.list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(
                    left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                    right=Or(left=Or(left=5, right=6), right=Or(left=7, right=8)),
                ),
                right=9,
            ),
            tree,
        )

    def test_make_contract_param(self):
        contract = Contract(
            storage=10,
            storage_type=t.Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
                "div": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
            },
            instructions=[],
        )
        self.assertEqual(
            Left(Left(1)),
            contract.make_contract_param("add", 1),
        )
        self.assertEqual(
            Left(Right(1)),
            contract.make_contract_param("div", 1),
        )
        self.assertEqual(
            Right(1),
            contract.make_contract_param("sub", 1),
        )

    def test_contract_parameter_type(self):
        contract = Contract(
            storage=10,
            storage_type=t.Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
                "div": Entrypoint(
                    FunctionPrototype(t.Int(), t.Int()),
                    [],
                ),
            },
            instructions=[],
        )
        self.assertEqual(
            Or(Or(t.Int(), t.Int()), t.Int()),
            contract.get_parameter_type(),
        )


suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestContract)
unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    unittest.main()
