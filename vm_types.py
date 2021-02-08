import unittest
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple

import instr_types as t


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


Addr, VarName = NewType("Address", int), NewType("VarName", str)


@dataclass
class Env:
    vars: NewType("Environment", Dict[VarName, Addr])
    sp: int
    args: NewType("Args", Dict[VarName, List[VarName]])

    def copy(self):
        return Env(self.vars.copy(), self.sp, self.args.copy())


@dataclass
class Instr:
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]


@dataclass
class Entrypoint:
    prototype: FunctionPrototype
    instructions: List[Instr]


def list_to_tree(l):
    ors = []
    for el in l:
        if not len(ors):
            ors.append(Or(left=el, right=None))
        elif not ors[-1].right:
            ors[-1].right = el
        else:
            ors.append(Or(left=el, right=None))

    if not ors[-1].right:
        ors[-1] = ors[-1].left

    def construct_tree(ors):
        new_ors = []
        i = 0
        while i < len(ors):
            try:
                new_ors.append(Or(left=ors[i], right=ors[i + 1]))
            except IndexError:
                new_ors.append(ors[i])
            i += 2
        if len(new_ors) == 1:
            return new_ors[0]
        else:
            return construct_tree(new_ors)

    return construct_tree(ors)


def entrypoints_to_tree(entrypoints: List[Entrypoint]):
    tree_leaves: List[Instr] = []
    for entrypoint in entrypoints:
        if not len(tree_leaves):
            tree_leaves.append(Instr("IF_LEFT", [entrypoint.instructions, []], {}))
        elif not tree_leaves[-1].args[1]:
            tree_leaves[-1].args[1] = entrypoint.instructions
        else:
            tree_leaves.append(Instr("IF_LEFT", [entrypoint.instructions, []], {}))

    if not tree_leaves[-1].args[1]:
        tree_leaves[-1] = tree_leaves[-1].args[0]

    def construct_tree(tree_leaves):
        tree = []
        i = 0
        while i < len(tree_leaves):
            try:
                cond_true = (
                    tree_leaves[i] if type(tree_leaves[i]) == list else [tree_leaves[i]]
                )
                cond_false = (
                    tree_leaves[i + 1]
                    if type(tree_leaves[i + 1]) == list
                    else [tree_leaves[i + 1]]
                )
                tree.append(Instr("IF_LEFT", [cond_true, cond_false], {}))
            except IndexError:
                tree.append(tree_leaves[i])
            i += 2
        if len(tree) == 1:
            return tree[0]
        else:
            return construct_tree(tree)

    return construct_tree(tree_leaves)


def left_side_tree_height(tree, height=0):
    if type(tree) is not Or:
        return height
    else:
        return left_side_tree_height(tree.left, height + 1)


def navigate_to_tree_leaf(tree, leaf_number, param):
    if type(tree) is not Or:
        return param

    left_max_leaf_number = 2 ** left_side_tree_height(tree.left)
    if leaf_number <= left_max_leaf_number:
        return Left(navigate_to_tree_leaf(tree.left, leaf_number, param))
    else:
        return Right(
            navigate_to_tree_leaf(tree.right, leaf_number - left_max_leaf_number, param)
        )


@dataclass
class Contract:
    storage: Any
    storage_type: t.Type
    entrypoints: NewType("Entrypoints", Dict[VarName, Entrypoint])
    instructions: List[Instr]

    def copy(self):
        return Env(self.vars.copy(), self.sp, self.args.copy())

    def add_entrypoint(self, name: str, entrypoint: Entrypoint):
        self.entrypoints[name] = entrypoint

    def make_contract_param(self, entrypoint_name, entrypoint_param):
        entrypoint_names = sorted(self.entrypoints.keys())
        if len(entrypoint_names) == 1:
            return entrypoint_param

        tree = list_to_tree(entrypoint_names)
        entrypoint_index = entrypoint_names.index(entrypoint_name)
        return navigate_to_tree_leaf(tree, entrypoint_index + 1, entrypoint_param)

    def get_parameter_type(self):
        entrypoint_names = self.entrypoints.keys()
        if len(entrypoint_names) == 1:
            return self.entrypoints[entrypoint_names[0]].arg_type
        else:
            return list_to_tree(
                [
                    self.entrypoints[name].prototype.arg_type
                    for name in sorted(entrypoint_names)
                ]
            )

    def get_contract_body(self):
        entrypoints = [
            self.entrypoints[name] for name in sorted(self.entrypoints.keys())
        ]
        return entrypoints_to_tree(entrypoints)


contract = Contract(
    storage=10,
    storage_type=t.Int(),
    entrypoints={
        "add": FunctionPrototype(t.Int(), t.Int()),
        "div": FunctionPrototype(t.Int(), t.Int()),
        "mul": FunctionPrototype(t.Int(), t.Int()),
        "sub": FunctionPrototype(t.Int(), t.Int()),
    },
    instructions=[],
)
# print(contract.get_parameter_type())
# print(Or(Or(t.Int(), t.Int()), Or(t.Int(), t.Int())))


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
        expected_result = entrypoints_to_tree(sorted_entrypoints)
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
        contract_body = entrypoints_to_tree(entrypoints)
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
        tree = list_to_tree(entrypoint_list)
        entrypoint_param = 111

        entrypoint_index = entrypoint_list.index("add")
        contract_param = navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Left(Left(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("div")
        contract_param = navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Left(Right(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("mul")
        contract_param = navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Right(Left(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("sub")
        contract_param = navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Left(Right(Right(entrypoint_param))))

        entrypoint_index = entrypoint_list.index("modulo")
        contract_param = navigate_to_tree_leaf(
            tree, entrypoint_index + 1, entrypoint_param
        )
        self.assertEqual(contract_param, Right(entrypoint_param))

    def test_left_side_tree_height(self):
        self.assertEqual(left_side_tree_height(list_to_tree([1, 2, 3, 4])), 2)
        self.assertEqual(left_side_tree_height(list_to_tree([1, 2, 3, 4, 5])), 3)
        self.assertEqual(left_side_tree_height(list_to_tree([1, 2, 3, 4, 5]).left), 2)

    def test_list_to_tree(self):
        l = [1, 2]
        tree = list_to_tree(l)
        self.assertEqual(Or(left=1, right=2), tree)

        l = [1, 2, 3]
        tree = list_to_tree(l)
        self.assertEqual(Or(left=Or(left=1, right=2), right=3), tree)

        l = [1, 2, 3, 4]
        tree = list_to_tree(l)
        self.assertEqual(Or(left=Or(left=1, right=2), right=Or(left=3, right=4)), tree)

        l = [1, 2, 3, 4, 5]
        tree = list_to_tree(l)
        self.assertEqual(
            Or(left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)), right=5),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6]
        tree = list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=5, right=6),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7]
        tree = list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=Or(left=5, right=6), right=7),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7, 8]
        tree = list_to_tree(l)
        self.assertEqual(
            Or(
                left=Or(left=Or(left=1, right=2), right=Or(left=3, right=4)),
                right=Or(left=Or(left=5, right=6), right=Or(left=7, right=8)),
            ),
            tree,
        )

        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tree = list_to_tree(l)
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
