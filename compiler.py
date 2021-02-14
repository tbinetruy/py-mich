import ast
import pprint
import unittest
from typing import List, Optional

import instr_types as t
from helpers import Tree, ast_to_tree
from vm import VM
from vm_types import (Array, Contract, Entrypoint, Env, FunctionPrototype,
                      Instr, Pair)


def debug(cb):
    def f(*args, **kwargs):
        self = args[0]
        if self.isDebug:
            print(cb.__name__)

        return cb(*args, **kwargs)

    return f


def Comment(msg: str):
    return Instr("COMMENT", [msg], {})


class Record(Tree):
    def make_node(self, left, right):
        return Pair(car=left, cdr=right)

    def get_left(self, tree_node):
        return tree_node.car

    def get_right(self, tree_node):
        return tree_node.cdr

    def set_right(self, tree_node, value):
        tree_node.cdr = value

    def left_side_tree_height(self, tree, height=0):
        if type(tree) is not Pair:
            return height
        else:
            return self.left_side_tree_height(self.get_left(tree), height + 1)

    def get_type(self, element_types):
        return self.list_to_tree(element_types)

    def navigate_to_tree_leaf(self, tree, leaf_number, acc=None):
        if not acc:
            acc = []

        if type(tree) is not Pair:
            return acc

        left_max_leaf_number = 2 ** self.left_side_tree_height(self.get_left(tree))
        if leaf_number <= left_max_leaf_number:
            return (
                acc
                + [Instr("CAR", [], {})]
                + self.navigate_to_tree_leaf(self.get_left(tree), leaf_number)
            )
        else:
            return (
                acc
                + [Instr("CDR", [], {})]
                + self.navigate_to_tree_leaf(
                    self.get_right(tree), leaf_number - left_max_leaf_number
                )
            )

    def compile_node(self, node, acc=None):
        if not acc:
            acc = []
        if type(node) == Pair:
            return (
                self.compile_node(node.cdr)
                + self.compile_node(node.car)
                + [Instr("PAIR", [], {})]
            )
        else:
            return [
                Instr("PUSH", [t.Int(), node], {}),
            ]

    def build_record(self, ordered_elements):
        tree = self.list_to_tree(ordered_elements)
        return self.compile_node(tree)


class Compiler:
    def __init__(self, src: str, isDebug=False):
        self.ast = ast.parse(src)
        self.isDebug = isDebug
        self.type_parser = t.TypeParser()
        self.contract = Contract(
            storage_type=t.Int(),
            storage=0,
            entrypoints={},
            instructions=[],
        )

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))

    def compile_module(self, m: ast.Module, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        for key, value in ast.iter_fields(m):
            if key == "body":
                for childNode in value:
                    instructions += self._compile(childNode, e)

        return instructions

    @debug
    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = assign.targets[0]
        value = assign.value
        instructions = self._compile(var_name, e) + self._compile(value, e)
        e.vars[var_name.id] = e.sp
        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"{var_name.id} = {print_val}")] + instructions

    @debug
    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self._compile(expr.value, e)

    @debug
    def compile_num(self, num: ast.Constant, e: Env) -> List[Instr]:
        e.sp += 1  # Account for PUSH
        return [
            Instr("PUSH", [t.Int(), num.value], {}),
        ]

    @debug
    def compile_name(self, name: ast.Name, e: Env) -> List[Instr]:
        var_name = name
        if type(name.ctx) == ast.Load:
            var_addr = e.vars[var_name.id]
            jump_length = e.sp - var_addr
            comment = [
                Comment(
                    f"Loading {var_name.id} at {var_addr}, e.sp = {e.sp}, jump = {jump_length}"
                )
            ]
            instructions = [
                Instr("DIG", [jump_length], {}),
                Instr("DUP", [], {}),
                Instr("DUG", [jump_length + 1], {}),
            ]
            e.sp += 1  # Account for DUP
            return comment + instructions
        elif type(name.ctx) == ast.Store:
            # will get set to actual value in `compile_assign`
            e.vars[var_name.id] = 42
            return []
        else:
            raise NotImplementedError

    @debug
    def compile_binop(self, t: ast.BinOp, e: Env) -> List[Instr]:
        left = self._compile(t.left, e)
        right = self._compile(t.right, e)
        op = self._compile(t.op, e)
        return left + right + op

    @debug
    def compile_add(self, t: ast.Add, e: Env) -> List[Instr]:
        e.sp -= 1  # Account for ADD
        return [
            Instr("ADD", [], {}),
        ]

    @debug
    def create_list(self, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        return [
            Instr("NIL", [t.Int()], {}),
        ]

    @debug
    def append_before_list_el(self, el, e) -> List[Instr]:
        # no sp chage b/c they cancel out btwn the two instructions
        return self._compile(el, e) + [Instr("CONS", [], {})]

    @debug
    def compile_list(self, l: ast.List, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        instructions = self.create_list(e)
        for el in reversed(l.elts):
            instructions += self.append_before_list_el(el, e)
        return instructions

    def free_var(self, var_name, e: Env):
        var_location = e.vars[var_name]
        jump = e.sp - var_location
        e.sp -= 1  # account for freeing var

        comment = [Comment(f"Freeing var {var_name} at {var_location}")]
        return (
            comment
            + [
                Instr(
                    "DIP",
                    [
                        jump,
                        [
                            Instr("DROP", [], {}),
                        ],
                    ],
                    {},
                ),
            ],
            e,
        )

    def _get_function_prototype(self, f: ast.FunctionDef) -> FunctionPrototype:
        return FunctionPrototype(
            self.type_parser.parse(f.args.args[0].annotation),
            self.type_parser.parse(f.returns),
        )

    @debug
    def compile_defun(self, f: ast.FunctionDef, e: Env) -> List[Instr]:
        e.sp += 1  # account for body push

        e.vars[f.name] = e.sp
        e.args[f.name] = f.args.args[0].arg

        ast.dump(f.args.args[0])
        prototype = self._get_function_prototype(f)
        arg_type, return_type = prototype.arg_type, prototype.return_type
        # get init env keys
        init_var_names = set(e.vars.keys())

        func_env = e.copy()

        # store argument in env
        func_env.sp += 1
        func_env.vars[f.args.args[0].arg] = func_env.sp

        # iterate body instructions
        body_instructions = []
        for i in f.body:
            body_instructions += self._compile(i, func_env)

        # get new func_env keys
        new_var_names = set(func_env.vars.keys())

        # intersect init and new env keys
        intersection = list(new_var_names - init_var_names)

        # Free from the top of the stack. this ensures that the variable pointers
        # are not changed as variables are freed from the stack
        sorted_keys = sorted(intersection, key=lambda a: func_env.vars[a], reverse=True)

        # remove env vars from memory
        free_var_instructions = []
        tmp_env = func_env
        for var_name in sorted_keys:
            instr, tmp_env = self.free_var(var_name, tmp_env)
            free_var_instructions += instr
            try:
                del e.vars[var_name]
            except:
                pass

        comment = [Comment(f"Storing function {f.name} at {e.vars[f.name]}")]
        return comment + [
            Instr(
                "LAMBDA",
                [arg_type, return_type, body_instructions + free_var_instructions],
                {},
            ),
        ]

    @debug
    def compile_fcall(self, f: ast.Call, e: Env):
        # We work on an env copy to prevent from polluting the environment
        # with vars that we'd need to remove. We have to remember to pass
        # back the new stack pointer however
        tmp_env = e.copy()

        func_addr = tmp_env.vars[f.func.id]
        jump_length = tmp_env.sp - func_addr
        comment = [
            Comment(f"Moving to function {f.func.id} at {func_addr}, e.sp = {e.sp}")
        ]

        load_function = [
            Instr("DIG", [jump_length], {}),
            Instr("DUP", [], {}),
            Instr("DUG", [jump_length + 1], {}),
        ]

        tmp_env.sp += 1  # Account for DUP

        # fetch arg name for function
        load_arg = self._compile(f.args[0], tmp_env)

        tmp_env.sp += 1  # Account for pushing argument

        execute_function = [Instr("EXEC", [], {})]

        tmp_env.sp -= 2  # Account popping EXEC and LAMBDA

        instr = comment + load_function + load_arg + execute_function

        # We pass back the new stack pointer
        e.sp = tmp_env.sp

        return instr

    @debug
    def compile_return(self, r: ast.FunctionDef, e: Env):
        return self._compile(r.value, e)

    def get_init_env(self):
        return Env({}, -1, {})

    @debug
    def compile_entrypoint(self, f: ast.FunctionDef, e: Env) -> List[Instr]:
        prototype = self._get_function_prototype(f)

        # Save the storage and entrypoint argument on the stack
        if not self.contract.instructions:
            self.contract.instructions = [
                Instr("DUP", [], {}),  # [Pair(param, storage), Pair(param, storage)]
                Instr("CDR", [], {}),  # [Pair(param, storage), storage]
                Instr("DUG", [1], {}),  # [storage, Pair(param, storage)]
                Instr("CAR", [], {}),  # [storage, param]
            ]
        e.sp = 1  # update stack pointer
        e.vars["storage"] = 0
        e.vars[f.args.args[0].arg] = 1

        free_argument_instructions = [
            Comment(f"Freeing argument at sp={e.vars[f.args.args[0].arg]}"),
            Instr("DIP", [1, [Instr("DROP", [], {})]], {}),
        ]
        free_storage_instructions = [
            Comment("Freeing storage at e.sp=" + str(e.vars["storage"])),
            Instr("DIP", [1, [Instr("DROP", [], {})]], {}),
        ]
        epilogue = [
            Instr("NIL", [t.Operation()], {}),
            Instr("PAIR", [], {}),
        ]

        entrypoint_instructions = (
            self._compile(f, e)[-1].args[2]
            + free_argument_instructions
            + free_storage_instructions
            + epilogue
        )
        entrypoint = Entrypoint(prototype, entrypoint_instructions)
        self.contract.add_entrypoint(f.name, entrypoint)
        return []

    @debug
    def compile_contract(self, contract_ast: ast.ClassDef, e: Env) -> List[Instr]:
        instructions = []
        for entrypoint in contract_ast.body:
            instructions += self.compile_entrypoint(entrypoint, e)
        return instructions

    def compile(self):
        self._compile(self.ast)
        return self

    def _compile(self, node_ast, e: Optional[Env] = None) -> List[Instr]:
        e = self.get_init_env() if not e else e
        self.env = e  # saving as attribute for debug purposes
        instructions: List[Instr] = []

        if type(node_ast) == ast.Module:
            instructions += self.compile_module(node_ast, e)
            if self.isDebug:
                self.print_instructions(instructions)
        elif type(node_ast) == ast.Assign:
            instructions += self.compile_assign(node_ast, e)
        elif type(node_ast) == ast.Expr:
            instructions += self.compile_expr(node_ast, e)
        elif type(node_ast) == ast.Constant:
            instructions += self.compile_num(node_ast, e)
        elif type(node_ast) == ast.Name:
            instructions += self.compile_name(node_ast, e)
        elif type(node_ast) == ast.BinOp:
            instructions += self.compile_binop(node_ast, e)
        elif type(node_ast) == ast.Add:
            instructions += self.compile_add(node_ast, e)
        elif type(node_ast) == ast.List:
            instructions += self.compile_list(node_ast, e)
        elif type(node_ast) == ast.FunctionDef:
            instructions += self.compile_defun(node_ast, e)
        elif type(node_ast) == ast.Return:
            instructions += self.compile_return(node_ast, e)
        elif type(node_ast) == ast.Call:
            instructions += self.compile_fcall(node_ast, e)
        elif type(node_ast) == ast.ClassDef:
            if node_ast.name == "Contract":
                instructions += self.compile_contract(node_ast, e)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.isDebug:
            print(e)

        return instructions

    @staticmethod
    def print_instructions(instructions):
        print("\n".join([f"{i.name} {i.args} {i.kwargs}" for i in instructions]))


class TestContract(unittest.TestCase):
    def test_build_get_record_entry(self):
        tree = Record()
        array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        build_record_instructions = tree.build_record(array)
        for i in range(1, len(array) + 1):
            vm = VM()
            vm._run_instructions(build_record_instructions)
            get_record_entry = tree.navigate_to_tree_leaf(tree.get_type(array), i)
            vm._run_instructions(get_record_entry)
            self.assertEqual(vm.stack, [i])

    def test_build_record(self):
        tree = Record()

        instructions = tree.build_record([1, 2])
        expected_instructions = [
            Instr("PUSH", [t.Int(), 2], {}),
            Instr("PUSH", [t.Int(), 1], {}),
            Instr("PAIR", [], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

        instructions = tree.build_record([1, 2, 3, 4, 5])
        expected_instructions = [
            Instr("PUSH", [t.Int(), 5], {}),
            Instr("PUSH", [t.Int(), 4], {}),
            Instr("PUSH", [t.Int(), 3], {}),
            Instr("PAIR", [], {}),
            Instr("PUSH", [t.Int(), 2], {}),
            Instr("PUSH", [t.Int(), 1], {}),
            Instr("PAIR", [], {}),
            Instr("PAIR", [], {}),
            Instr("PAIR", [], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

    def test_record_tree(self):
        tree = Record()
        record = tree.list_to_tree([1, 2, 3, 4, 5])
        self.assertEqual(
            Pair(car=Pair(car=Pair(car=1, cdr=2), cdr=Pair(car=3, cdr=4)), cdr=5),
            record,
        )
        instructions = tree.navigate_to_tree_leaf(record, 1)
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = tree.navigate_to_tree_leaf(record, 2)
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = tree.navigate_to_tree_leaf(record, 3)
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = tree.navigate_to_tree_leaf(record, 4)
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = tree.navigate_to_tree_leaf(record, 5)
        self.assertEqual(
            [
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )

    def test_compile_record(self):
        source = """
    def test_multi_entrypoint_contract(self):
        vm = VM(isDebug=False)
        source = """
class Contract:
    def incrementByTwo(a: int) -> int:
        b = 1
        return a + b + 1

    def bar(b: int) -> int:
        return b
        """
        c = Compiler(source, isDebug=False)
        c._compile(c.ast)
        vm.run_contract(c.contract, "incrementByTwo", 10)
        self.assertEqual(c.contract.storage, 12)
        self.assertEqual(vm.stack, [])

        c._compile(c.ast)
        vm.run_contract(c.contract, "bar", 10)
        self.assertEqual(c.contract.storage, 10)
        self.assertEqual(vm.stack, [])


class TestCompilerUnit(unittest.TestCase):
    def test_create_list(self):
        vm = VM(isDebug=False)
        source = "[]"
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[0].els, [])

    def test_print_ast(self):
        pass


class TestCompilerList(unittest.TestCase):
    def test_func_def(self):
        vm = VM(isDebug=False)
        source = """
[1, 2, 3]
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [Array([1, 2, 3])])


class TestCompilerAssign(unittest.TestCase):
    def test_reassign(self):
        vm = VM(isDebug=False)
        source = """
a = 1
b = 2
a = b
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        # TODO: make vm.stack == [1, 2]
        #       and c.env.vars['a'] == 0 even
        #       after reassign
        self.assertEqual(vm.stack, [1, 2, 2])
        self.assertEqual(c.env.vars["a"], 2)
        self.assertEqual(c.env.vars["b"], 1)


class TestCompilerDefun(unittest.TestCase):
    def test_func_def(self):
        vm = VM(isDebug=False)
        source = """baz = 1
def foo(a: int) -> int:
    b = 2
    return a + b + 3
bar = foo(baz)
fff = foo(bar)
foo(foo(bar))
"""
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[-1], 16)
        self.assertEqual(instructions[3].args[0], t.Int())
        self.assertEqual(instructions[3].args[1], t.Int())
        self.assertEqual(len(vm.stack), 5)

    def todo_test_multiple_args_func(self):
        vm = VM(isDebug=False)
        source = """
def add(a, b):
    return a + b
foo(1, 2)
"""
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[-1], 16)
        self.assertEqual(len(vm.stack), 5)


class TestCompilerIntegration(unittest.TestCase):
    def test_store_vars_and_add(self):
        vm = VM(isDebug=False)
        source = """
a = 1
b = 2
c = a + b + b
a + b + c
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [1, 2, 5, 8])


for TestSuite in [
    TestContract,
    TestCompilerUnit,
    TestCompilerList,
    TestCompilerAssign,
    TestCompilerDefun,
    TestCompilerIntegration,
]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    unittest.main()



