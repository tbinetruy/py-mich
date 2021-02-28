import ast
import pprint
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional

import instr_types as t
from helpers import Tree, ast_to_tree
from vm import VM, VMFailwithException
from vm_types import (Array, Contract, Entrypoint, FunctionPrototype, Instr,
                      Pair)


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
    def __init__(self, attribute_names, attribute_types):
        self.attribute_names = attribute_names
        self.attribute_types = attribute_types

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

    def get_type(self):
        return self.list_to_tree(self.attribute_types)

    def _attribute_name_to_leaf_number(self, attribute_name):
        for i, target_name in enumerate(self.attribute_names):
            if attribute_name == target_name:
                return i + 1

    def navigate_to_tree_leaf(self, attribute_name, acc=None):
        leaf_number = self._attribute_name_to_leaf_number(attribute_name)
        tree = self.list_to_tree([i for i, _ in enumerate(self.attribute_names)])
        return self._navigate_to_tree_leaf(tree, leaf_number)

    def _navigate_to_tree_leaf(self, tree, leaf_number, acc=None):
        if not acc:
            acc = []

        if type(tree) is not Pair:
            return acc

        left_max_leaf_number = 2 ** self.left_side_tree_height(self.get_left(tree))
        if leaf_number <= left_max_leaf_number:
            return (
                acc
                + [Instr("CAR", [], {})]
                + self._navigate_to_tree_leaf(self.get_left(tree), leaf_number)
            )
        else:
            return (
                acc
                + [Instr("CDR", [], {})]
                + self._navigate_to_tree_leaf(
                    self.get_right(tree), leaf_number - left_max_leaf_number
                )
            )

    def _compile_node(self, node, acc=None):
        if not acc:
            acc = []
        if type(node) == Pair:
            return (
                self._compile_node(node.cdr)
                + self._compile_node(node.car)
                + [Instr("PAIR", [], {})]
            )
        else:
            return [
                Instr("PUSH", [t.Int(), node], {}),
            ]

    def build_record(self, attribute_values):
        tree = self.list_to_tree(attribute_values)
        return self._compile_node(tree)

    def _compile_node_new(self, node, compile_function, env):
        if type(node) == Pair:
            el1 = self._compile_node_new(node.cdr, compile_function, env)
            el2 = self._compile_node_new(node.car, compile_function, env)
            env.sp -= 1  # account for pair
            return el1 + el2 + [Instr("PAIR", [], {})]
        else:
            return compile_function(node, env)

    def compile_record(self, attribute_values, compile_function, env):
        tree = self.list_to_tree(attribute_values)
        return self._compile_node_new(tree, compile_function, env)


@dataclass
class Env:
    vars: Dict[str, int]
    sp: int
    args: Dict[str, List[str]]
    records: Dict[str, Record]
    types: Dict[str, str]

    def copy(self):
        return Env(
            self.vars.copy(),
            self.sp,
            self.args.copy(),
            self.records.copy(),
            self.types.copy(),
        )


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

    def _compile_reassign(self, reassign_ast: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = reassign_ast.targets[0]
        value = reassign_ast.value
        var_addr = e.vars[var_name.id]
        instructions = self._compile(value, e)
        free_vars_instructions, _ = self.free_var(var_name.id, e)
        instructions = instructions + free_vars_instructions + [
            Instr("DUG", [e.sp - var_addr], {}),
        ]
        e.vars[var_name.id] = var_addr

        try:
            if reassign_ast.value.func.id in e.records:
                e.types[var_name.id] = reassign_ast.value.func.id
        except:
            pass

        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"Reassigning {var_name.id} = {print_val}")] + instructions

    @debug
    def compile_assign_storage_attribute(self, assign_ast: ast.Assign, e: Env) -> List[Instr]:
        attribute_to_assign = assign_ast.targets[0].attr
        attribute_names = e.records['Storage'].attribute_names

        args = []
        for attribute_name in attribute_names:
            if attribute_name == attribute_to_assign:
                args.append(assign_ast.value)
            else:
                args.append(
                    ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='storage',
                            ctx=ast.Load(),
                        ),
                        attr=attribute_name,
                        ctx=ast.Load(),
                    ),
                )

        new_ast = ast.Assign(
            targets=[
                ast.Name(id='storage', ctx=ast.Load()),
            ],
            value=ast.Call(
                func=ast.Name(id='Storage', ctx=ast.Load()),
                args=args,
                keywords=[]
            ),
            type_comment=None,
        )
        return self.compile_assign(new_ast, e)

    @debug
    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        try:
            cond1 = assign.targets[0].value.value.id == "self"
            cond2 = assign.targets[0].value.attr == "storage"
            if cond1 and cond2:
                return self.compile_assign_storage_attribute(assign, e)
        except:
            pass

        try:
            if assign.targets[0].id in e.vars.keys():
                return self._compile_reassign(assign, e)
        except:
            pass

        instructions: List[Instr] = []
        var_name = assign.targets[0]
        value = assign.value
        instructions = self._compile(var_name, e) + self._compile(value, e)
        e.vars[var_name.id] = e.sp

        try:
            if assign.value.func.id in e.records:
                e.types[var_name.id] = assign.value.func.id
        except:
            pass

        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"{var_name.id} = {print_val}")] + instructions

    @debug
    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self._compile(expr.value, e)

    def _is_string_address(self, string: str) -> bool:
        is_tz_address = len(string) == 36 and string[:2] == "tz"
        is_kt_address = len(string) == 36 and string[:2] == "KT"
        return is_tz_address or is_kt_address

    @debug
    def compile_constant(self, constant: ast.Constant, e: Env) -> List[Instr]:
        e.sp += 1  # Account for PUSH

        constant_type: t.Type = t.Int()
        if type(constant.value) == str:
            if self._is_string_address(constant.value):
                constant_type = t.Address()
            else:
                constant_type = t.String()

        return [
            Instr("PUSH", [constant_type, constant.value], {}),
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

    def _get_function_prototype(self, f: ast.FunctionDef, e: Env) -> FunctionPrototype:
        return FunctionPrototype(
            self.type_parser.parse(f.args.args[0].annotation, e),
            self.type_parser.parse(f.returns, e),
        )

    @debug
    def compile_defun(self, f: ast.FunctionDef, e: Env) -> List[Instr]:
        e.sp += 1  # account for body push

        e.vars[f.name] = e.sp
        e.args[f.name] = f.args.args[0].arg

        # type argument
        if f.args.args[0].annotation.id in e.records:
            e.types[f.args.args[0].arg] = f.args.args[0].annotation.id

        prototype = self._get_function_prototype(f, e)
        arg_type, return_type = prototype.arg_type, prototype.return_type
        # get init env keys
        init_var_names = set(e.vars.keys())

        func_env = e.copy()

        # store argument in env
        func_env.sp += 1
        func_env.vars[f.args.args[0].arg] = func_env.sp

        body_instructions = self._compile_block(f.body, func_env)
        # account for pushed result
        func_env.sp += 1
        body_instructions += self.free_var(f.args.args[0].arg, func_env)[0]

        comment = [Comment(f"Storing function {f.name} at {e.vars[f.name]}")]
        return comment + [
            Instr(
                "LAMBDA",
                [arg_type, return_type, body_instructions],
                {},
            ),
        ]

    @debug
    def compile_ccall(self, c: ast.Call, e: Env):
        """Call to class constructor"""
        instructions = e.records[c.func.id].compile_record(c.args, self._compile, e)
        return instructions

    @debug
    def compile_fcall(self, f: ast.Call, e: Env):
        if f.func.id in e.records.keys():
            return self.compile_ccall(f, e)

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
        return Env({}, -1, {}, {}, {})

    @debug
    def compile_entrypoint(self, f: ast.FunctionDef, e: Env) -> List[Instr]:

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

        # type argument
        if f.args.args[0].annotation.id in e.records:
            e.types[f.args.args[0].arg] = f.args.args[0].annotation.id

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

        entrypoint_instructions = self._compile_block(f.body, e) + free_argument_instructions + free_storage_instructions + epilogue
        prototype = self._get_function_prototype(f, e)
        entrypoint = Entrypoint(prototype, entrypoint_instructions)
        self.contract.add_entrypoint(f.name, entrypoint)
        return []

    def _compile_block(self, block_ast: List[ast.AST], e: Env) -> List[Instr]:
        """frees newly declared variables at the end of the block, hence °e°
        should be the same befor and after the block"""
        # get init env keys
        init_var_names = set(e.vars.keys())

        block_env = e.copy()

        # iterate body instructions
        block_instructions = []
        for i in block_ast:
            block_instructions += self._compile(i, block_env)

        # get new func_env keys
        new_var_names = set(block_env.vars.keys())

        # intersect init and new env keys
        intersection = list(new_var_names - init_var_names)

        # Free from the top of the stack. this ensures that the variable pointers
        # are not changed as variables are freed from the stack
        sorted_keys = sorted(intersection, key=lambda a: block_env.vars[a], reverse=True)

        # remove env vars from memory
        free_var_instructions = []
        for var_name in sorted_keys:
            instr, block_env = self.free_var(var_name, block_env)
            free_var_instructions += instr

        return block_instructions + free_var_instructions

    @debug
    def compile_storage(self, storage_ast, e: Env):
        if type(storage_ast) == ast.Call:
            # assume constructed from record
            storage_type = storage_ast.func.id
            e.types["__STORAGE__"] = storage_type
            self.contract.storage_type = e.records[storage_type].get_type()

            # TODO fix this mess
            vm = VM()
            init_storage_instr = e.records[storage_type].compile_record(storage_ast.args, self._compile, self.get_init_env())
            vm._run_instructions(init_storage_instr)
            self.contract.storage = vm.stack[-1]
        else:
            return NotImplementedError

    @debug
    def compile_contract(self, contract_ast: ast.ClassDef, e: Env) -> List[Instr]:
        instructions = []
        for entrypoint in contract_ast.body:
            if entrypoint.name == "deploy":
                if type(entrypoint.body[0]) == ast.Return:
                    self.compile_storage(entrypoint.body[0].value, e)
                else:
                    return NotImplementedError
            else:
                instructions += self.compile_entrypoint(entrypoint, e)
        return instructions

    @debug
    def compile_record(self, record_ast: ast.ClassDef, e: Env) -> List[Instr]:
        attribute_names = [attr.target.id for attr in record_ast.body]
        attribute_types = []
        for attr in record_ast.body:
            attribute_types.append(self.type_parser.parse(attr.annotation, e))

        e.records[record_ast.name] = Record(attribute_names, attribute_types)
        return []

    def handle_get_storage(self, storage_get_ast: ast.Attribute, e: Env) -> List[Instr]:
        if storage_get_ast.attr != "storage":
            # storage is record
            key = storage_get_ast.attr
            load_storage_instr = self.compile_name(ast.Name(id='storage', ctx=ast.Load()), e)
            storage_key_name = storage_get_ast.attr
            get_storage_key_instr = e.records[e.types['__STORAGE__']].navigate_to_tree_leaf(storage_key_name)
            return load_storage_instr + get_storage_key_instr
        else:
            return self.compile_name(ast.Name(id='storage', ctx=ast.Load()), e)

    def check_get_storage(self, storage_get_ast: ast.Attribute) -> bool:
        try:
            return (
                storage_get_ast.value.value.id == "self"
                and storage_get_ast.value.attr == "storage"
            )
        except:
            return (
                storage_get_ast.value.id == "self"
                and storage_get_ast.attr == "storage"
            )

    @debug
    def compile_attribute(self, attribute_ast: ast.Attribute, e: Env) -> List[Instr]:
        if self.check_get_storage(attribute_ast):
            return self.handle_get_storage(attribute_ast, e)

        load_object_instructions = self.compile_name(attribute_ast.value, e)
        record = e.records[e.types[attribute_ast.value.id]]
        load_attribute_instructions = record.navigate_to_tree_leaf(attribute_ast.attr)
        return load_object_instructions + load_attribute_instructions

    @debug
    def compile_compare(self, compare_ast: ast.Compare, e: Env) -> List[Instr]:
        compare_instructions = (
            self._compile(compare_ast.comparators[0], e)
            + self._compile(compare_ast.left, e)
            + [Instr("COMPARE", [], {})]
        )
        # Account for COMPARE
        e.sp -= 1

        operator_type = type(compare_ast.ops[0])
        if operator_type == ast.Eq:
            operator_instructions = [Instr("EQ", [], {})]
        elif operator_type == ast.NotEq:
            operator_instructions = [Instr("NEQ", [], {})]
        elif operator_type == ast.Lt:
            operator_instructions = [Instr("LT", [], {})]
        elif operator_type == ast.Gt:
            operator_instructions = [Instr("GT", [], {})]
        elif operator_type == ast.LtE:
            operator_instructions = [Instr("LE", [], {})]
        elif operator_type == ast.GtE:
            operator_instructions = [Instr("GE", [], {})]
        else:
            return NotImplementedError

        return compare_instructions + operator_instructions

    def compile_if(self, if_ast: ast.If, e: Env) -> List[Instr]:
        test_instructions = self._compile(if_ast.test, e)

        # Account for "IF" poping the boolean sitting at the top of the stack
        e.sp -= 1

        if_true_instructions = self._compile_block(if_ast.body, e.copy())
        if_false_instructions = self._compile_block(if_ast.orelse, e.copy())
        if_instructions = [Instr("IF", [if_true_instructions, if_false_instructions], {})]
        return test_instructions + if_instructions

    def compile_raise(self, raise_ast: ast.Raise, e: Env) -> List[Instr]:
        return self._compile(raise_ast.exc, e) + [Instr("FAILWITH", [], {})]

    def compile(self):
        return self._compile(self.ast)
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
        elif type(node_ast) == ast.Attribute:
            instructions += self.compile_attribute(node_ast, e)
        elif type(node_ast) == ast.Expr:
            instructions += self.compile_expr(node_ast, e)
        elif type(node_ast) == ast.If:
            instructions += self.compile_if(node_ast, e)
        elif type(node_ast) == ast.Constant:
            instructions += self.compile_constant(node_ast, e)
        elif type(node_ast) == ast.Compare:
            instructions += self.compile_compare(node_ast, e)
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
        elif type(node_ast) == ast.Raise:
            instructions += self.compile_raise(node_ast, e)
        elif type(node_ast) == ast.Call:
            instructions += self.compile_fcall(node_ast, e)
        elif type(node_ast) == ast.ClassDef:
            if node_ast.name == "Contract":
                instructions += self.compile_contract(node_ast, e)
            elif "dataclass" in [decorator.id for decorator in node_ast.decorator_list]:
                instructions += self.compile_record(node_ast, e)
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


class TestRecord(unittest.TestCase):
    def test_compile_get_record_attribute(self):
        def test(attribute_name, stack_top_value):
            source = f"""
@dataclass
class Storage:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int

a = 2
b = 4
c = 6
my_storage = Storage(1, a, 3, b, 5, c)
my_storage.{attribute_name}
"""
            c = Compiler(source)
            instructions = c.compile()
            vm = VM()
            vm._run_instructions(instructions)
            self.assertEqual(vm.stack[-1], stack_top_value)

        test("a", 1)
        test("b", 2)
        test("c", 3)
        test("d", 4)
        test("e", 5)
        test("f", 6)

    def test_compile_create_record(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int

a = 2
b = 4
c = 6
my_storage = Storage(1, a, 3, b, 5, c)
d = 7
my_storage # get storage
"""
        c = Compiler(source)
        instructions = c.compile()
        vm = VM()
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[-1], Pair(Pair(Pair(1, 2), Pair(3, 4)), Pair(5, 6)))

    def test_get_record_entry(self):
        attribute_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        attribute_types = [t.Int() for _ in attribute_names]
        record = Record(attribute_names, attribute_types)
        attribute_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        build_record_instructions = record.build_record(attribute_values)
        for i in range(0, len(attribute_values)):
            vm = VM()
            vm._run_instructions(build_record_instructions)
            get_record_entry = record.navigate_to_tree_leaf(attribute_names[i])
            vm._run_instructions(get_record_entry)
            self.assertEqual(vm.stack, [attribute_values[i]])

    def test_build_record(self):
        record = Record(["a", "b"], [t.Int(), t.Int()])
        instructions = record.build_record([1, 2])
        expected_instructions = [
            Instr("PUSH", [t.Int(), 2], {}),
            Instr("PUSH", [t.Int(), 1], {}),
            Instr("PAIR", [], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

        record = Record(
            ["a", "b", "c", "d", "e"], [t.Int(), t.Int(), t.Int(), t.Int(), t.Int()]
        )
        instructions = record.build_record([1, 2, 3, 4, 5])
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
        record = Record(
            ["a", "b", "c", "d", "e"], [t.Int(), t.Int(), t.Int(), t.Int(), t.Int()]
        )

        self.assertEqual(
            Pair(car=Pair(car=Pair(car=1, cdr=2), cdr=Pair(car=3, cdr=4)), cdr=5),
            record.list_to_tree([1, 2, 3, 4, 5]),
        )
        instructions = record.navigate_to_tree_leaf("a")
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = record.navigate_to_tree_leaf("b")
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = record.navigate_to_tree_leaf("c")
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
                Instr(name="CAR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = record.navigate_to_tree_leaf("d")
        self.assertEqual(
            [
                Instr(name="CAR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )

        instructions = record.navigate_to_tree_leaf("e")
        self.assertEqual(
            [
                Instr(name="CDR", args=[], kwargs={}),
            ],
            instructions,
        )


class TestContract(unittest.TestCase):
    def test_attribute_reassign(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

class Contract:
    def deploy():
        return Storage(1, 2, 3)

    def set_a(new_a: int) -> int:
        self.storage.a = new_a
        return self.storage

    def set_b(new_b: int) -> int:
        self.storage.b = new_b
        return self.storage

    def set_c(new_c: int) -> int:
        self.storage.c = new_c
        return self.storage
        """
        vm = VM(isDebug=False)
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm.run_contract(c.contract, "set_a", 10)
        self.assertEqual(c.contract.storage, Pair(Pair(10, 2), 3))

        vm.run_contract(c.contract, "set_b", 20)
        self.assertEqual(c.contract.storage, Pair(Pair(10, 20), 3))

        vm.run_contract(c.contract, "set_c", 30)
        self.assertEqual(c.contract.storage, Pair(Pair(10, 20), 30))


    def test_contract_final(self):

        vm = VM(isDebug=False)
        owner =  "tzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        source = f"""
@dataclass
class Storage:
    owner: address
    name: str
    counter: int

class Contract:
    def deploy():
        return Storage("{owner}", "foo", 0)

    def add(a: int) -> int:
        if a < 10:
            raise 'input smaller than 10'
        else:
            a = a + a
            self.storage.counter = self.storage.counter + a
            return self.storage

    def update_owner(new_owner: address) -> int:
        self.storage.owner = new_owner
        return self.storage

    def update_name(new_name: str) -> int:
        self.storage.name = new_name
        return self.storage
        """
        c = Compiler(source, isDebug=False)
        c._compile(c.ast)
        try:
            vm.run_contract(c.contract, "add", 1)
            assert 0
        except VMFailwithException:
            assert 1

        vm.run_contract(c.contract, "add", 10)
        self.assertEqual(c.contract.storage, Pair(Pair(owner, "foo"), 20))
        self.assertEqual(vm.stack, [])

        new_owner = "tzbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        vm.run_contract(c.contract, "update_owner", new_owner)
        self.assertEqual(c.contract.storage, Pair(Pair(new_owner, "foo"), 20))
        self.assertEqual(vm.stack, [])

        vm.run_contract(c.contract, "update_name", "bar")
        self.assertEqual(c.contract.storage, Pair(Pair(new_owner, "bar"), 20))
        self.assertEqual(vm.stack, [])

    def test_contract_multitype_storage_with_condition(self):
        vm = VM(isDebug=False)
        source = """
@dataclass
class Storage:
    owner: str
    counter: int

class Contract:
    def deploy():
        return Storage("foo", 0)

    def add(a: int) -> int:
        if a < 10:
            raise 'input smaller than 10'
        else:
            a = a + a
            return Storage(self.storage.owner, self.storage.counter + a)

    def update_owner(new_owner: str) -> int:
        return Storage(new_owner, self.storage.counter)
        """
        c = Compiler(source, isDebug=False)
        c._compile(c.ast)
        try:
            vm.run_contract(c.contract, "add", 1)
            assert 0
        except VMFailwithException:
            assert 1

        vm.run_contract(c.contract, "add", 10)
        self.assertEqual(c.contract.storage, Pair("foo", 20))
        self.assertEqual(vm.stack, [])

    def test_contract_multitype_storage(self):
        vm = VM(isDebug=False)
        source = """
@dataclass
class Storage:
    owner: str
    counter: int

class Contract:
    def deploy():
        return Storage("foo", 0)

    def add(a: int) -> int:
        return Storage(self.storage.owner, self.storage.counter + a)

    def update_owner(new_owner: str) -> int:
        return Storage(new_owner, self.storage.counter)
        """
        c = Compiler(source, isDebug=False)
        c._compile(c.ast)
        vm.run_contract(c.contract, "add", 10)
        self.assertEqual(c.contract.storage, Pair("foo", 10))
        self.assertEqual(vm.stack, [])

        vm.run_contract(c.contract, "update_owner", "bar")
        self.assertEqual(c.contract.storage, Pair("bar", 10))
        self.assertEqual(vm.stack, [])

    def test_contract_storage(self):
        vm = VM(isDebug=False)
        source = """
@dataclass
class Storage:
    owner_id: int
    counter: int

class Contract:
    def deploy():
        return Storage(1, 0)

    def add(a: int) -> int:
        b = 10
        new_storage = Storage(self.storage.owner_id, self.storage.counter + a + b)
        return new_storage

    def update_owner_id(new_id: int) -> int:
        return Storage(new_id, self.storage.counter)
        """
        c = Compiler(source, isDebug=False)
        c._compile(c.ast)
        vm.run_contract(c.contract, "add", 10)
        self.assertEqual(c.contract.storage, Pair(1, 20))
        self.assertEqual(vm.stack, [])

        vm.run_contract(c.contract, "update_owner_id", 111)
        self.assertEqual(c.contract.storage, Pair(111, 20))
        self.assertEqual(vm.stack, [])

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
a = a + 2
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        # TODO: make vm.stack == [1, 2]
        #       and c.env.vars['a'] == 0 even
        #       after reassign
        self.assertEqual(vm.stack, [3, 2])
        self.assertEqual(c.env.vars["a"], 0)
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
    def test_push_address(self):
        user_address = "tz1S792fHX5rvs6GYP49S1U58isZkp2bNmn6"
        contract_address = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        regular_string = "foobar"
        source = f"""
user_address = "{user_address}"
contract_address = "{contract_address}"
regular_string = "{regular_string}"
        """
        c = Compiler(source, isDebug=False)
        instructions = [instr for instr in c._compile(c.ast) if instr.name != "COMMENT"]
        expected_instructions = [
            Instr("PUSH", [t.Address(), user_address], {}),
            Instr("PUSH", [t.Address(), contract_address], {}),
            Instr("PUSH", [t.String(), regular_string], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

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

    def test_push_string(self):
        vm = VM(isDebug=False)
        source = "'foobar'"
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, ["foobar"])

    def test_compare(self):
        vm = VM(isDebug=False)
        source = "1 < 2"
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        expected_instructions = [
            Instr("PUSH", [t.Int(), 2], {}),
            Instr("PUSH", [t.Int(), 1], {}),
            Instr("COMPARE", [], {}),
            Instr("LT", [], {}),
        ]
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [True])
        self.assertEqual(instructions, expected_instructions)

    def test_if(self):
        vm = VM(isDebug=False)
        source = """
if 1 < 2:
    "foo"
else:
    "bar"
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, ["foo"])

    def test_if_reassign(self):
        vm = VM(isDebug=False)
        source = """
foo = "foo"
if 1 < 2:
    foo = "bar"
else:
    foo = "baz"
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, ["bar"])
        self.assertEqual(c.env.vars["foo"], 0)

    def test_if_failwith(self):
        vm = VM(isDebug=False)
        source = """
foo = "foo"
if 1 < 2:
    raise "my error"
else:
    foo = "baz"
        """
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        try:
            vm._run_instructions(instructions)
            assert 0
        except VMFailwithException:
            assert 1

    def test_raise(self):
        vm = VM(isDebug=False)
        source = "raise 'foobar'"
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        expected_instructions = [
            Instr("PUSH", [t.String(), 'foobar'], {}),
            Instr("FAILWITH", [], {})
        ]
        self.assertEqual(instructions, expected_instructions)
        try:
            vm._run_instructions(instructions)
            assert 0
        except VMFailwithException:
            assert 1

    def test_record_as_function_argument(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

def add(storage: Storage) -> int:
    return storage.a + storage.b + storage.c

add(Storage(1, 2, 3))
        """
        vm = VM(isDebug=False)
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[-1], 6)


for TestSuite in [
    TestRecord,
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
