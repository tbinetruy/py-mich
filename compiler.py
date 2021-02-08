import ast
import pprint
import unittest
from typing import List, Optional

import instr_types as t
from helpers import ast_to_tree
from vm import VM
from vm_types import Array, Contract, Env, FunctionPrototype, Instr


def debug(cb):
    def f(*args, **kwargs):
        self = args[0]
        if self.isDebug:
            print(cb.__name__)

        return cb(*args, **kwargs)

    return f


def Comment(msg: str):
    return Instr("COMMENT", [msg], {})


class Compiler:
    def __init__(self, src: str, isDebug=True):
        self.ast = ast.parse(src)
        self.isDebug = isDebug
        self.type_parser = t.TypeParser()

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))

    def compile_module(self, m: ast.Module, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        for key, value in ast.iter_fields(m):
            if key == "body":
                for childNode in value:
                    instructions += self.compile(childNode, e)

        return instructions

    @debug
    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = assign.targets[0]
        value = assign.value
        instructions = self.compile(var_name, e) + self.compile(value, e)
        e.vars[var_name.id] = e.sp
        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"{var_name.id} = {print_val}")] + instructions

    @debug
    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self.compile(expr.value, e)

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
                Instr("DIP", [jump_length], {}),
                Instr("DUP", [], {}),
                Instr("DIG", [jump_length], {}),
                # Instr('IIP', [], {}),
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
        left = self.compile(t.left, e)
        right = self.compile(t.right, e)
        op = self.compile(t.op, e)
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
            Instr("LIST", [], {}),
        ]

    @debug
    def append_before_list_el(self, el, e) -> List[Instr]:
        # no sp chage b/c they cancel out btwn the two instructions
        return self.compile(el, e) + [Instr("CONS", [], {})]

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
        e.sp -= jump  # DIP

        # If the stack pointer is at 0, then don't increment it
        # see VM.pop that has particular behavior when it results
        # in empty stack
        if e.sp:
            epilogue = [Instr("IIP", [jump], {})]
        else:
            epilogue = []

        comment = [Comment(f"Freeing var {var_name} at {var_location}")]
        return (
            comment
            + [
                Instr("DIP", [jump], {}),
                Instr("DROP", [], {}),
            ]
            + epilogue,
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
            body_instructions += self.compile(i, func_env)

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
        prologue_instr = comment + [
            Instr("DIP", [jump_length], {}),
        ]
        tmp_env.sp -= jump_length  # Account for DIP

        # fetch arg name for function
        arg_name = tmp_env.args[f.func.id]

        # compile arg
        arg = self.compile(f.args[0], tmp_env)

        # Store arg stack location
        tmp_env.vars[arg_name] = tmp_env.sp

        tmp_env.sp += jump_length  # Account for DIG

        # We pass back the new stack pointer
        e.sp = tmp_env.sp

        comment = [Comment(f"Executing function {f.func.id} at {func_addr}")]
        return (
            prologue_instr
            + arg
            + comment
            + [
                Instr("EXEC", [], {}),
                Instr("DIG", [jump_length], {}),
            ]
        )

    @debug
    def compile_return(self, r: ast.FunctionDef, e: Env):
        return self.compile(r.value, e)

    def get_init_env(self):
        return Env({}, -1, {})

    def compile(self, node_ast, e: Optional[Env] = None) -> List[Instr]:
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
        else:
            import ipdb

            ipdb.set_trace()
            return NotImplementedError

        if self.isDebug:
            print(e)

        return instructions

    @staticmethod
    def print_instructions(instructions):
        print("\n".join([f"{i.name} {i.args} {i.kwargs}" for i in instructions]))


class TestCompilerUnit(unittest.TestCase):
    def test_create_list(self):
        vm = VM(isDebug=False)
        source = "[]"
        c = Compiler(source, isDebug=False)
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack[0].els, [])
        self.assertEqual(vm.sp, 0)

    def test_print_ast(self):
        pass


class TestCompilerList(unittest.TestCase):
    def test_func_def(self):
        vm = VM(isDebug=False)
        source = """
[1, 2, 3]
        """
        c = Compiler(source, isDebug=False)
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [Array([1, 2, 3])])
        self.assertEqual(vm.sp, 0)


class TestCompilerAssign(unittest.TestCase):
    def test_reassign(self):
        vm = VM(isDebug=False)
        source = """
a = 1
b = 2
a = b
        """
        c = Compiler(source, isDebug=False)
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        # TODO: make vm.stack == [1, 2]
        #       and c.env.vars['a'] == 0 even
        #       after reassign
        self.assertEqual(vm.stack, [1, 2, 2])
        self.assertEqual(vm.sp, 2)
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
        instructions = c.compile(c.ast)
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
        instructions = c.compile(c.ast)
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
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [1, 2, 5, 8])
        self.assertEqual(vm.sp, 3)


if __name__ == "__main__":
    unittest.main()
