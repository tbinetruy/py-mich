import ast
import pprint
import unittest
from typing import List

from helpers import ast_to_tree
from vm import VM
from vm_types import Env, Instr, Array


def debug(cb):
    def f(*args, **kwargs):
        self = args[0]
        if self.isDebug:
            print(cb.__name__)

        return cb(*args, **kwargs)
    return f

class Compiler:
    def __init__(self, src, isDebug=True):
        self.ast = ast.parse(src)
        self.isDebug = isDebug

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))

    def compile_module(self, m: ast.Module, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        for key, value in ast.iter_fields(m):
            if key == 'body':
                for childNode in value:
                    instructions += self.compile(childNode, e)

        return instructions

    @debug
    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = assign.targets[0].id
        value = assign.value
        instructions = self.compile(var_name, e) + self.compile(value, e)
        e.vars[var_name.id] = e.sp
        return instructions

    @debug
    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self.compile(expr.value, e)

    @debug
    def compile_num(self, num: ast.Num, e: Env) -> List[Instr]:
        e.sp += 1  # Account for PUSH
        return [
            Instr('PUSH', [num.value], {}),
        ]

    @debug
    def compile_name(self, name: ast.Name, e: Env) -> List[Instr]:
        var_name = name
        if type(name.ctx) == ast.Load:
            var_addr = e.vars[var_name.id]
            jump_length = e.sp - var_addr
            e.sp += 1  # Account for DUP
            return [
                Instr('DIP', [jump_length], {}),
                Instr('DUP', [], {}),
                Instr('DIG', [], {}),
            ]
        elif type(name.ctx) == ast.Store:
            e.vars[var_name.id] = 42
            return []
        else:
            return NotImplementedError

    @debug
    def compile_binop(self, t: ast.Name, e: Env) -> List[Instr]:
        left = self.compile(t.left, e)
        right = self.compile(t.right, e)
        op = self.compile(t.op, e)
        return left + right + op

    @debug
    def compile_add(self, t: ast.Add, e: Env) -> List[Instr]:
        e.sp -= 1  # Account for ADD
        return [
            Instr('ADD', [], {}),
        ]

    @debug
    def create_list(self, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        return [
            Instr('LIST', [], {}),
        ]

    @debug
    def append_before_list_el(self, el, e) -> List[Instr]:
        # no sp chage b/c they cancel out btwn the two instructions
        return self.compile(el, e) + [Instr('CONS', [], {})]

    @debug
    def compile_list(self, l: ast.List, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        instructions = self.create_list(e)
        for el in reversed(l.elts):
            instructions += self.append_before_list_el(el, e)
        return instructions

    @debug
    def compile_defun(self, f: ast.FunctionDef, e: Env):
        e.sp += 1  # account for body push

        e.vars[f.name] = e.sp
        e.args[f.name] = f.args.args[0].arg

        func_env = e.copy()
        func_env.vars[f.args.args[0].arg] = e.sp

        return [
            Instr('PUSH', [self.compile(f.body[0], func_env)], {})
        ]

    @debug
    def compile_fcall(self, f: ast.FunctionDef, e: Env):

        # fetch arg name for function
        arg_name = e.args[f.func.id]

        # compile arg
        arg = self.compile(f.args[0])

        # Account for pushing argument
        e.sp += 1

        # Store arg stack location
        e.vars[arg_name] = e.sp - 1

            #var_addr = e.vars[var_name.id]
            #jump_length = e.sp - var_addr
            #e.sp += 1  # Account for DUP
            #return [
            #    Instr('DIP', [jump_length], {}),
            #    Instr('DUP', [], {}),
            #    Instr('DIG', [], {}),
            #]

        return arg + [Instr('EXEC', [], {})]

    @debug
    def compile_return(self, r: ast.FunctionDef, e: Env):
        return self.compile(r.value, e)

    def compile(self, node_ast,  e: Env = Env({}, -1, {})) -> List[Instr]:
        instructions: List[Instr] = []
        if type(node_ast) == ast.Module:
            instructions += self.compile_module(node_ast, e)
            if self.isDebug:
                self.print_instructions(instructions)
        if type(node_ast) == ast.Assign:
            instructions += self.compile_assign(node_ast, e)
        if type(node_ast) == ast.Expr:
            instructions += self.compile_expr(node_ast, e)
        if type(node_ast) == ast.Constant:
            instructions += self.compile_num(node_ast, e)
        if type(node_ast) == ast.Name:
            instructions += self.compile_name(node_ast, e)
        if type(node_ast) == ast.BinOp:
            instructions += self.compile_binop(node_ast, e)
        if type(node_ast) == ast.Add:
            instructions += self.compile_add(node_ast, e)
        if type(node_ast) == ast.List:
            instructions += self.compile_list(node_ast, e)
        elif type(node_ast) == ast.FunctionDef:
            instructions += self.compile_defun(node_ast, e)
        elif type(node_ast) == ast.Return:
            instructions += self.compile_return(node_ast, e)
        elif type(node_ast) == ast.Call:
            instructions += self.compile_fcall(node_ast, e)

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
        c.print_ast()
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [Array([1, 2, 3])])
        self.assertEqual(vm.sp, 0)

class TestCompilerDefun(unittest.TestCase):
    def test_func_def(self):
        vm = VM(isDebug=False)
        source = """
def foo(a):
    return a + 1
foo(20)
        """
        c = Compiler(source, isDebug=False)
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [21])
        self.assertEqual(vm.sp, 0)



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
    for TestSuite in [TestCompilerUnit, TestCompilerIntegration]:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
        unittest.TextTestRunner().run(suite)
