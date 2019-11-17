import unittest
import ast
import pprint

from typing import List

from vm_types import Env, Instr
from vm import VM

from helpers import ast_to_tree


class Compiler:
    def __init__(self, src):
        self.ast = ast.parse(src)

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))

    def compile_module(self, m: ast.Module, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        for key, value in ast.iter_fields(m):
            if key == 'body':
                for childNode in value:
                    instructions += self.compile(childNode, e)

        return instructions

    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = assign.targets[0].id
        value = assign.value
        instructions = self.compile(var_name, e) + self.compile(value, e)
        e.vars[var_name] = e.sp
        return instructions

    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self.compile(expr.value, e)

    def compile_num(self, num: ast.Num, e: Env) -> List[Instr]:
        e.sp += 1  # Account for PUSH
        return [
            Instr('PUSH', [num.n], {}),
        ]

    def compile_name(self, name: ast.Name, e: Env) -> List[Instr]:
        var_name = name.id
        if type(name.ctx) == ast.Load:
            var_addr = e.vars[var_name]
            jump_length = e.sp - var_addr
            e.sp += 1  # Account for DUP
            return [
                Instr('DIP', [jump_length], {}),
                Instr('DUP', [], {}),
                Instr('DIG', [], {}),
            ]
        elif type(name.ctx) == ast.Store:
            e.vars[var_name] = 42
            return []
        else:
            return NotImplementedError

    def compile_binop(self, t: ast.Name, e: Env) -> List[Instr]:
        left = self.compile(t.left, e)
        right = self.compile(t.right, e)
        op = self.compile(t.op, e)
        return left + right + op

    def compile_add(self, t: ast.Name, e: Env) -> List[Instr]:
        e.sp -= 1  # Account for ADD
        return [
            Instr('ADD', [], {}),
        ]

    def compile(self, node_ast,  e: Env = Env({}, -1)) -> List[Instr]:
        instructions: List[Instr] = []
        if type(node_ast) == ast.Module:
            instructions += self.compile_module(node_ast, e)
        if type(node_ast) == ast.Assign:
            instructions += self.compile_assign(node_ast, e)
        if type(node_ast) == ast.Expr:
            instructions += self.compile_expr(node_ast, e)
        if type(node_ast) == ast.Num:
            instructions += self.compile_num(node_ast, e)
        if type(node_ast) == ast.Name:
            instructions += self.compile_name(node_ast, e)
        if type(node_ast) == ast.BinOp:
            instructions += self.compile_binop(node_ast, e)
        if type(node_ast) == ast.Add:
            instructions += self.compile_add(node_ast, e)

        print(e)
        return instructions


class TestCompilerUnit(unittest.TestCase):
    def test_print_ast(self):
        pass



class TestCompilerIntegration(unittest.TestCase):
    def test_store_vars_and_add(self):
        vm = VM()
        source = """
a = 1
b = 2
c = a + b + b
a + b + c
        """
        c = Compiler(source)
        instructions = c.compile(c.ast)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [1, 2, 5, 8])
        self.assertEqual(vm.sp, 3)



for TestSuite in [TestCompilerUnit, TestCompilerIntegration]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)
