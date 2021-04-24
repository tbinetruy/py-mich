import ast
from typing import Any
from dataclasses import dataclass

import unittest


class TuplifyFunctionArguments(ast.NodeTransformer):
    """
    Input
    -----

    def my_function(arg1: type1, arg2: type2):
        return arg1 + arg2

    Result
    ------

    @dataclass
    Arg:
        arg1: type1
        arg2: type2

    def my_function(param: Arg):
        arg1 = param.arg1
        arg2 = param.arg2
        return arg1 + arg2

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = {}  # key: fun_nam, val: param_dataclass_name
        self.dataclasses = []
        self.defined_class_names = []

    def make_dataclass(self, name, arguments_spec):
        return ast.ClassDef(
            name=name,
            bases=[],
            keywords=[],
            body=[
                ast.AnnAssign(
                    target=ast.Name(id=argument_name, ctx=ast.Store()),
                    annotation=argument_type,
                    value=None,
                    simple=1,
                )
                for argument_name, argument_type in arguments_spec.items()
            ],
            decorator_list=[ast.Name(id='dataclass', ctx=ast.Load())]
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.defined_class_names.append(node.name)
        node.body = [self.visit(body_element) for body_element in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        prologue_body_instructions = []
        arguments = node.args.args

        # skip class instantiations and functions of 1 argument
        if len(arguments) > 1:
            ### generate argument dataclass
            arguments_spec = {
                argument_node.arg: argument_node.annotation
                for argument_node in arguments
            }
            param_dataclass_name = node.name + "Param"
            self.dataclasses.append(self.make_dataclass(param_dataclass_name, arguments_spec))

            # tuplify arguments
            param_name = node.name + "__param"

            self.env[node.name] = param_dataclass_name

            node.args.args = [ast.arg(arg=param_name, annotation=ast.Name(id=param_dataclass_name, ctx=ast.Load()))]

            # destructure tuplified arguments
            prologue_body_instructions = [
                ast.Assign(
                    targets=[ast.Name(id=attr_name, ctx=ast.Store())],
                    value=ast.Attribute(
                        value=ast.Name(id=param_name, ctx=ast.Load()),
                        attr=attr_name,
                        ctx=ast.Load()
                    ),
                    type_comment=None)
                for attr_name in arguments_spec.keys()
            ]

        new_body = [self.visit(body_node) for body_node in node.body]

        node.body = prologue_body_instructions + new_body

        return node

    def visit_Call(self, node: ast.Call) -> Any:
        fun_name = node.func.id
        if len(node.args) > 1 and fun_name not in self.defined_class_names:
            node.args = [
                ast.Call(
                    func=ast.Name(id=self.env[fun_name], ctx=ast.Load()),
                    args=[self.visit(arg) for arg in node.args],
                    keywords=[])
            ]
        return node

def macro_expander(source_ast):
    pass1 = TuplifyFunctionArguments()
    new_ast = pass1.visit(source_ast)
    new_ast.body = pass1.dataclasses + new_ast.body
    return ast.fix_missing_locations(new_ast)


class TestTuplifyFunctionArguments(unittest.TestCase):
    def test_new_function_evaluates(self):
        source = """
def add(x: int, y: int, z: int):
    return x + y + z

def increment(x: int):
    return add(x, 1, 0)

assert add(1, 2, add(3, 4, 5)) == 15
assert increment(10) == 11
"""
        f_ast = ast.parse(source)
        pass1 = TuplifyFunctionArguments()
        new_f_ast = pass1.visit(f_ast)
        new_f_ast.body = pass1.dataclasses + new_f_ast.body
        new_f_ast = ast.fix_missing_locations(new_f_ast)
        eval(compile(new_f_ast, '', mode='exec'))
        source = "add(addParam(1, 2, add(addParam(3, 4, 5))))"
        self.assertEqual(eval(source), 15)


for TestSuite in [
        #TestTuplifyFunctionArguments,
]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)
