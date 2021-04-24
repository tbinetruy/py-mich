import ast
from typing import Any
from dataclasses import dataclass

import unittest


def make_dataclass(name, arguments_spec):
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
            self.dataclasses.append(make_dataclass(param_dataclass_name, arguments_spec))

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


class AssignAllFunctionCalls(ast.NodeTransformer):
    """
    Input
    -----

    foo(*args)

    Result
    ------

    _ = foo(*args)
    """

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Funcalls which's return value are not assigned are wrapped in an ast.Expr"""
        if type(node.value) == ast.Call:
            call_node = node.value
            return ast.Assign(
                targets=[ast.Name(id='__placeholder__', ctx=ast.Store())],
                value=call_node,
                type_comment=None)

        return node


class RemoveSelfArgFromMethods(ast.NodeTransformer):
    """
    Input
    -----

    class C:
        def f(self, x: t1, y: t2) -> t3:

    Result
    ------

    class C:
        def f(x: t1, y: t2) -> t3:
    """

    def remove_first_untyped_arg(self, node: ast.FunctionDef) -> Any:
        if len(node.args.args) and node.args.args[0].annotation == None:
            del node.args.args[0]

        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        new_body = []
        for body_node in node.body:
            if type(body_node) == ast.FunctionDef:
                new_body.append(self.remove_first_untyped_arg(body_node))
            else:
                new_body.append(body_node)

        return node


class ExpandStorageInEntrypoints(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.value.id != 'self':
            return node

        node.value = ast.Attribute(
            value=ast.Name(id=node.value.id, ctx=ast.Load()),
            attr='storage',
            ctx=ast.Load()
        )

        return node

class FactorOutStorage(ast.NodeTransformer):
    """
    Input
    -----

    @dataclass
    class Contract:
        counter: int
        admin: address

        def update_counter(self, new_counter: int) -> int:
            self.counter = 1

    Output
    ------

    @dataclass
    class Storage
        counter: int
        admin: address

    class Contract:
        def deploy():
            return Storage()

        def update_counter(self, new_counter: int) -> int:
            self.storage.counter = 1

            return self.storage
    """
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if node.name != "Contract":
            return node

        # if there is a `deploy` method, do nothing
        for body_node in node.body:
            if type(body_node) == ast.FunctionDef:
                if body_node.name == "deploy":
                    return node

        # Factor body ast.AnnAssign into dataclass
        storage_keys_spec = {}
        new_node_body = []
        for i, body_node in enumerate(node.body):
            if type(body_node) == ast.AnnAssign:
                storage_keys_spec[body_node.target.id] = body_node.annotation
            else:
                new_node_body.append(body_node)
        node.body = new_node_body

        self.storage_dataclass = make_dataclass('Storage', storage_keys_spec)

        # For all methods, update `self.<storage_key>` into `self.storage.<key>`
        # and add return `self.storage`
        new_body = []
        for body_node in node.body:
            new_body_node = ExpandStorageInEntrypoints().visit(body_node)
            if type(body_node) == ast.FunctionDef:
                return_storage_node = ast.Return(
                    value=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr='storage',
                        ctx=ast.Load()
                    )
                )
                new_body_node.body.append(return_storage_node)
            new_body.append(new_body_node)
        node.body = new_body

        # Create deploy function
        deploy_function_node = ast.FunctionDef(
            name='deploy',
            args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[ast.Return(value=ast.Call(func=ast.Name(id='Storage', ctx=ast.Load()), args=[], keywords=[]))],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        node.body = [deploy_function_node] + node.body

        return node


def macro_expander(source_ast):
    pass1 = FactorOutStorage()
    pass2 = RemoveSelfArgFromMethods()
    pass3 = TuplifyFunctionArguments()
    pass4 = AssignAllFunctionCalls()
    new_ast = pass1.visit(source_ast)
    new_ast = pass2.visit(new_ast)
    new_ast = pass3.visit(new_ast)
    new_ast = pass4.visit(new_ast)
    new_ast.body = pass3.dataclasses + new_ast.body
    if hasattr(pass1, 'storage_dataclass'):
        new_ast.body = [pass1.storage_dataclass] + new_ast.body
    return ast.fix_missing_locations(new_ast)


class TestFactorOutStorage(unittest.TestCase):
    def test_new_function_evaluates(self):
        source = """
class Contract:
    counter: int
    admin: str

    def update_counter(self, new_counter: int) -> int:
        self.counter = 1

    def update_admin(self, new_admin: str) -> int:
        self.admin = new_admin
        """
        source_ast = ast.parse(source)
        expander = FactorOutStorage()
        new_ast = expander.visit(source_ast)
        new_ast.body = [expander.storage_dataclass] + new_ast.body
        new_ast = ast.fix_missing_locations(new_ast)


class TestRemoveSelfArgFromMethods(unittest.TestCase):
    def test_new_function_evaluates(self):
        source = """
class C:
    def f(self, x: int, y: str, z: int): return 1
        """
        source_ast = ast.parse(source)
        new_ast = RemoveSelfArgFromMethods().visit(source_ast)
        new_method_ast = source_ast.body[0].body[0]
        self.assertEqual(len(new_method_ast.args.args), 3)

        for arg_node, arg_name, arg_type in zip(new_method_ast.args.args, ['x', 'y', 'z'], ['int', 'str', 'int']):
            self.assertEqual(arg_node.arg, arg_name)
            self.assertEqual(arg_node.annotation.id, arg_type)


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


class TestAssignAllFunctionCallsTests(unittest.TestCase):
    def test_function_call_in_block(self):
        source = """
f = lambda x: x
if True:
    y = f(1)
    f(2)

assert __placeholder__ == 2
assert y == 1
        """
        source_ast = ast.parse(source)
        new_ast = AssignAllFunctionCalls().visit(source_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        eval(compile(new_ast, '', mode='exec'))


for TestSuite in [
        TestTuplifyFunctionArguments,
        TestAssignAllFunctionCallsTests,
        TestRemoveSelfArgFromMethods,
        TestFactorOutStorage,
]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)
