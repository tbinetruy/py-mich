import unittest
import ast
import pprint

import vm_types


# thanks http://dev.stephendiehl.com/numpile/ :)
def ast_to_tree(node, include_attrs=True):
    def _transform(node):
        if isinstance(node, ast.AST):
            fields = ((a, _transform(b))
                    for a, b in ast.iter_fields(node))
            if include_attrs:
                attrs = ((a, _transform(getattr(node, a)))
                        for a in node._attributes
                        if hasattr(node, a))
                return (node.__class__.__name__, dict(fields), dict(attrs))
            return (node.__class__.__name__, dict(fields))
        elif isinstance(node, list):
            return [_transform(x) for x in node]
        elif isinstance(node, str):
            return repr(node)
        return node
    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _transform(node)


class Compiler:
    def __init__(self, src):
        self.ast = ast.parse(src)

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))


source = """
def f(c):
    a = 1
    b = 2
    return a + b + c
f(3)
"""
c = Compiler(source)
c.print_ast()

class TestCompiler(unittest.TestCase):
    def test_print_ast(self):
        pass

suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCompiler)
unittest.TextTestRunner().run(suite)
