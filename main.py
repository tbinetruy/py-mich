import ast
import pprint

from . import Compiler

source = """
def f(c):
    a = 1
    b = 2
    return a + b + c
f(3)
"""
c = Compiler(source)
c.print_ast()
