import ast
import pprint

from compiler import Compiler
from vm import VM

source = """
baz = 1
def foo(a):
    def ggg(n):
        return n + 2

    b = ggg(a + baz)
    return a + b

def foo2(arg):
    return arg + 12

bar = foo(baz)
fff = foo2(foo(bar))
"""
c = Compiler(source, isDebug=True)
instructions = c.compile(c.ast)
vm = VM(isDebug=True)
vm._run_instructions(instructions)

exec(source)
assert(vm.stack[-1] == fff)
