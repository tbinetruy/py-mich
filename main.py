import ast
import pprint

from compiler import Compiler
from vm import VM

source = """
a = 1
b = 2
c = a + b
d = 1 + b + 2 + c
f = [1, 2]
"""
c = Compiler(source, isDebug=False)
instructions = c.compile(c.ast)
vm = VM(isDebug=False)
vm._run_instructions(instructions)
print(vm.stack)
print(vm.sp)
