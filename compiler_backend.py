import unittest
from typing import Any, List

import instr_types as t
from vm_types import Contract, Instr, Or, Pair


class CompilerBackend:
    def __init__(self):
        pass

    def compile_type(self, parameter):
        if type(parameter) == Or:
            micheline = {
                "prim": "or",
                "args": [
                    self.compile_type(parameter.left),
                    self.compile_type(parameter.right),
                ],
            }
        elif type(parameter) == t.Int:
            micheline = {
                "prim": "int",
            }
        elif type(parameter) == t.String:
            micheline = {
                "prim": "string",
            }
        elif type(parameter) == t.Address:
            micheline = {
                "prim": "address",
            }
        elif type(parameter) == Pair:
            micheline = {
                "prim": "pair",
                "args": [
                    self.compile_type(parameter.car),
                    self.compile_type(parameter.cdr),
                ],
            }

        else:
            return NotImplementedError

        try:
            if parameter.annotation:
                micheline["annots"] = [parameter.annotation]
        except AttributeError:
            pass

        return micheline

    def compile_instruction(self, instruction: Instr):
        if instruction.name == "ADD":
            return {"prim": "ADD"}
        elif instruction.name == "CAR":
            return {"prim": "CAR"}
        elif instruction.name == "CDR":
            return {"prim": "CDR"}
        elif instruction.name == "DIP":
            return {
                "prim": "DIP",
                "args": [
                    {"int": str(instruction.args[0])},
                    self.compile_instructions(instruction.args[1]),
                ],
            }
        elif instruction.name == "DIG":
            return {"prim": "DIG", "args": [{"int": str(instruction.args[0])}]}
        elif instruction.name == "DUG":
            return {"prim": "DUG", "args": [{"int": str(instruction.args[0])}]}
        elif instruction.name == "DROP":
            return {"prim": "DROP"}
        elif instruction.name == "DUP":
            return {"prim": "DUP"}
        elif instruction.name == "NIL":
            return {"prim": "NIL", "args": [{"prim": str(instruction.args[0])}]}
        elif instruction.name == "LIST":
            return NotImplementedError
        elif instruction.name == "PAIR":
            return {"prim": "PAIR"}
        elif instruction.name == "PUSH":
            constant_type = instruction.args[0]
            constant = instruction.args[1]
            type_name = self.compile_type(constant_type)
            pushed_constant = {str(constant_type): str(constant)}
            return {"prim": "PUSH", "args": [type_name, pushed_constant]}
        elif instruction.name == "SWAP":
            return {"prim": "SWAP"}
        elif instruction.name == "EXEC":
            return {"prim": "EXEC"}
        elif instruction.name == "LAMBDA":
            return {"prim": ""}
        elif instruction.name == "CONS":
            return {"prim": "CONS"}
        elif instruction.name == "IF_LEFT":
            return {
                "prim": "IF_LEFT",
                "args": [
                    self.compile_instructions(instruction.args[0]),
                    self.compile_instructions(instruction.args[1]),
                ],
            }
        else:
            breakpoint()
            raise NotImplementedError

    def compile_instructions(self, instructions: List[Instr]):
        print(instructions)
        if type(instructions) != list:
            breakpoint()
        micheline = []
        for instruction in instructions:
            if not instruction.name == "COMMENT":
                micheline.append(self.compile_instruction(instruction))

        return micheline

    def compile_contract(self, contract: Contract):
        parameter_type = self.compile_type(contract.get_parameter_type())
        storage_type = self.compile_type(contract.get_storage_type())
        code = self.compile_instructions(contract.instructions) + [
            self.compile_instruction(contract.get_contract_body())
        ]
        return [
            {
                "prim": "parameter",
                "args": [parameter_type],
            },
            {
                "prim": "storage",
                "args": [storage_type],
            },
            {
                "prim": "code",
                "args": [code],
            },
        ]


class TestCompilerBackend(unittest.TestCase):
    def test_compile_param(self):
        b = CompilerBackend()
        compiled_parameter = b.compile_type(Or(t.Int(), t.Int()))
        expected_parameter = {
            "prim": "or",
            "args": [
                {
                    "prim": "int",
                },
                {
                    "prim": "int",
                },
            ],
        }
        self.assertEqual(compiled_parameter, expected_parameter)


for TestSuite in [
    TestCompilerBackend,
]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)


from compiler import Compiler

source = """
class Contract:
    def incrementByTwo(a: int) -> int:
        b = 1
        return a + b + 1

    def bar(b: int) -> int:
        return b
"""
c = Compiler(source, isDebug=False)
c.compile()
b = CompilerBackend()

micheline = b.compile_contract(c.contract)

import json

with open("my_contract.json", "w+") as f:
    f.write(json.dumps(micheline))
