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
        elif type(parameter) == t.Mutez:
            micheline = {
                "prim": "mutez",
            }
        elif type(parameter) == t.String:
            micheline = {
                "prim": "string",
            }
        elif type(parameter) == t.Address:
            micheline = {
                "prim": "address",
            }
        elif type(parameter) == t.Bool:
            micheline = {
                "prim": "bool",
            }
        elif type(parameter) == Pair:
            micheline = {
                "prim": "pair",
                "args": [
                    self.compile_type(parameter.car),
                    self.compile_type(parameter.cdr),
                ],
            }
        elif type(parameter) == t.Dict:
            micheline = {
                "prim": "map",
                "args": [
                    self.compile_type(parameter.key_type),
                    self.compile_type(parameter.value_type),
                ]
            }
        elif type(parameter) == t.Callable:
            micheline = {
                "prim": "lambda",
                "args": [
                    self.compile_type(parameter.param_type),
                    self.compile_type(parameter.return_type),
                ]
            }
        elif type(parameter) == t.Contract:
            micheline = {
                "prim": "contract",
                "args": [
                    self.compile_type(parameter.param_type),
                ]
            }
        else:
            breakpoint()
            return NotImplementedError

        try:
            if parameter.annotation:
                micheline["annots"] = [parameter.annotation]
        except AttributeError:
            pass

        return micheline

    def compile_instruction(self, instruction: Instr):
        if instruction.name == "TRANSFER_TOKENS":
            return {"prim": "TRANSFER_TOKENS"}
        if instruction.name == "ADD":
            return {"prim": "ADD"}
        if instruction.name == "SUB":
            return {"prim": "SUB"}
        if instruction.name == "MEM":
            return {"prim": "MEM"}
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
            if len(instruction.args):
                return {"prim": "PAIR", "args": [{"int": str(instruction.args[0])}]}
            return {"prim": "PAIR"}
        elif instruction.name == "PUSH":
            constant_type = instruction.args[0]
            constant = instruction.args[1]
            type_name = self.compile_type(constant_type)
            if type(constant_type) == t.Bool:
                pushed_constant = {"prim": "True" if constant else "False"}
            else:
                if type(constant_type) == t.Mutez:
                    constant_type = "int"
                pushed_constant = {str(constant_type): str(constant)}
            return {"prim": "PUSH", "args": [type_name, pushed_constant]}
        elif instruction.name == "SWAP":
            return {"prim": "SWAP"}
        elif instruction.name == "EXEC":
            return {"prim": "EXEC"}
        elif instruction.name == "LAMBDA":
            arg_type = self.compile_type(instruction.args[0])
            return_type = self.compile_type(instruction.args[1])
            body = self.compile_instructions(instruction.args[2])
            return {"prim": "LAMBDA", "args": [arg_type, return_type, body]}
        elif instruction.name == "EMPTY_MAP":
            key_type = self.compile_type(instruction.args[0])
            value_type = self.compile_type(instruction.args[1])
            return {"prim": "EMPTY_MAP", "args": [key_type, value_type]}
        elif instruction.name == "CONS":
            return {"prim": "CONS"}
        elif instruction.name == "SENDER":
            return {"prim": "SENDER"}
        elif instruction.name == "COMPARE":
            return {"prim": "COMPARE"}
        elif instruction.name == "EQ":
            return {"prim": "EQ"}
        elif instruction.name == "NEQ":
            return {"prim": "NEQ"}
        elif instruction.name == "LT":
            return {"prim": "LT"}
        elif instruction.name == "GT":
            return {"prim": "GT"}
        elif instruction.name == "LE":
            return {"prim": "LE"}
        elif instruction.name == "GE":
            return {"prim": "GE"}
        elif instruction.name == "GET":
            if not instruction.args:
                return {"prim": "GET"}
            else:
                return {
                    "prim": "GET",
                    "args": [{"int": str(instruction.args[1])}]
                }
        elif instruction.name == "AND":
            return {"prim": "AND"}
        elif instruction.name == "OR":
            return {"prim": "OR"}
        elif instruction.name == "FAILWITH":
            return {"prim": "FAILWITH"}
        elif instruction.name == "SOME":
            return {"prim": "SOME"}
        elif instruction.name == "NONE":
            return {"prim": "NONE"}
        elif instruction.name == "UPDATE":
            if not instruction.args:
                return {"prim": "UPDATE"}
            else:
                return {
                    "prim": "UPDATE",
                    "args": [{"int": str(instruction.args[1])}]
                }
        elif instruction.name == "NOT":
            return {"prim": "NOT"}
        elif instruction.name == "IF":
            return {
                "prim": "IF",
                "args": [
                    self.compile_instructions(instruction.args[0]),
                    self.compile_instructions(instruction.args[1]),
                ],
            }
        elif instruction.name == "IF_NONE":
            return {
                "prim": "IF_NONE",
                "args": [
                    self.compile_instructions(instruction.args[0]),
                    self.compile_instructions(instruction.args[1]),
                ],
            }
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

