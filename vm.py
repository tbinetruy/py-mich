import copy
import unittest
from typing import Any, Callable, Dict, List, Tuple

from instr_types import Int
from vm_types import (Array, Contract, Entrypoint, FunctionPrototype, Instr,
                      Left, Or, Pair, Right)

ph = 42  # placeholder


class VMStackException(Exception):
    """Raised when stack error occurs

    Attributes:
        message -- error message
    """

    def __init__(self, message):
        self.message = message


class VMTypeException(Exception):
    """Raised when type error occurs

    Attributes:
        expected -- expected type
        actual -- type the vm got
        message -- explanation of type error
    """

    def __init__(self, expected, actual, message):
        self.expected = expected
        self.actual = actual
        self.message = message


def debug(in_func):
    def out_func(*args, **kwargs):
        self = args[0]

        # if self.isDebug:
        #    print(in_func, args, kwargs)

        result = in_func(*args, **kwargs)

        # self._debug()
        return result

    return out_func


class VM:
    @staticmethod
    def get_init_stack():
        return []

    @staticmethod
    def get_init_sp():
        return -1

    def __init__(self, isDebug=False):
        self.isDebug = isDebug

        # See: https://tezos.gitlab.io/whitedoc/michelson.html
        self.instruction_mapping = {
            "ADD": self.add,
            "CAR": self.car,
            "CDR": self.cdr,
            "DIP": self.dip,
            "DIG": self.dig,
            "DUG": self.dug,
            "DROP": self.pop,
            "DUP": self.dup,
            "IIP": self.increment_sp,
            "LIST": self.make_list,
            "PAIR": self.make_pair,
            "PUSH": self.push,
            "SWAP": self.swap,
            "EXEC": self.run_lambda,
            "LAMBDA": self.store_lambda,
            "CONS": self.append_before_list,
            "IF_LEFT": self.if_left,
            "COMMENT": lambda *args, **kwargs: 1,
        }

        self._reset_stack()

    def store_lambda(self, args_types, return_type, body) -> None:
        self._push(body)

    def run_contract(
        self, contract: Contract, entrypoint_name: str, entrypoint_param: Any
    ) -> None:
        contract_param = contract.make_contract_param(entrypoint_name, entrypoint_param)
        body_instructions = [contract.get_contract_body()]
        self._push(Pair(car=contract_param, cdr=contract.storage))
        self._run_instructions(contract.instructions + body_instructions)
        contract.storage = self.stack[-1]

    def _run_instructions(self, instructions: List[Instr]) -> None:
        for instr in instructions:
            if self.isDebug:
                print("=== ", instr.name, instr.args, instr.kwargs, " ===")
                self._debug()
            instr_function = self.instruction_mapping[instr.name]
            instr_function(*instr.args, **instr.kwargs)
            if self.isDebug:
                self._debug()

    def _reset_stack(self):
        self.stack = self.get_init_stack()
        self.sp = self.get_init_sp()

    def get_state_as_str(self):
        def clean(el):
            if type(el) == list:
                return "[func]"
            else:
                return el

        stack = [
            ("*", clean(el)) if i == self.sp else clean(el)
            for i, el in enumerate(self.stack)
        ]
        return "S: " + str(stack) + " ; sp: ", str(self.sp)

    def _debug(self):
        if self.isDebug:
            print(self.get_state_as_str())

    def _stack_top(self):
        return self.stack[-1]

    def _assert_min_stack_length(self, min_len: int) -> None:
        stack_length = len(self.stack)
        if stack_length < min_len:
            raise VMStackException(
                "Stack too short, need to be at least of length "
                + str(min_len)
                + " but is currently "
                + str(stack_length)
            )

    def _check_union(self):
        """Checks that the stack top element is a union type."""
        self._assert_min_stack_length(1)

        actual_t = type(self.stack[-1])
        expected_t = [Left, Right]
        if actual_t not in expected_t:
            raise VMTypeException(
                expected_t, actual_t, "Stack top element is not a union."
            )

    def _check_pair(self):
        self._assert_min_stack_length(1)

        expected_t = type(self.stack[-1])
        actual_t = Pair
        if expected_t != actual_t:
            raise VMTypeException(expected_t, actual_t, "Car requires a pair")

    @debug
    def if_left(self, cond_true, cond_false) -> None:
        self._check_union()
        union = self.pop()
        self._push(union.value)
        if type(union) == Left:
            self._run_instructions(cond_true)
        else:
            self._run_instructions(cond_false)

    @debug
    def run_lambda(self):
        self._assert_min_stack_length(2)

        if self.isDebug:
            print("@@@@@@@ Start executing function @@@@@@@")

        arg = self.pop()
        function_body = self.pop()
        protected_stack = self.stack
        self.stack = [arg]
        self._run_instructions(function_body)
        self.stack = protected_stack + self.stack

        if self.isDebug:
            print("@@@@@@@ End executing function @@@@@@@")

    @debug
    def add(self):
        self._assert_min_stack_length(2)
        a = self.pop()
        b = self.pop()
        self._push(a + b)

    def get_f_from_instr_name(self, instr_name: str):
        return self.instruction_mapping[instr_name]

    @debug
    def car(self):
        self._check_pair()
        pair = self.pop()
        self._push(pair.car)

    @debug
    def cdr(self):
        self._check_pair()
        pair = self.pop()
        self._push(pair.cdr)

    @debug
    def make_pair(self, car: Any, cdr: Any):
        self._push(Pair(car, cdr))

    @debug
    def make_list(self):
        self._push(Array([]))

    @debug
    def append_before_list(self):
        self._assert_min_stack_length(2)

        el = self.pop()
        array = self.pop()

        actual_t = type(array)
        expected_t = Array
        if expected_t != actual_t:
            raise VMTypeException(expected_t, actual_t, "CONS requires a list")

        self._push(Array([el] + array.els))

    @debug
    def dip(self, delta: int = 1, instructions: List[Instr] = []):
        self._assert_min_stack_length(delta)

        protected_stack_top = []
        if delta != 0:
            protected_stack_top = self.stack[-delta:]
            self.stack = self.stack[:-delta]

        self._run_instructions(instructions)

        self.stack += protected_stack_top

    @debug
    def decrement_sp(self, delta: int = 1):
        self.sp -= delta

    @debug
    def increment_sp(self, delta: int = 1):
        self.sp += delta

    def _push(self, val):
        self.stack += [val]

    @debug
    def push(self, val_type, val):
        self._push(val)

    @debug
    def pop(self):
        """Removes and returns the element int the stack at the
        stack pointer location and decrements it *unless* sp is the
        bottom of a non-empty stack in which case it does not change."""
        self._assert_min_stack_length(1)

        el = self.stack[-1]
        del self.stack[-1]

        return el

    @debug
    def swap(self):
        self._assert_min_stack_length(2)

        a = self.pop()
        b = self.pop()
        self._push(a)
        self._push(b)

    @debug
    def dup(self):
        self._assert_min_stack_length(1)

        self.stack += [copy.deepcopy(self.stack[-1])]

    @debug
    def dig(self, jump=None):
        reverse_index = - jump - 1
        el_to_dig = self.stack[reverse_index]
        del self.stack[reverse_index]
        self.stack.append(el_to_dig)

    @debug
    def dug(self, jump=None):
        self._assert_min_stack_length(jump + 1)
        self.stack.insert(len(self.stack) - 1 - jump, self.stack[-1])
        del self.stack[-1]


class TestContract(unittest.TestCase):
    def test_run_contract(self):
        contract = Contract(
            storage=10,
            storage_type=Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 1], {}),
                        Instr("ADD", [], {}),
                        Instr("DIP", [1, [Instr("DROP", [], {})]], {}),
                    ],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 2], {}),
                        Instr("ADD", [], {}),
                        Instr("DIP", [1, [Instr("DROP", [], {})]], {}),
                    ],
                ),
                "div": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 3], {}),
                        Instr("ADD", [], {}),
                        Instr("DIP", [1, [Instr("DROP", [], {})]], {}),
                    ],
                ),
            },
            instructions=[
                Instr("DUP", [], {}),     # stack: [(param, storage), * (param, storage)]
                Instr("CAR", [], {}),     #        [(param, storage), * param]
                Instr("DIP", [1, [        #        [* (param, storage), param]
                    Instr("CDR", [], {})  # stack: [* storage, param]
                ]], {}),
            ],
        )

        vm = VM()
        vm.run_contract(contract, "add", 10)
        self.assertEqual(contract.storage, 11)
        self.assertEqual(vm.stack, [11])

    def test_contract(self):
        contract = Contract(
            storage=10,
            storage_type=Int(),
            entrypoints={
                "add": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 1], {}),
                        Instr("ADD", [], {}),
                    ],
                ),
                "sub": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 2], {}),
                        Instr("ADD", [], {}),
                    ],
                ),
                "div": Entrypoint(
                    FunctionPrototype(Int(), Int()),
                    [
                        Instr("PUSH", [Int(), 3], {}),
                        Instr("ADD", [], {}),
                    ],
                ),
            },
            instructions=[],
        )
        instructions = [contract.get_contract_body()]

        vm = VM()
        contract_param = contract.make_contract_param("add", 1)
        vm._push(contract_param)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [2])

        vm = VM()
        contract_param = contract.make_contract_param("sub", 1)
        vm._push(contract_param)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [3])

        vm = VM()
        contract_param = contract.make_contract_param("div", 1)
        vm._push(contract_param)
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [4])


class TestVM(unittest.TestCase):
    def test_if_left(self):
        vm = VM()
        instructions = [
            Instr("PUSH", [Int(), Left(10)], {}),
            Instr(
                "IF_LEFT",
                [
                    [
                        Instr("PUSH", [Int(), 10], {}),
                        Instr("ADD", [], {}),
                    ],
                    [
                        Instr("PUSH", [Int(), 20], {}),
                        Instr("ADD", [], {}),
                    ],
                ],
                {},
            ),
        ]
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [20])

    def test_if_not_left(self):
        vm = VM()
        instructions = [
            Instr("PUSH", [Int(), Right(10)], {}),
            Instr(
                "IF_LEFT",
                [
                    [
                        Instr("PUSH", [Int(), 10], {}),
                        Instr("ADD", [], {}),
                    ],
                    [
                        Instr("PUSH", [Int(), 20], {}),
                        Instr("ADD", [], {}),
                    ],
                ],
                {},
            ),
        ]
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [30])

    def test_if_left_stack_top_type(self):
        vm = VM()
        instructions = [
            Instr("PUSH", [Int(), 10], {}),
            Instr(
                "IF_LEFT",
                [[], []],
                {},
            ),
        ]
        try:
            vm._run_instructions(instructions)
            assert 0
        except VMTypeException:
            assert 1

    def test_store_lambda(self):
        vm = VM()
        body = [
            Instr("PUSH", [Int, 2], {}),
            Instr("ADD", [], {}),
        ]
        arg = 2
        arg_types = ([Int],)
        return_type = Int
        vm._run_instructions(
            [
                Instr("LAMBDA", [arg_types, return_type, body], {}),
                Instr("PUSH", [Int, arg], {}),
                Instr("EXEC", [], {}),
            ]
        )
        assert [arg + 2] == vm.stack

    def test_run_lambda(self):
        vm = VM()
        body = [
            Instr("DUP", [], {}),
            Instr("ADD", [], {}),
        ]
        arg = 2
        vm._push(body)
        vm._push(arg)
        vm._run_instructions([Instr("EXEC", [], {})])
        assert [arg * 2] == vm.stack

    def test_run_instructions(self):
        vm = VM()
        instructions = [
            Instr("PUSH", [Int, 1], {}),
            Instr("PUSH", [Int, 2], {}),
            Instr("SWAP", [], {}),
        ]
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [2, 1])

    def test_check_pair(self):
        vm = VM()
        vm._reset_stack()

        vm._push(1)
        self.assertRaises(VMTypeException, vm._check_pair)
        vm.make_pair(1, 2)
        try:
            self.assertRaises(VMTypeException, vm._check_pair)
            self.fail("check_pair raised VMTypeException unexpectedly!")
        except AssertionError:
            pass

    def test_stack_top(self):
        vm = VM()

        b = 2
        vm._push(1)
        vm._push(2)

        self.assertEqual(vm._stack_top(), b)

    def test_reset_stack(self):
        vm = VM()
        vm._reset_stack()
        self.assertEqual(vm.sp, VM.get_init_sp())
        self.assertEqual(vm.stack, VM.get_init_stack())

    def test_increment_sp(self):
        vm = VM()

        vm.increment_sp()
        vm.increment_sp()
        self.assertEqual(vm.sp, VM.get_init_sp() + 2)

    def test_decrement_sp(self):
        vm = VM()

        vm.decrement_sp()
        self.assertEqual(vm.sp, VM.get_init_sp() - 1)

    def test_push(self):
        vm = VM()

        a, b, c = 1, 2, 3
        vm.push(Int, a)
        vm.push(Int, b)
        vm.push(Int, c)
        # stack grows towards larger addresses
        self.assertEqual(vm.stack, [a, b, c])

    def test_pop(self):
        vm = VM()
        a, b, c = 1, 2, 3

        # check that poping empty stack raises
        self.assertRaises(VMStackException, vm.pop)

        ### check that we can pop on top of the stack
        vm._push(a)
        vm._push(b)
        vm.pop()
        self.assertEqual(vm.stack, [a])

        vm.pop()
        self.assertRaises(VMStackException, vm.pop)

    def test_make_list(self):
        vm = VM()

        a, b = 1, 2
        vm.make_list()
        self.assertEqual(vm.stack[-1].__class__, Array([]).__class__)

    def test_make_pair(self):
        vm = VM()

        a, b = 1, 2
        vm.make_pair(a, b)
        self.assertEqual([Pair(a, b)], vm.stack)

    def test_car(self):
        vm = VM()

        a, b = 1, 2
        vm.make_pair(a, b)
        vm.car()
        self.assertEqual(vm._stack_top(), a)

    def test_cdr(self):
        vm = VM()
        a, b = 1, 2
        vm.make_pair(a, b)
        vm.cdr()
        self.assertEqual(vm._stack_top(), b)

    def test_swap(self):
        vm = VM()
        a, b, c = 1, 2, 3

        # Raises if stack too small
        self.assertRaises(VMStackException, vm.swap)
        vm._push(a)
        self.assertRaises(VMStackException, vm.swap)

        # swaps top of stack
        vm._push(b)
        vm._push(c)
        vm.swap()
        self.assertEqual(vm.stack, [1, 3, 2])

    def test_dup(self):
        vm = VM()

        self.assertRaises(VMStackException, vm.dup)

        a, b, c = 1, 2, 3
        vm._push(a)
        vm._push(b)
        vm.dup()
        self.assertEqual(vm.stack, [a, b, b])

    def test_add(self):
        vm = VM()

        a, b, c, d = 1, 2, 3, 4
        vm._push(a)
        vm._push(b)
        vm._push(c)
        vm._push(d)
        vm.add()
        self.assertEqual(vm.stack, [a, b, c + d])
        vm.add()
        self.assertEqual(vm.stack, [a, b + c + d])
        vm.add()
        self.assertEqual(vm.stack, [a + b + c + d])
        try:
            vm.add()
            assert 0
        except VMStackException:
            assert 1

    def test_dip(self):
        vm = VM()

        a, b, c = 1, 2, 3
        vm._push(a)
        vm._push(b)
        vm._push(c)
        vm.dip(1, [Instr("ADD", [], {})])
        self.assertEqual(vm.stack, [a + b, c])
        vm.dip(0, [Instr("ADD", [], {})])
        self.assertEqual(vm.stack, [a + b + c])
        try:
            vm.dip(10, [Instr("ADD", [], {})])
            assert 0
        except VMStackException:
            assert 1

    def test_dig(self):
        vm = VM()

        a, b, c = 1, 2, 3
        vm._push(a)
        vm._push(b)
        vm._push(c)
        vm.dig(1)  # eq to swap
        self.assertEqual(vm.stack, [a, c, b])
        vm.dig(0)  # eq to no op
        self.assertEqual(vm.stack, [a, c, b])
        vm.dig(2)
        self.assertEqual(vm.stack, [c, b, a])

    def test_dug(self):
        vm = VM()

        a, b, c = 1, 2, 3
        vm._push(a)
        vm._push(b)
        vm._push(c)
        vm.dug(1)  # eq to swap
        self.assertEqual(vm.stack, [a, c, b])
        vm.dug(0)  # eq to no op
        self.assertEqual(vm.stack, [a, c, b])
        vm.dug(2)
        self.assertEqual(vm.stack, [b, a, c])

    def test_append_before_list(self):
        vm = VM()

        a, b = 1, 2
        vm.make_list()
        vm._push(a)
        vm.append_before_list()
        vm._push(b)
        vm.append_before_list()
        self.assertEqual(vm.stack[0].els, [b, a])


for TestSuite in [TestContract, TestVM]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVM)
    unittest.TextTestRunner().run(suite)
