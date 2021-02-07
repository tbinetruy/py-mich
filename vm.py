import unittest

from typing import Tuple, List, Callable, Dict

from vm_types import Pair, Instr, Array


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

        #if self.isDebug:
        #    print(in_func, args, kwargs)

        result = in_func(*args, **kwargs)

        #self._debug()
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
            "DIP": self.decrement_sp,
            "DIG": self.dig,
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
            "COMMENT": lambda *args, **kwargs: 1,
        }

        self._reset_stack()

    def store_lambda(self, args_types, return_type, body) -> None:
        self.push(body)

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
            ("*", clean(el))
            if i == self.sp
            else clean(el)
            for i, el in enumerate(self.stack)
        ]
        return "S: " + str(stack) + " ; sp: ", str(self.sp)

    def _debug(self):
        if self.isDebug:
            print(self.get_state_as_str())

    def _stack_top(self):
        return self.stack[-1]

    def _stack_at_sp(self):
        return self.stack[self.sp] \
            if self.sp >= 0 and self.sp <= len(self.stack) \
            else self.get_init_sp()

    def _assert_min_stack_length(self, min_len: int) -> None:
        stack_length = len(self.stack)
        if stack_length < min_len:
            raise VMStackException("Stack too short, need to be at least of length " + str(min_len) + " but is currently " + str(stack_length))

    def _check_pair(self):
        self._assert_min_stack_length(1)

        expected_t = self._stack_top().__class__
        actual_t = Pair(ph, ph).__class__
        if expected_t != actual_t:
            raise VMTypeException(expected_t, actual_t, "Car requires a pair")

    @debug
    def run_lambda(self):
        self._assert_min_stack_length(2)

        if self.isDebug:
            print("@@@@@@@ Start executing function @@@@@@@")
        self._run_instructions(self.stack[self.sp - 1])
        if self.isDebug:
            print("@@@@@@@ End executing function @@@@@@@")

    @debug
    def add(self):
        self._assert_min_stack_length(2)
        a = self.pop()
        b = self.stack[self.sp]
        self.stack[self.sp] = a + b

    def get_f_from_instr_name(self, instr_name: str):
        return self.instruction_mapping[instr_name]

    @debug
    def car(self):
        self._check_pair()
        pair = self.pop()
        self.push(pair.car)

    @debug
    def cdr(self):
        self._check_pair()
        pair = self.pop()
        self.push(pair.cdr)

    @debug
    def make_pair(self, car: any, cdr: any):
        self.push(Pair(car, cdr))

    @debug
    def make_list(self):
        self.push(Array([]))

    @debug
    def append_before_list(self):
        self._assert_min_stack_length(1)

        el = self.pop()

        actual_t = self._stack_top().__class__
        expected_t = Array([]).__class__
        if expected_t != actual_t:
            raise VMTypeException(expected_t, actual_t, "CONS requires a list")

        self.stack[self.sp] = Array([el] + self._stack_at_sp().els)

    @debug
    def decrement_sp(self, delta: int = 1):
        self.sp -= delta

    @debug
    def increment_sp(self, delta: int = 1):
        self.sp += delta

    @debug
    def push(self, val):
        self.stack.insert(self.sp + 1, val)
        self.increment_sp()

    @debug
    def pop(self):
        """Removes and returns the element int the stack at the
        stack pointer location and decrements it *unless* sp is the
        bottom of a non-empty stack in which case it does not change."""
        if not len(self.stack):
            raise VMStackException("Cannot pop an empty stack!")

        el = self._stack_at_sp()
        del(self.stack[self.sp])

        do_not_decrement = self.sp == self.get_init_sp() + 1 and len(self.stack)
        if do_not_decrement:
            pass
        else:
            self.decrement_sp()

        return el

    @debug
    def swap(self):
        if len(self.stack) < 2:
            raise VMStackException("Cannot swap less than two elements.")

        min_sp = self.get_init_sp() + 2
        if self.sp < min_sp:
            raise VMStackException("Stack pointer needs to be at least " + str(min_sp))

        cache_current_el = self.stack[self.sp]
        cache_previous_el = self.stack[self.sp - 1]
        self.stack[self.sp] = cache_previous_el
        self.stack[self.sp - 1] = cache_current_el

    @debug
    def dup(self):
        if len(self.stack) < 1:
            raise VMStackException("Cannot duplicate on an empty stack")

        self.stack.insert(self.sp, self.stack[self.sp])
        self.increment_sp()

    @debug
    def dig(self, jump = None):
        tmp = self.stack[self.sp]
        del(self.stack[self.sp])
        if jump is None:
            self.stack += [tmp]
            self.sp = len(self.stack) - 1
        else:
            self.stack.insert(self.sp + jump, tmp)
            self.sp += jump


class TestVM(unittest.TestCase):
    def test_store_lambda(self):
        vm = VM()
        body = [
            Instr("DUP", [], {}),
            Instr("PUSH", [2], {}),
            Instr("ADD", [], {}),
        ]
        arg = 2
        arg_types = ([int],)
        return_type = int
        vm._run_instructions(
            [
                Instr("LAMBDA", [arg_types, return_type, body], {}),
                Instr("PUSH", [arg], {}),
                Instr("EXEC", [], {}),
            ]
        )
        assert [body, arg, arg + 2] == vm.stack

    def test_run_lambda(self):
        vm = VM()
        body = [
            Instr('DUP', [], {}),
            Instr('PUSH', [2], {}),
            Instr('ADD', [], {}),
        ]
        arg = 2
        vm.push(body)
        vm.push(arg)
        vm._run_instructions([Instr('EXEC', [], {})])
        assert [body, arg, arg + 2] == vm.stack

    def test_run_instructions(self):
        vm = VM()
        instructions = [
            Instr('PUSH', [1], {}),
            Instr('PUSH', [2], {}),
            Instr('SWAP', [], {}),
        ]
        vm._run_instructions(instructions)
        self.assertEqual(vm.stack, [2, 1])

    def test_check_pair(self):
        vm = VM()
        vm._reset_stack()

        vm.push(1)
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
        vm.push(1)
        vm.push(2)

        self.assertEqual(vm._stack_top(), b)

    def test_stack_at_sp(self):
        vm = VM()
        a, b = 1, 2
        vm.push(a)
        vm.push(b)
        vm.decrement_sp()
        self.assertEqual(vm._stack_at_sp(), a)

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
        vm.push(a)
        vm.push(b)
        # stack grows towards larger addresses
        self.assertEqual(vm.stack, [a, b])

        vm.decrement_sp()
        vm.push(c)
        self.assertEqual(vm.stack, [a, c, b])

    def test_pop(self):
        vm = VM()
        a, b, c = 1, 2, 3

        # check that poping empty stack raises
        self.assertRaises(VMStackException, vm.pop)

        ### check that we can pop on top of the stack
        vm.push(a)
        vm.push(b)
        vm.pop()
        self.assertEqual(vm.stack, [a])
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)

        ### check that we can pop inside the stack

        # case 1: sp = 0 but stack is not empty => we keep sp untouched
        vm.push(c)
        vm.decrement_sp()
        vm.pop()
        self.assertEqual(vm.stack, [c])
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)

        # case 1: sp != 0 => we keep decrement sp
        vm.push(b)
        vm.push(a)
        vm.decrement_sp()
        vm.pop()
        self.assertEqual(vm.stack, [c, a])
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)

    def test_make_list(self):
        vm = VM()

        a, b = 1, 2
        vm.make_list()
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)
        self.assertEqual(vm.stack[-1].__class__, Array([]).__class__)

    def test_make_pair(self):
        vm = VM()

        a, b = 1, 2
        vm.make_pair(a, b)
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)
        ph = 42  # placeholder, just need the pair instance class
        self.assertEqual(vm.stack[-1].__class__, Pair(ph, ph).__class__)

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
        cdr = vm.cdr()
        self.assertEqual(vm._stack_top(), b)

    def test_swap(self):
        vm = VM()
        a, b, c = 1, 2, 3

        # Raises if stack too small
        self.assertRaises(VMStackException, vm.swap)
        vm.push(a)
        self.assertRaises(VMStackException, vm.swap)

        # swaps top of stack
        vm.push(b)
        vm.push(c)
        vm.swap()
        self.assertEqual(vm.stack, [1, 3, 2])

        # swaps inside stack
        vm._reset_stack()
        vm.push(a)
        vm.push(b)
        vm.push(c)
        vm.decrement_sp()
        vm.swap()
        self.assertEqual(vm.stack, [2, 1, 3])

        # Raises if sp < 2
        vm.decrement_sp()
        vm.decrement_sp()
        vm.decrement_sp()
        self.assertRaises(VMStackException, vm.swap)

    def test_dup(self):
        vm = VM()

        self.assertRaises(VMStackException, vm.dup)

        a, b, c = 1, 2, 3
        vm.push(a)
        vm.push(b)
        vm.push(c)
        vm.decrement_sp()
        vm.dup()
        self.assertEqual(vm.stack, [a, b, b, c])
        self.assertEqual(vm.sp, vm.get_init_sp() + 3)

    def test_add(self):
        vm = VM()

        a, b, c, d = 1, 2, 3, 4
        vm.push(a)
        vm.push(b)
        vm.push(c)
        vm.push(d)
        vm.add()
        self.assertEqual(vm.stack, [a, b, c + d])
        vm.decrement_sp()
        vm.add()
        self.assertEqual(vm.stack, [a + b, c + d])
        vm.add()
        self.assertEqual(vm.stack, [a + b + c + d])

    def test_dig(self):
        vm = VM()

        a, b, c = 1, 2, 3
        vm.push(a)
        vm.push(b)
        vm.push(c)
        vm.decrement_sp()
        vm.decrement_sp()
        vm.dig()
        self.assertEqual(vm.stack, [b, c, a])

    def test_append_before_list(self):
        vm = VM()

        a, b = 1, 2
        vm.make_list()
        vm.push(a)
        vm.append_before_list()
        vm.push(b)
        vm.append_before_list()
        self.assertEqual(vm.stack[0].els, [b, a])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVM)
    unittest.TextTestRunner().run(suite)
