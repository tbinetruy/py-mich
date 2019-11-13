import unittest

from vm_types import Pair


ph = 42  # placeholder

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


class VM:
    @staticmethod
    def get_init_stack():
        return []

    @staticmethod
    def get_init_sp():
        return -1

    def __init__(self, isDebug=True):
        self.isDebug = isDebug

        self._reset_stack()

    def _reset_stack(self):
        self.stack = self.get_init_stack()
        self.sp = self.get_init_sp()
        self._debug()

    def _debug(self):
        if self.isDebug:
            stack = [
                ("*", el)
                if i == self.sp
                else el
                for i, el in enumerate(self.stack)
            ]
            print("S:", stack, "sp:", self.sp)

    def _stack_top(self):
        return self.stack[-1]

    def _check_pair(self):
        expected_t = self._stack_top().__class__
        actual_t = Pair(ph, ph).__class__
        if expected_t != actual_t:
            raise VMTypeException(expected_t, actual_t, "Car requires a pair")

    def car(self):
        self._check_pair()
        pair = self.pop()
        self.push(pair.car)

    def cdr(self):
        self._check_pair()
        pair = self.pop()
        self.push(pair.cdr)

    def make_pair(self, car, cdr):
        self.push(Pair(car, cdr))

    def decrement_sp(self):
        self.sp -= 1

    def increment_sp(self):
        self.sp += 1

    def push(self, val):
        self.stack.append(val)
        self.increment_sp()
        self._debug()

    def pop(self):
        el = self.stack.pop()
        self.decrement_sp()
        self._debug()
        return el

class TestVM(unittest.TestCase):
    def test_check_pair(self):
        vm = VM()
        vm._reset_stack()

        self.assertRaises(VMTypeException, vm._check_pair)
        vm.make_pair(1, 2)
        try:
            self.assertEqual(VMTypeException, vm._check_pair)
        except VMTypeException:
            self.fail("check_pair raised VMTypeException unexpectedly!")


    def test_check_pair(self):
        vm = VM()
        vm._reset_stack()

        b = 2
        vm.push(1)
        vm.push(2)

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

        vm.push(1)
        vm.push(2)
        # stack grows towards larger addresses
        self.assertEqual(vm.stack, [1, 2])

    def test_pop(self):
        vm = VM()

        vm.push(1)
        vm.push(2)
        vm.pop()
        self.assertEqual(vm.stack, [1])
        self.assertEqual(vm.sp, VM.get_init_sp() + 1)

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

suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVM)
unittest.TextTestRunner().run(suite)
