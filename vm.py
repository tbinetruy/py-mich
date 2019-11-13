import unittest

from vm_types import Pair


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

    def _stack_at_sp(self):
        return self.stack[self.sp] \
            if self.sp >= 0 or self.sp >= len(self.stack) \
            else self.get_init_sp()

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
        self.stack.insert(self.sp + 1, val)
        self.increment_sp()
        self._debug()

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

        self._debug()
        return el

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

suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVM)
unittest.TextTestRunner().run(suite)
