import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
pymich_dir = path.dirname(path.dirname(path.dirname(current_dir)))
sys.path.append(pymich_dir)

#####

import unittest
from compiler import Compiler
from compiler import VM
from pytezos.michelson.micheline import MichelsonRuntimeError

with open("contract.py") as f:
    source = f.read()


class TestContract(unittest.TestCase):
    def test_mint(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        new_storage = vm.contract.mint({"to": vm.context.sender, "amount": 10}).interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['balances'], {vm.context.sender: 10})

        try:
            vm.contract.mint({"to": vm.context.sender, "amount": 10}).interpret(storage=init_storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only admin can mint'")

    def test_transfer(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender
        init_storage['balances'] = {vm.context.sender: 10}

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        new_storage = vm.contract.transfer({"to": investor, "amount": 4}).interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['balances'], {vm.context.sender: 6, investor: 4})

        try:
            vm.contract.transfer({"to": investor, "amount": -10}).interpret(storage=new_storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'You need to transfer a positive amount of tokens'")

        try:
            vm.contract.transfer({"to": investor, "amount": 10}).interpret(storage=new_storage, sender=vm.context.sender).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Insufficient sender balance'")

if __name__ == "__main__":
    unittest.main()
