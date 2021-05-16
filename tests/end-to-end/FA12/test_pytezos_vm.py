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

        storage = vm.contract.storage.dummy()
        storage['owner'] = vm.context.sender

        storage = vm.contract.mint({"_to": vm.context.sender, "value": 10}).interpret(storage=storage, sender=vm.context.sender).storage
        self.assertEqual(storage['tokens'], {vm.context.sender: 10})

        try:
            vm.contract.mint({"_to": vm.context.sender, "value": 10}).interpret(storage=storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only owner can mint'")

    def test_getAllowance(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        storage = vm.contract.storage.dummy()
        storage['owner'] = vm.context.sender

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        amount = 10
        initial_storage = {"owner": vm.context.sender, "total_supply": amount, "tokens": {}, "allowances": {(vm.context.sender, investor): amount}}
        res = vm.contract.getAllowance({"owner": vm.context.sender, "spender": investor, "contract_2": None}).callback_view(storage= initial_storage)
        self.assertEqual(res, 10)

        res = vm.contract.getAllowance({"owner": vm.context.sender, "spender": investor, "contract_2": None}).callback_view()
        self.assertEqual(res, 0)

    def test_getBalance(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        storage = vm.contract.storage.dummy()
        storage['owner'] = vm.context.sender

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        amount = 10
        initial_storage = {"owner": vm.context.sender, "total_supply": amount, "tokens": {investor: amount}, "allowances": {}}
        res = vm.contract.getBalance({"owner": investor, "contract_1": None}).callback_view(storage= initial_storage)
        self.assertEqual(res, 10)

        res = vm.contract.getBalance({"owner": vm.context.sender, "contract_1": None}).callback_view()
        self.assertEqual(res, 0)

    def test_getTotalSupply(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        self.assertEqual(vm.contract.getTotalSupply().callback_view(), 0)

    def test_transfer(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        storage = vm.contract.storage.dummy()
        storage['owner'] = vm.context.sender
        storage['tokens'] = {vm.context.sender: 10}

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        storage = vm.contract.transfer({"_to": investor, "_from": vm.context.sender, "value": 4}).interpret(storage=storage, sender=vm.context.sender).storage
        self.assertEqual(storage['tokens'], {vm.context.sender: 6, investor: 4})

        try:
            vm.contract.transfer({"_from": vm.context.sender, "_to": investor, "value": 10}).interpret(storage=storage, sender=vm.context.sender).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'NotEnoughBalance'")

        try:
            vm.contract.transfer({"_from": vm.context.sender, "_to": investor, "value": 10}).interpret(storage=storage, sender=investor).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'NotEnoughAllowance'")

        storage["allowances"] = {(vm.context.sender, investor): 10}
        storage = vm.contract.transfer({"_from": vm.context.sender, "_to": investor, "value": 2}).interpret(storage=storage, sender=investor).storage
        assert storage["tokens"][investor] == 6

        try:
            vm.contract.transfer({"_from": vm.context.sender, "_to": investor, "value": 8}).interpret(storage=storage, sender=investor).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'NotEnoughBalance'")

    def test_approve(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        storage = vm.contract.storage.dummy()
        storage['owner'] = vm.context.sender

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        storage = vm.contract.approve({"spender": investor, "value": 4}).interpret(storage=storage, sender=vm.context.sender).storage
        self.assertEqual(storage['allowances'], {(vm.context.sender, investor): 4})

if __name__ == "__main__":
    unittest.main()
