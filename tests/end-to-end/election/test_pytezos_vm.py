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
    def test_open(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        new_storage = vm.contract.open({"_open": "foo", "manifest_url": "bar", "manifest_hash": "baz"}).interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['_open'], "foo")
        self.assertEqual(new_storage['manifest_url'], "bar")
        self.assertEqual(new_storage['manifest_hash'], "baz")

        try:
            new_storage = vm.contract.open({"_open": "foo", "manifest_url": "bar", "manifest_hash": "baz"}).interpret(storage=init_storage)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only admin can call this entrypoint'")

    def test_close(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        new_storage = vm.contract.close("foo").interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['_close'], "foo")

        try:
            new_storage = vm.contract.close("foo").interpret(storage=init_storage)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only admin can call this entrypoint'")

    def test_artifacts(self):
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)

        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        new_storage = vm.contract.artifacts({"artifacts_url": "url", "artifacts_hash": "hash"}).interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['artifacts_url'], "url")
        self.assertEqual(new_storage['artifacts_hash'], "hash")

        try:
            new_storage = vm.contract.close("foo").interpret(storage=init_storage)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only admin can call this entrypoint'")

if __name__ == "__main__":
    unittest.main()
