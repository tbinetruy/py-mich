import unittest
from pytezos.michelson.micheline import MichelsonRuntimeError
import stubs
admin = "Mrs. Foo"
stubs.SENDER = admin

from contract import Contract


class TestContract(unittest.TestCase):
    def test_open(self):
        from contract import Contract
        contract = Contract(stubs.SENDER, "", "", "", "", "", "")
        contract.open("foo", "bar", "baz")

        assert contract._open == "foo"
        assert contract.manifest_url == "bar"
        assert contract.manifest_hash == "baz"

        contract = Contract("yolo", "", "", "", "", "", "")
        try:
            contract.open("foo", "bar", "baz")
            assert 0
        except Exception as e:
            assert e.args[0] == 'Only admin can call this entrypoint'

    def test_close(self):
        from contract import Contract
        contract = Contract(stubs.SENDER, "", "", "", "", "", "")
        contract.close("foo")

        assert contract._close == "foo"

        contract = Contract("yolo", "", "", "", "", "", "")
        try:
            contract.close("foo")
            assert 0
        except Exception as e:
            assert e.args[0] == 'Only admin can call this entrypoint'

    def test_artifacts(self):
        from contract import Contract
        contract = Contract(stubs.SENDER, "", "", "", "", "", "")
        contract.artifacts("url", "hash")

        assert contract.artifacts_url == "url"
        assert contract.artifacts_hash == "hash"

        contract = Contract("yolo", "", "", "", "", "", "")
        try:
            contract.artifacts("url", "hash")
            assert 0
        except Exception as e:
            assert e.args[0] == 'Only admin can call this entrypoint'

if __name__ == "__main__":
    unittest.main()
