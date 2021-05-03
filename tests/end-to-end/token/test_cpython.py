import unittest
from pytezos.michelson.micheline import MichelsonRuntimeError
import stubs
admin = "tz3M4KAnKF2dCSjqfa1LdweNxBGQRqzvPL88"
investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
stubs.SENDER = admin

from contract import Contract, AllowanceKey


class TestContract(unittest.TestCase):
    def test_mint(self):
        from contract import Contract
        contract = Contract(owner=admin, tokens={}, allowances={}, total_supply=0)
        amount = 10
        contract.mint(admin, amount)

        assert contract.tokens[admin] == amount

        contract = Contract(owner=investor, tokens={}, allowances={}, total_supply=0)
        try:
            contract.mint(admin, amount)
            assert 0
        except Exception as e:
            assert e.args[0] == 'Only owner can mint'

    def test_transfer(self):
        amount_1 = 10
        contract = Contract(owner=admin, tokens={admin: amount_1}, allowances={}, total_supply=amount_1)

        investor = "Mr. Bar"
        amount_2 = 4

        contract.transfer(admin, investor, amount_2)

        assert contract.tokens[admin] == amount_1 - amount_2
        assert contract.tokens[investor] == amount_2

        try:
            contract.transfer(admin, investor, 100)
            assert 0
        except Exception as e:
            assert e.args[0] == 'NotEnoughBalance'

    def todo_test_approve(self):
        # make dataclasses hashable by default by ignoring `unsafe_hash` in dataclass param in compiler)
        amount = 10
        contract = Contract(owner=admin, tokens={}, allowances={}, total_supply=0)
        contract.approve(investor, amount)
        assert contract.allowances[AllowanceKey(admin, investor)] == amount

if __name__ == "__main__":
    unittest.main()
