import unittest
from pytezos.michelson.micheline import MichelsonRuntimeError
import stubs
admin = "Mrs. Foo"
stubs.SENDER = admin

from contract import Contract


class TestContract(unittest.TestCase):
    def test_mint(self):
        from contract import Contract
        contract = Contract(admin=admin, balances={}, total_supply=0)
        amount = 10
        contract.mint(admin, amount)

        assert contract.balances[admin] == amount

        contract = Contract(admin="yolo", balances={}, total_supply=0)
        try:
            contract.mint(admin, amount)
            assert 0
        except Exception as e:
            assert e.args[0] == 'Only admin can mint'

    def test_transfer(self):
        amount_1 = 10
        contract = Contract(admin=admin, balances={admin: amount_1}, total_supply=amount_1)

        investor = "Mr. Bar"
        amount_2 = 4

        contract.transfer(investor, amount_2)

        assert contract.balances[admin] == amount_1 - amount_2
        assert contract.balances[investor] == amount_2

        try:
            contract.transfer(admin, -10)
            assert 0
        except Exception as e:
            assert e.args[0] == 'You need to transfer a positive amount of tokens'

        try:
            contract.transfer(admin, 100)
            assert 0
        except Exception as e:
            assert e.args[0] == 'Insufficient sender balance'


if __name__ == "__main__":
    unittest.main()
