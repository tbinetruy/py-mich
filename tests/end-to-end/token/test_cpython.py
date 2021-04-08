from contract import Contract

admin = "Mrs. Foo"
contract = Contract(admin=admin, sender=admin)

amount_1 = 10
contract.mint(admin, amount_1)
assert contract.storage.balances[admin] == amount_1

investor = "Mr. Bar"
amount_2 = 4
contract.transfer(investor, amount_2)
assert contract.storage.balances[admin] == amount_1 - amount_2

assert contract.storage.balances[investor] == amount_2
