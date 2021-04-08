from typing import Dict
from dataclasses import dataclass
address = str


@dataclass
class Storage:
    balances: Dict[address, int]
    total_supply: int
    admin: address


def require(condition: bool) -> int:
    if not condition:
        raise Exception("Error")

    return 0


class Contract:
    def __init__(self, admin, sender):
        self.storage = Storage({}, 0, admin)
        self.sender = sender

    def mint(self, to: address, amount: int) -> Storage:
        _ = require(self.sender == self.storage.admin)

        self.storage.total_supply = self.storage.total_supply + amount

        balances = self.storage.balances

        if to in balances:
            balances[to] = balances[to] + amount
        else:
            balances[to] = amount

        self.storage.balances = balances

        return self.storage

    def transfer(self, to: address, amount: int) -> Storage:
        _ = require(self.sender == self.storage.admin and amount > 0)

        balances = self.storage.balances

        sender_balance = balances[self.sender]
        _ = require(sender_balance >= amount)

        balances[self.sender] = sender_balance - amount

        if to in balances:
            balances[to] = balances[to] + amount
        else:
            balances[to] = amount

        self.storage.balances = balances

        return self.storage
