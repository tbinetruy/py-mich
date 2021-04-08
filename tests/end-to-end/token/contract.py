from dataclasses import dataclass
from typing import Dict

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

    def mint(self, to: address, amount: int):
        _ = require(self.sender == self.storage.admin)

        self.storage.total_supply = self.storage.total_supply + amount

        if to in self.storage.balances:
            self.storage.balances[to] = self.storage.balances[to] + amount
        else:
            self.storage.balances[to] = amount

    def transfer(self, to: address, amount: int):
        _ = require(
            self.sender == self.storage.admin
            and amount > 0
            and self.storage.balances[self.sender] >= amount
        )

        self.storage.balances[self.sender] = self.storage.balances[self.sender] - amount

        if to in self.storage.balances:
            self.storage.balances[to] = self.storage.balances[to] + amount
        else:
            self.storage.balances[to] = amount
