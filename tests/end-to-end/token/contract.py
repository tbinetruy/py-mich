from dataclasses import dataclass
from typing import Dict
from stubs import *


def require(condition: bool, message: str) -> int:
    if not condition:
        raise Exception(message)

    return 0


@dataclass
class Contract:
    balances: Dict[address, int]
    total_supply: int
    admin: address

    def mint(self, to: address, amount: int):
        require(SENDER == self.admin, "Only admin can mint")

        self.total_supply = self.total_supply + amount

        if to in self.balances:
            self.balances[to] = self.balances[to] + amount
        else:
            self.balances[to] = amount

    def transfer(self, to: address, amount: int):
        require(amount > 0, "You need to transfer a positive amount of tokens")
        require(self.balances[SENDER] >= amount, "Insufficient sender balance")

        self.balances[SENDER] = self.balances[SENDER] - amount

        if to in self.balances:
            self.balances[to] = self.balances[to] + amount
        else:
            self.balances[to] = amount

