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

        balances = self.balances
        if to in balances:
            balances[to] = balances[to] + amount
        else:
            balances[to] = amount
        self.balances = balances

    def transfer(self, to: address, amount: int):
        balances = self.balances

        require(amount > 0, "You need to transfer a positive amount of tokens")
        require(balances[SENDER] >= amount, "Insufficient sender balance")

        balances[SENDER] = balances[SENDER] - amount

        if to in balances:
            balances[to] = balances[to] + amount
        else:
            balances[to] = amount

        self.balances = balances

