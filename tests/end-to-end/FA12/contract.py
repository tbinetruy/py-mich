from dataclasses import dataclass
from typing import Dict
from stubs import *

@dataclass
class AllowanceKey:
    owner: address
    spender: address

@dataclass
class Contract:
    tokens: Dict[address, int]
    allowances: Dict[AllowanceKey, int]
    total_supply: int
    owner: address

    def mint(self, _to: address, value: int):
        if SENDER != self.owner:
            raise Exception("Only owner can mint")

        self.total_supply = self.total_supply + value

        if _to in self.tokens:
            self.tokens[_to] = self.tokens[_to] + value
        else:
            self.tokens[_to] = value

    def approve(self, spender: address, value: int):
        allowance_key = AllowanceKey(SENDER, spender)

        previous_value = self.allowances.get(allowance_key, 0)

        if previous_value > 0 and value > 0:
            raise Exception("UnsafeAllowanceChange")

        self.allowances[allowance_key] = value

    def transfer(self, _from: address, _to: address, value: int):
        if SENDER != _from:
            allowance_key = AllowanceKey(_from, SENDER)

            authorized_value = self.allowances.get(allowance_key, 0)

            if (authorized_value - value) < 0:
                raise Exception("NotEnoughAllowance")

            self.allowances[allowance_key] = authorized_value - value

        from_balance = self.tokens.get(_from, 0)

        if from_balance - value < 0:
            raise Exception("NotEnoughBalance")

        self.tokens[_from] = from_balance - value

        to_balance = self.tokens.get(_to, 0)

        self.tokens[_to] = to_balance + value
