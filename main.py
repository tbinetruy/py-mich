import json

from compiler import Compiler
from vm import VM
from compiler_backend import CompilerBackend


source = """
@dataclass
class Storage:
    balances: Dict[address, int]
    total_supply: int
    admin: address

@dataclass
class MintParam:
    to: address
    amount: int

@dataclass
class TransferParam:
    to: address
    amount: int

def require(arg: bool) -> int:
    _ = 0
    if arg:
        _ = 0
    else:
        raise "Error"

    return 0

class Contract:
    def deploy():
        return Storage({}, 0, "tz1VSUr8wwNhLAzempoch5d6hLRiTh8Cjcjb")

    def mint(param: MintParam) -> Storage:
        _ = require(self.sender == self.storage.admin)

        self.storage.total_supply = self.storage.total_supply + param.amount

        balances = self.storage.balances

        if param.to in balances:
            balances[param.to] = balances[param.to] + param.amount
        else:
            balances[param.to] = param.amount

        self.storage.balances = balances

        return self.storage

    def transfer(param: TransferParam) -> Storage:
        _ = require(self.sender == self.storage.admin)
        _ = require(param.amount > 0)

        balances = self.storage.balances

        sender_balance = balances[self.sender]
        _ = require(sender_balance >= param.amount)

        balances[self.sender] = sender_balance - param.amount

        if param.to in balances:
            balances[param.to] = balances[param.to] + param.amount
        else:
            balances[param.to] = param.amount

        self.storage.balances = balances

        return self.storage
"""

vm = VM(isDebug=False)
c = Compiler(source, isDebug=False)
instructions = c._compile(c.ast)

micheline = CompilerBackend().compile_contract(c.contract)

with open("my_contract.json", "w+") as f:
    f.write(json.dumps(micheline))
