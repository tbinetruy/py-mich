import sys
from os import path

current_dir = path.dirname(path.abspath(__file__))
pymich_dir = path.dirname(path.dirname(path.dirname(current_dir)))
sys.path.append(pymich_dir)

from compiler import Compiler
from compiler_backend import CompilerBackend

with open("contract.py") as f:
    source = f.read()

compiler = Compiler(source)
compiler.compile()
micheline = CompilerBackend().compile_contract(compiler.contract)

## run in pytezos vm
from pytezos import ContractInterface
ci = ContractInterface.from_micheline(micheline)

admin = "tz1VSUr8wwNhLAzempoch5d6hLRiTh8Cjcjb"
init_storage = {
    "admin": admin,
    "total_supply": 0,
    "balances": {},
}
amount_1 = 10
res = ci.mint({"to": admin, "amount": amount_1}).interpret(storage=init_storage, sender=admin)
assert res.storage['balances'][admin] == amount_1
assert res.storage['total_supply'] == amount_1

investor = "KT1TQLYApWLn8M6XNDAJ5YyoSsFoQauxygCa"
amount_2 = 4
res = ci.transfer({"to": investor, "amount": amount_2}).interpret(storage=res.storage, sender=admin)
assert res.storage['balances'][admin] == amount_1 - amount_2
assert res.storage['balances'][investor] == amount_2
