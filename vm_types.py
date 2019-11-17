from dataclasses import dataclass
from typing import NewType, Dict, List, Tuple, Callable

@dataclass
class Pair:
    '''Michelson pair.

    Parameters
    ----------
    car: first element
    cdr: second element'''
    car: any
    cdr: any

Addr, VarName = NewType('Address', int), NewType('VarName', str)
@dataclass
class Env:
    vars: NewType('Environment', Dict[VarName, Addr])
    sp: int

    def copy(self):
        return Env(self.vars.copy(), self.sp)


@dataclass
class Instr:
    name: str
    args: List[any]
    kwargs: Dict[str, any]
