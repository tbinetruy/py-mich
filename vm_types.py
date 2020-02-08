from dataclasses import dataclass
from typing import NewType, Dict, List, Tuple, Callable


@dataclass
class Array:
    """Michelson list. Called array not to conflict with `typing.List`
    used in type annotations.

    Parameters
    ----------
    els: python list representing the Michelson list"""
    els: List[any]

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
    args: NewType('Args', Dict[VarName, VarName])

    def copy(self):
        return Env(self.vars.copy(), self.sp, self.args.copy())


@dataclass
class Instr:
    name: str
    args: List[any]
    kwargs: Dict[str, any]
