from dataclasses import dataclass
from typing import NewType, Dict, List, Tuple, Callable, Any


@dataclass
class Array:
    """Michelson list. Called array not to conflict with `typing.List`
    used in type annotations.

    Parameters
    ----------
    els: python list representing the Michelson list"""
    els: List[Any]

@dataclass
class Pair:
    '''Michelson pair.

    Parameters
    ----------
    car: first element
    cdr: second element'''
    car: Any
    cdr: Any

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
    args: List[Any]
    kwargs: Dict[str, Any]
