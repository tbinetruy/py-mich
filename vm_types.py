from dataclasses import dataclass

@dataclass
class Pair:
    '''Michelson pair.

    Parameters
    ----------
    car: first element
    cdr: second element'''
    car: any
    cdr: any
