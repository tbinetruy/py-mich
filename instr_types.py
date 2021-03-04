import ast
from typing import Optional


class Type:
    def __init__(self, annotation: Optional[str] = None):
        self.annotation = annotation
        pass

    def __eq__(self, o):
        return type(self) == type(o)


class Unit(Type):
    pass


class Int(Type):
    def __str__(self):
        return "int"


class String(Type):
    def __str__(self):
        return "string"


class Address(Type):
    def __str__(self):
        return "address"


class Operation(Type):
    def __str__(self):
        return "operation"


class List(Type):
    def __init__(self, element_type: Type, annotation: Optional[str] = None):
        super().__init__(annotation)
        self.element_type = element_type


class Dict(Type):
    def __init__(self, key_type: Type, value_type: Type, annotation: Optional[str] = None):
        super().__init__(annotation)
        self.key_type = key_type
        self.value_type = value_type


class TypeParser:
    def __init__(self):
        pass

    def parse_name(self, name: ast.Name, e, annotation: Optional[str] = None) -> Type:
        if name.id == 'int':
            return Int(annotation)
        if name.id == 'str':
            return String(annotation)
        if name.id == 'address':
            return Address(annotation)
        if name.id in e.records.keys():
            return e.records[name.id].get_type()
        raise NotImplementedError

    def parse(self, type_ast, e, annotation: Optional[str] = None) -> Type:
        if type(type_ast) == ast.Name:
            return self.parse_name(type_ast, e, annotation)
        raise NotImplementedError
