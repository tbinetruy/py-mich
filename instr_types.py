import ast
from typing import Optional


class Type:
    def __init__(self, annotation: Optional[str] = None):
        self.annotation = annotation
        pass

    def __eq__(self, o):
        return type(self) == type(o)


class Unit(Type):
    def __str__(self):
        return "unit"


class Int(Type):
    def __str__(self):
        return "int"


class String(Type):
    def __str__(self):
        return "string"


class Bool(Type):
    def __str__(self):
        return "bool"


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
        if name.id == 'bool':
            return Bool(annotation)
        if name.id == 'unit':
            return Unit(annotation)
        if name.id in e.records.keys():
            return e.records[name.id].get_type()
        raise NotImplementedError

    def parse_dict(self, dictionary: ast.Subscript, e, annotation):
        key_type = self.parse(dictionary.slice.value.elts[0], e, annotation)
        value_type = self.parse(dictionary.slice.value.elts[1], e, annotation)
        return Dict(key_type, value_type)

    def parse_subscript(self, subscript: ast.Dict, e, annotation):
        if subscript.value.id == "Dict":
            return self.parse_dict(subscript, e, annotation)

    def parse(self, type_ast, e, annotation: Optional[str] = None) -> Type:
        if type(type_ast) == ast.Name:
            return self.parse_name(type_ast, e, annotation)
        if type(type_ast) == ast.Subscript:
            return self.parse_subscript(type_ast, e, annotation)
        raise NotImplementedError
