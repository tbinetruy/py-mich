import ast


class Type:
    def __init__(self):
        pass


class Unit(Type):
    def __init__(self):
        pass

    def __eq__(self, o):
        return type(self) == type(o)


class Int(Type):
    def __init__(self):
        pass

    def __eq__(self, o):
        return type(self) == type(o)

    def __str__(self):
        return "int"


class String(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "string"


class Operation(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "operation"


class List(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type


class Dict(Type):
    def __init__(self, key_type: Type, value_type: Type):
        self.key_type = key_type
        self.value_type = value_type


class TypeParser:
    def __init__(self):
        pass

    def parse_name(self, name: ast.Name) -> Type:
        if name.id == 'int':
            return Int()
        raise NotImplementedError

    def parse(self, type_ast) -> Type:
        if type(type_ast) == ast.Name:
            return self.parse_name(type_ast)
        raise NotImplementedError
