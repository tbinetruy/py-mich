import ast
import pprint
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional

from pytezos.context.impl import ExecutionContext
from pytezos.michelson.instructions.adt import *
from pytezos.michelson.instructions.arithmetic import *
from pytezos.michelson.instructions.control import *
from pytezos.michelson.instructions.stack import *
from pytezos.michelson.instructions.struct import *
from pytezos.michelson.micheline import MichelsonRuntimeError
from pytezos.michelson.program import *
from pytezos.michelson.repl import InterpreterResult
from pytezos.michelson.sections import CodeSection
from pytezos.michelson.sections.parameter import ParameterSection
from pytezos.michelson.sections.storage import StorageSection
from pytezos.michelson.stack import MichelsonStack
from pytezos.michelson.types import core
from pytezos.michelson.types.core import *
from pytezos.michelson.types.domain import AddressType
from pytezos.michelson.types.map import MapType
from pytezos.michelson.types.operation import *
from pytezos.michelson.types.pair import PairType
from pytezos import ContractInterface

import instr_types as t
from compiler_backend import CompilerBackend
from macro_expander import macro_expander
from helpers import Tree, ast_to_tree
from vm_types import (Array, Contract, Entrypoint, FunctionPrototype, Instr,
                      Pair, Some)


class CompilerError(Exception):
    """Raised when the compiler fails

    Attributes:
        message -- error message
    """

    def __init__(self, message):
        self.message = message


def debug(cb):
    def f(*args, **kwargs):
        self = args[0]
        if self.isDebug:
            print(cb.__name__)

        return cb(*args, **kwargs)

    return f


def Comment(msg: str):
    return Instr("COMMENT", [msg], {})


class Record(Tree):
    def __init__(self, attribute_names, attribute_types, attribute_annotations):
        self.attribute_names = attribute_names
        self.attribute_types = attribute_types
        self.attribute_annotations = attribute_annotations

    def make_node(self, left, right):
        return Pair(car=left, cdr=right)

    def get_type(self, i=0, acc=None):
        attribute_type = self.attribute_types[i]
        if acc == None:
            acc = self.make_node(None, None)

        if i == 0:
            return self.make_node(attribute_type, self.get_type(i+1))

        elif i == len(self.attribute_types) - 1:
            return attribute_type

        else:
            return self.make_node(attribute_type, self.get_type(i+1))

    def get_attribute_type(self, attribute_name: str):
        attribute_index = self.attribute_names.index(attribute_name)
        return self.attribute_types[attribute_index]

    def get_attribute_annotation(self, attribute_name: str):
        attribute_index = self.attribute_names.index(attribute_name)
        return self.attribute_annotations[attribute_index]

    def navigate_to_tree_leaf(self, attribute_name, acc=None):
        el_number = 1
        for i, candidate in enumerate(self.attribute_names):
            if i == len(self.attribute_names) - 1:
                el_number = 2 * i
            elif attribute_name == candidate:
                el_number = 2 * i + 1
                break

        return [Instr("GET", [t.Int(), el_number], {})]

    def update_tree_leaf(self, attribute_name, e):
        el_number = 1
        for i, candidate in enumerate(self.attribute_names):
            if i == len(self.attribute_names) - 1:
                el_number = 2 * i
            elif attribute_name == candidate:
                el_number = 2 * i + 1
                break

        e.sp -= 1  # account for update
        return [Instr("UPDATE", [t.Int(), el_number], {})]

    def compile_record(self, attribute_values, compile_function, env):
        instructions = []
        for attribute_value, attribute_type in zip(reversed(attribute_values), reversed(self.attribute_types)):
            instructions += compile_function(attribute_value, env, current_type=attribute_type)
        env.sp -= len(attribute_values) - 1
        return instructions + [Instr("PAIR", [len(attribute_values)], {})]


@dataclass
class Env:
    vars: Dict[str, int]
    sp: int
    args: Dict[str, List[str]]
    records: Dict[str, Record]
    types: Dict[str, str]

    def copy(self):
        return Env(
            self.vars.copy(),
            self.sp,
            self.args.copy(),
            self.records.copy(),
            self.types.copy(),
        )


class Compiler:
    def __init__(self, src: str, isDebug=False):
        self.ast = ast.parse(src)
        self.isDebug = isDebug
        self.type_parser = t.TypeParser()
        self.contract = Contract(
            storage_type=t.Int(),
            storage=0,
            entrypoints={},
            instructions=[],
        )

    def print_ast(self):
        print(pprint.pformat(ast_to_tree(self.ast)))

    def compile_module(self, m: ast.Module, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        for key, value in ast.iter_fields(m):
            if key == "body":
                for childNode in value:
                    if type(childNode) == ast.ClassDef:
                        if childNode.name == "Contract":
                            instructions += self._compile(childNode, e, instructions)
                        else:
                            instructions += self._compile(childNode, e)
                    else:
                        instructions += self._compile(childNode, e)

        return instructions

    def _compile_reassign(self, reassign_ast: ast.Assign, e: Env) -> List[Instr]:
        instructions: List[Instr] = []
        var_name = reassign_ast.targets[0]
        value = reassign_ast.value
        var_addr = e.vars[var_name.id]
        instructions = self._compile(value, e)
        free_vars_instructions, _ = self.free_var(var_name.id, e)
        instructions = instructions + free_vars_instructions + [
            Instr("DUG", [e.sp - var_addr], {}),
        ]
        e.vars[var_name.id] = var_addr

        try:
            if reassign_ast.value.func.id in e.records:
                e.types[var_name.id] = reassign_ast.value.func.id
        except:
            pass

        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"Reassigning {var_name.id} = {print_val}")] + instructions

    @debug
    def compile_assign_storage_attribute(self, assign_ast: ast.Assign, e: Env) -> List[Instr]:
        new_ast = ast.Assign(
            targets=[
                ast.Attribute(value=ast.Name(id='__STORAGE__', ctx=ast.Load()), attr=assign_ast.targets[0].attr, ctx=ast.Store())
            ],
            value=assign_ast.value,
            type_comment=None,
        )
        return self.compile_assign(new_ast, e)

    def compile_dict(self, dict_ast: ast.Dict, key_type: t.Type, value_type: t.Type, e: Env) -> List[Instr]:
        e.sp += 1  # account for pushing dict
        return [Instr("EMPTY_MAP", [key_type, value_type], {})]

    def compile_literal(self, literal, e: Env) -> List[Instr]:
        if type(literal) == ast.Dict:
            return self.compile_dict(literal, e)
        else:
            return self.compile_expr(literal, e)

    def _is_literal(self, literal_ast):
        if type(literal_ast) == ast.Dict:
            return True
        else:
            return False

    @debug
    def compile_ann_assign(self, assign: ast.AnnAssign, e: Env) -> List[Instr]:
        try:
            # is reassignment
            if assign.targets[0].id in e.vars.keys():
                raise CompilerError("Cannot reassign with annotation")
        except:
            pass

        instructions: List[Instr] = []
        var_name = assign.target

        if self._is_literal(assign.value):
            compiled_value = self.compile_literal(assign.value, e)
        else:
            compiled_value = self._compile(assign.value, e)

        value = assign.value
        instructions = self._compile(var_name, e) + compiled_value
        e.vars[var_name.id] = e.sp

        try:
            if assign.value.func.id in e.records:
                e.types[var_name.id] = assign.value.func.id
        except:
            pass

        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"{var_name.id} = {print_val}")] + instructions


    def _get_assign_subscript_target_addr(self, node) -> str:
        if type(node) == ast.Name:
            return node.id
        if type(node) == ast.Attribute:
            current_node = node
            while 1:
                if type(current_node.value) == ast.Attribute:
                    current_node = current_node.value
                elif type(current_node.value) == ast.Name:
                    return node.value.id
                else:
                    raise NotImplementedError
        raise NotImplementedError

    def _fetch_variable(self, var_name: str, e: Env) -> List[Instr]:
        jump_length = e.sp - e.vars[var_name]
        e.sp += 1
        return [
            Instr("DIG", [jump_length], {}),
            Instr("DUP", [], {}),
            Instr("DUG", [jump_length + 1], {}),
        ]

    @debug
    def compile_assign_subscript(self, assign_subscript: ast.Assign, e: Env) -> List[Instr]:

        if type(assign_subscript.targets[0].value) == ast.Attribute:
            record_spec = self._get_record_spec(assign_subscript.targets[0].value, e)

            # fetch the dictionary
            #fetch_dict_bk = self._fetch_variable(record_spec["attribute_names"][0], e)
            dictionary = self._get_record_with_replace(record_spec, e)  + [Instr("DUP", [], {})] + record_spec["records"][-1].navigate_to_tree_leaf(record_spec["attribute_names"][-1])
            e.sp += 1  # account for DUP
            value = self._compile(assign_subscript.value, e)
            key = self._compile(assign_subscript.targets[0].slice.value, e)
            update_dictionary =  dictionary + value + [Instr("SOME", [], {})] +  key + [Instr("UPDATE", [], {})]
            e.sp -= 2  # account for update dropping the key and value from the stack

            # set the record
            set_dictionary = self._set_record(record_spec, e)

            var_name = record_spec["attribute_names"][0]
            return update_dictionary + set_dictionary + self._replace_var(var_name, e)

        else:
            dictionary = self._compile(assign_subscript.targets[0].value, e)
            value = self._compile(assign_subscript.value, e)
            key = self._compile(assign_subscript.targets[0].slice.value, e)
            e.sp -= 2  # account for update dropping the key and value from the stack
            #dictionary_name = assign_subscript.targets[0].value.id
            dictionary_name = self._get_assign_subscript_target_addr(assign_subscript.targets[0].value)

            update_dictionary = dictionary + value + [Instr("SOME", [], {})] +  key + [Instr("UPDATE", [], {})]
            dict_addr = e.vars[dictionary_name]
            free_old_dict, _ = self.free_var(dictionary_name, e)
            replace_old_dict = [Instr("DUG", [e.sp - dict_addr], {})]
            e.vars[dictionary_name] = dict_addr
            return update_dictionary + free_old_dict + replace_old_dict

    def _get_record_spec(self, node: ast.Attribute, e: Env):
        acc = []
        cond = True
        attribute_names = []
        current_node = node
        while cond:
            if type(current_node.value) == ast.Attribute:
                acc.append(current_node.attr)
                current_node = current_node.value
                pass
            elif type(current_node.value) == ast.Name:
                acc.append(current_node.attr)
                acc.append(current_node.value.id)
                cond = False
            else:
                cond = False
        acc.reverse()

        records = []
        for i, el in enumerate(acc[:-1]):
            if i == 0:
                record_type_name = e.types[el]
                records.append(e.records[record_type_name])
            else:
                index = None
                current_record = records[-1]
                for i, attribute_name in enumerate(current_record.attribute_names):
                    if attribute_name == el:
                        index = i
                nested_record_type = current_record.attribute_annotations[index].id
                records.append(e.records[nested_record_type])

        return {"attribute_names": acc, "records": records}

    def _get_record_with_replace(self, record_spec, e: Env) -> List[Instr]:
        attribute_names = record_spec["attribute_names"]
        records = record_spec["records"]

        instructions = self._compile(ast.Name(attribute_names[0], ctx=ast.Load()), e)

        for i in range(len(records[:-1])):
            instructions += [Instr("DUP", [], {})]
            e.sp += 1
            record = records[i]
            attr_name = attribute_names[i + 1]
            instructions += record.navigate_to_tree_leaf(attr_name)

        return instructions

    def _set_record(self, record_spec, e: Env) -> List[Instr]:
        attribute_names = record_spec["attribute_names"]
        records = record_spec["records"]
        instructions = []
        for i in range(len(records)):
            reversed_i = len(records) - 1 - i
            record = records[reversed_i]
            attr_name = attribute_names[reversed_i + 1]
            instructions += record.update_tree_leaf(attr_name, e)
        return instructions

    def _replace_var(self, var_name: str, e: Env) -> List[Instr]:
        # override new record with old record
        old_record_location = e.vars[var_name]
        free_old_record, _ = self.free_var(var_name, e)
        move_back_new_record= [
            Instr("DUG", [e.sp - old_record_location], {}),
        ]
        e.vars[var_name] = old_record_location

        return free_old_record + move_back_new_record

    @debug
    def compile_assign_record(self, node: ast.Assign, e: Env) -> List[Instr]:
        record_spec = self._get_record_spec(node.targets[0], e)
        attribute_names = record_spec["attribute_names"]
        var_name = attribute_names[0]

        instructions = self._get_record_with_replace(record_spec, e)

        instructions += self._compile(node.value, e)

        instructions += self._set_record(record_spec, e)

        instructions += self._replace_var(var_name, e)

        return instructions

    @debug
    def compile_assign(self, assign: ast.Assign, e: Env) -> List[Instr]:
        # Handle storage assignments
        try:
            cond1 = assign.targets[0].value.value.id == "self"
            cond2 = assign.targets[0].value.attr == "storage"
            if cond1 and cond2:
                return self.compile_assign_storage_attribute(assign, e)
        except:
            pass

        # Handle reassignments
        try:
            if assign.targets[0].id in e.vars.keys():
                return self._compile_reassign(assign, e)
        except:
            pass

        # Handle subscript assignments
        if type(assign.targets[0]) == ast.Subscript:
            return self.compile_assign_subscript(assign, e)

        # Handle record key assignments
        if type(assign.targets[0]) == ast.Attribute:
            return self.compile_assign_record(assign, e)

        instructions: List[Instr] = []
        var_name = assign.targets[0]

        if self._is_literal(assign.value):
            compiled_value = self.compile_literal(assign.value, e)
        else:
            compiled_value = self._compile(assign.value, e)

        value = assign.value
        instructions = self._compile(var_name, e) + compiled_value
        e.vars[var_name.id] = e.sp

        try:
            if assign.value.func.id in e.records:
                e.types[var_name.id] = assign.value.func.id
        except:
            pass

        try:
            print_val = value.value
        except:
            print_val = "[object]"
        return [Comment(f"{var_name.id} = {print_val}")] + instructions

    @debug
    def compile_expr(self, expr: ast.Expr, e: Env) -> List[Instr]:
        return self._compile(expr.value, e)

    def _is_string_address(self, string: str) -> bool:
        is_tz_address = len(string) == 36 and string[:2] == "tz"
        is_kt_address = len(string) == 36 and string[:2] == "KT"
        return is_tz_address or is_kt_address

    @debug
    def compile_constant(self, constant: ast.Constant, e: Env) -> List[Instr]:
        e.sp += 1  # Account for PUSH

        constant_type: t.Type = t.Int()
        if type(constant.value) == str:
            if self._is_string_address(constant.value):
                constant_type = t.Address()
            else:
                constant_type = t.String()
        elif type(constant.value) == int:
            constant_type = t.Int()
        elif type(constant.value) == bool:
            constant_type = t.Bool()
        else:
            raise NotImplementedError

        return [
            Instr("PUSH", [constant_type, constant.value], {}),
        ]

    @debug
    def compile_name(self, name: ast.Name, e: Env) -> List[Instr]:
        if name.id == "SENDER":
            return self.get_sender(name, e)

        var_name = name

        if type(name.ctx) == ast.Load:
            var_addr = e.vars[var_name.id]
            jump_length = e.sp - var_addr
            comment = [
                Comment(
                    f"Loading {var_name.id} at {var_addr}, e.sp = {e.sp}, jump = {jump_length}"
                )
            ]
            instructions = [
                Instr("DIG", [jump_length], {}),
                Instr("DUP", [], {}),
                Instr("DUG", [jump_length + 1], {}),
            ]
            e.sp += 1  # Account for DUP
            return comment + instructions
        elif type(name.ctx) == ast.Store:
            # will get set to actual value in `compile_assign`
            e.vars[var_name.id] = 42
            return []
        else:
            raise NotImplementedError

    @debug
    def compile_binop(self, t: ast.BinOp, e: Env) -> List[Instr]:
        left = self._compile(t.right, e)
        right = self._compile(t.left, e)
        op = self._compile(t.op, e)
        return left + right + op

    @debug
    def compile_sub(self, t: ast.Sub, e: Env) -> List[Instr]:
        e.sp -= 1  # Account for SUB
        return [
            Instr("SUB", [], {}),
        ]

    @debug
    def compile_add(self, t: ast.Add, e: Env) -> List[Instr]:
        e.sp -= 1  # Account for ADD
        return [
            Instr("ADD", [], {}),
        ]

    @debug
    def create_list(self, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        return [
            Instr("NIL", [t.Int()], {}),
        ]

    @debug
    def append_before_list_el(self, el, e) -> List[Instr]:
        # no sp chage b/c they cancel out btwn the two instructions
        return self._compile(el, e) + [Instr("CONS", [], {})]

    @debug
    def compile_list(self, l: ast.List, e: Env) -> List[Instr]:
        e.sp += 1  # Account for pushing list
        instructions = self.create_list(e)
        for el in reversed(l.elts):
            instructions += self.append_before_list_el(el, e)
        return instructions

    def free_var(self, var_name, e: Env):
        var_location = e.vars[var_name]
        comment = [Comment(f"Freeing var {var_name} at {var_location}, e.sp = {e.sp}")]

        jump = e.sp - var_location
        e.sp -= 1  # account for freeing var
        del e.vars[var_name]

        return (
            comment
            + [
                Instr(
                    "DIP",
                    [
                        jump,
                        [
                            Instr("DROP", [], {}),
                        ],
                    ],
                    {},
                ),
            ],
            e,
        )

    def _get_function_prototype(self, f: ast.FunctionDef, e: Env) -> FunctionPrototype:
        return FunctionPrototype(
            self.type_parser.parse(f.args.args[0].annotation, e),
            self.type_parser.parse(f.returns, e),
        )

    @debug
    def compile_defun(self, f: ast.FunctionDef, e: Env) -> List[Instr]:
        e.sp += 1  # account for body push

        e.vars[f.name] = e.sp

        for arg_ast in f.args.args:
            e.args[f.name] = arg_ast.arg

            # type argument
            if type(arg_ast.annotation) == ast.Name:
                if arg_ast.annotation.id in e.records:
                    e.types[arg_ast.arg] = arg_ast.annotation.id

        prototype = self._get_function_prototype(f, e)
        arg_type, return_type = prototype.arg_type, prototype.return_type
        # get init env keys
        init_var_names = set(e.vars.keys())


        # We work on an env copy to prevent from polluting the environment
        # with vars that we'd need to remove.
        func_env = e.copy()

        # store argument in env
        for arg_ast in f.args.args:
            func_env.sp += 1
            func_env.vars[arg_ast.arg] = func_env.sp

        body_instructions = self._compile_block(f.body, func_env)

        # freeing the argument
        body_instructions += self.free_var(f.args.args[0].arg, func_env)[0]

        comment = [Comment(f"Storing function {f.name} at {e.vars[f.name]}")]
        return comment + [
            Instr(
                "LAMBDA",
                [arg_type, return_type, body_instructions],
                {},
            ),
        ]

    @debug
    def compile_ccall(self, c: ast.Call, e: Env):
        """Call to class constructor"""
        instructions = e.records[c.func.id].compile_record(c.args, self._compile, e)
        return instructions

    def _compile_dict_safe_get(self, key, default, e: Env) -> List[Instr]:
        # dictionary = self._compile(subscript.value, e)
        key = self._compile(key, e)
        # get_instructions = self._compile_get_subscript(e)
        # return dictionary + key + get_instructions

        e.sp -= 1  # account for get

        if_not_env = e.copy()
        if_not_env.sp -= 1  # account for if none
        default = self._compile(default, if_not_env)
        return key + [
            Instr("GET", [], {}),
            Instr("IF_NONE", [default, []], {}),
        ]

    @debug
    def compile_fcall(self, f: ast.Call, e: Env):
        # check if we are calling .dict on a dictionary
        if type(f.func) == ast.Attribute and f.func.attr == "get":
            instructions = self._compile(f.func.value, e)
            instructions += self._compile_dict_safe_get(f.args[0], f.args[1], e)
            return instructions


        # if dealing with a record instantiation, compile as such
        if f.func.id in e.records.keys():
            return self.compile_ccall(f, e)

        func_addr = e.vars[f.func.id]
        jump_length = e.sp - func_addr
        comment = [
            Comment(f"Moving to function {f.func.id} at {func_addr}, e.sp = {e.sp}")
        ]

        load_function = [
            Instr("DIG", [jump_length], {}),
            Instr("DUP", [], {}),
            Instr("DUG", [jump_length + 1], {}),
        ]

        e.sp += 1  # Account for DUP

        # fetch arg name for function
        load_arg = self._compile(f.args[0], e)

        e.sp += 1  # Account for pushing argument

        execute_function = [Instr("EXEC", [], {})]

        e.sp -= 2  # Account popping EXEC and LAMBDA

        instr = comment + load_function + load_arg + execute_function

        return instr

    @debug
    def compile_return(self, r: ast.FunctionDef, e: Env):
        return self._compile(r.value, e)

    def get_init_env(self):
        return Env({}, -1, {}, {}, {})

    @debug
    def compile_entrypoint(self, f: ast.FunctionDef, e: Env, prologue_instructions: List[Instr]) -> List[Instr]:
        e = e.copy()
        # we update the variable pointers to account for the fact that the first
        # element on the stack is Pair(param, storage).
        # we are targetting a stack that will look like [storage, {prologue_instructions}, param]
        # hence, we need to add 1 to all the addresses of the variables in `prologue_instructions`
        e.vars = {var_name: address + 1 for var_name, address in e.vars.items()}

        e.sp += 1  # account for pushing Pair(param, storage)
        e.sp += 1  # account for breaking up Pair(param, storage)

        # Save the storage and entrypoint argument on the stack
        self.contract.instructions = [
            Instr("DUP", [], {}),  # [Pair(param, storage), Pair(param, storage)]
            Instr("CDR", [], {}),  # [Pair(param, storage), storage]
            Instr("DUG", [1], {}), # [storage, Pair(param, storage)]
            Instr("CAR", [], {}),  # [storage, param]
        ] + prologue_instructions + [
            Instr("DIG", [e.sp - 1], {}),  # fetch the entrypoint argument
        ]

        # the storage is at the bottom of the stack
        e.vars["__STORAGE__"] = 0

        # the parameter is a the top of the stack
        # N.B. all variables declared in the prologue instructions) are
        #      laying between the storage and the parameter (hence the +1 above)
        e.vars[f.args.args[0].arg] = e.sp

        # type argument
        if f.args.args[0].annotation.id in e.records:
            e.types[f.args.args[0].arg] = f.args.args[0].annotation.id

        block_instructions = self._compile_block(f.body, e)
        entrypoint_instructions = block_instructions

        free_vars_instructions = self.free_vars(list(e.vars.keys()), e)
        epilogue = [
            Instr("NIL", [t.Operation()], {}),
            Instr("PAIR", [], {}),
        ]

        entrypoint_instructions = entrypoint_instructions + free_vars_instructions + epilogue

        prototype = self._get_function_prototype(f, e)
        entrypoint = Entrypoint(prototype, entrypoint_instructions)
        self.contract.add_entrypoint(f.name, entrypoint)
        return []

    def free_vars(self, var_names: List[str], e: Env) -> List[Instr]:
        # Free from the top of the stack. this ensures that the variable pointers
        # are not changed as variables are freed from the stack
        sorted_keys = sorted(var_names, key=lambda var_name: e.vars[var_name], reverse=True)

        # remove env vars from memory
        free_var_instructions = []
        for var_name in sorted_keys:
            instr, _ = self.free_var(var_name, e)
            free_var_instructions += instr

        return free_var_instructions

    def _compile_block(self, block_ast: List[ast.AST], block_env: Env) -> List[Instr]:
        """frees newly declared variables at the end of the block, hence °e°
        should be the same befor and after the block"""
        # get init env keys
        init_var_names = set(block_env.vars.keys())

        # iterate body instructions
        block_instructions = []
        for i in block_ast:
            block_instructions += self._compile(i, block_env)

        # get new func_env keys
        new_var_names = set(block_env.vars.keys())

        # intersect init and new env keys
        intersection = list(new_var_names - init_var_names)

        free_var_instructions = self.free_vars(intersection, block_env)

        return block_instructions + free_var_instructions

    @debug
    def compile_storage(self, storage_ast, e: Env):
        if type(storage_ast) == ast.Call:
            # assume constructed from record
            storage_type = storage_ast.func.id
            e.types["__STORAGE__"] = storage_type
            self.contract.storage_type = e.records[storage_type].get_type()
        else:
            return NotImplementedError

    @debug
    def _compile_contract(self, contract_ast: ast.ClassDef, e: Env, prologue_instructions: List[Instr]) -> List[Instr]:
        instructions = []
        for entrypoint in contract_ast.body:
            if entrypoint.name == "deploy":
                if type(entrypoint.body[0]) == ast.Return:
                    self.compile_storage(entrypoint.body[0].value, e)
                else:
                    return NotImplementedError
            else:
                instructions += self.compile_entrypoint(entrypoint, e, prologue_instructions)
        return instructions

    @debug
    def compile_record(self, record_ast: ast.ClassDef, e: Env) -> List[Instr]:
        attribute_names = [attr.target.id for attr in record_ast.body]
        attribute_types = []
        attribute_annotations = []
        for attr_name, attr in zip(attribute_names, record_ast.body):
            attribute_annotations.append(attr.annotation)
            attribute_types.append(self.type_parser.parse(attr.annotation, e, "%" + attr_name))

        e.records[record_ast.name] = Record(attribute_names, attribute_types, attribute_annotations)
        return []

    def handle_get_storage(self, storage_get_ast: ast.Attribute, e: Env) -> List[Instr]:
        if storage_get_ast.attr != "storage":
            # storage is record
            key = storage_get_ast.attr
            load_storage_instr = self.compile_name(ast.Name(id='__STORAGE__', ctx=ast.Load()), e)
            storage_key_name = storage_get_ast.attr
            get_storage_key_instr = e.records[e.types['__STORAGE__']].navigate_to_tree_leaf(storage_key_name)
            return load_storage_instr + get_storage_key_instr
        else:
            return self.compile_name(ast.Name(id='__STORAGE__', ctx=ast.Load()), e)

    def check_get_storage(self, storage_get_ast: ast.Attribute) -> bool:
        # tmp fix
        if type(storage_get_ast.value) == ast.Subscript:
            return False

        try:
            return (
                storage_get_ast.value.value.id == "self"
                and storage_get_ast.value.attr == "storage"
            )
        except:
            return (
                storage_get_ast.value.id == "self"
                and storage_get_ast.attr == "storage"
            )

    def check_get_sender(self, sender_ast: ast.Attribute) -> bool:
        return (
            sender_ast.value.id == "self"
            and sender_ast.attr == "sender"
        )

    def get_sender(self, sender_ast: ast.Name, e: Env) -> List[Instr]:
        e.sp += 1  # account for pushing sender
        return [Instr("SENDER", [], {})]

    @debug
    def _compile_attribute(self, node: ast.Attribute, e: Env, instructions = None, recursing = False):
        if instructions is None:
            instructions = []

        def get_next_record(annotation):
            if type(annotation) == ast.Name:
                return annotation.id
            elif type(annotation) == ast.Subscript:
                if len(annotation.slice.value.elts) == 2:
                    # dict
                    return annotation.slice.value.elts[1].id
                elif len(annotation.slice.value.elts) == 1:
                    # list
                    raise NotImplementedError
                else:
                    raise NotImplementedError

        if type(node.value) == ast.Attribute:
            instructions, record = self._compile_attribute(node.value, e, instructions, True)
            instructions += record.navigate_to_tree_leaf(node.attr)
            if recursing:
                new_record_name = record.get_attribute_annotation(node.attr).id
                new_record = e.records[new_record_name]
                return instructions, new_record
        elif type(node.value) == ast.Name:
            var_name = node.value.id
            var_type = e.types[var_name]
            record = e.records[var_type]
            instructions += self._compile(node.value, e)
            instructions += record.navigate_to_tree_leaf(node.attr)
            if recursing:
                new_record_annotation = record.get_attribute_annotation(node.attr)
                new_record_name = get_next_record(new_record_annotation)
                new_record = e.records[new_record_name]
                return instructions, new_record
        elif type(node.value) == ast.Subscript:
            instructions, record = self._compile_attribute(node.value.value, e, instructions, True)
            instructions += self._compile(node.value.slice.value, e)
            instructions += self._compile_get_subscript(e)
            instructions += record.navigate_to_tree_leaf(node.attr)
            if recursing:
                new_record_name = record.get_attribute_annotation(node.attr).id
                new_record = e.records[new_record_name]
                return instructions, new_record
        else:
            raise NotImplementedError

        return instructions


    @debug
    def compile_attribute(self, attribute_ast: ast.Attribute, e: Env) -> List[Instr]:
        if self.check_get_storage(attribute_ast):
            return self.handle_get_storage(attribute_ast, e)

        return self._compile_attribute(attribute_ast, e)

    @debug
    def compile_compare(self, compare_ast: ast.Compare, e: Env) -> List[Instr]:
        compare_instructions = (
            self._compile(compare_ast.comparators[0], e)
            + self._compile(compare_ast.left, e)
            + [Instr("COMPARE", [], {})]
        )
        # Account for COMPARE
        e.sp -= 1

        operator_type = type(compare_ast.ops[0])
        if operator_type == ast.Eq:
            operator_instructions = [Instr("EQ", [], {})]
        elif operator_type == ast.NotEq:
            operator_instructions = [Instr("NEQ", [], {})]
        elif operator_type == ast.Lt:
            operator_instructions = [Instr("LT", [], {})]
        elif operator_type == ast.Gt:
            operator_instructions = [Instr("GT", [], {})]
        elif operator_type == ast.LtE:
            operator_instructions = [Instr("LE", [], {})]
        elif operator_type == ast.GtE:
            operator_instructions = [Instr("GE", [], {})]
        elif operator_type == ast.In:
            # remove COMPARE instruction
            del compare_instructions[-1]
            operator_instructions = [Instr("MEM", [], {})]
        else:
            return NotImplementedError

        return compare_instructions + operator_instructions

    def compile_if(self, if_ast: ast.If, e: Env) -> List[Instr]:
        test_instructions = self._compile(if_ast.test, e)

        # Account for "IF" poping the boolean sitting at the top of the stack
        e.sp -= 1

        if_true_instructions = self._compile_block(if_ast.body, e.copy())
        if_false_instructions = self._compile_block(if_ast.orelse, e.copy())
        if_instructions = [Instr("IF", [if_true_instructions, if_false_instructions], {})]
        return test_instructions + if_instructions

    def compile_raise(self, raise_ast: ast.Raise, e: Env) -> List[Instr]:
        try:
            raise_ast.exc.args[0]
        except:
            breakpoint()
            pass
        return self._compile(raise_ast.exc.args[0], e) + [Instr("FAILWITH", [], {})]

    def _compile_get_subscript(self, e: Env) -> List[Instr]:
        e.sp -= 1  # account for get
        return [
            Instr("GET", [], {}),
            Instr("IF_NONE", [
                [
                    Instr("PUSH", [t.String(), "Key does not exist"], {}),
                    Instr("FAILWITH", [], {})
                ],
                [],
            ], {}),
        ]

    def compile_subscript(self, subscript: ast.Subscript, e: Env) -> List[Instr]:
        dictionary = self._compile(subscript.value, e)
        key = self._compile(subscript.slice.value, e)
        get_instructions = self._compile_get_subscript(e)
        return dictionary + key + get_instructions

    def compile_unary_op(self, node: ast.UnaryOp, e: Env) -> List[Instr]:
        return self._compile(node.operand, e) + [Instr("NOT", [], {})]

    def compile(self):
        return self._compile(self.ast)

    def compile_new(self):
        e = self.get_init_env()
        e.sp = 1  # account for storage and entrypoint arg
        self.contract.instructions = self._compile(self.ast, e)
        print("e.sp = ", e.sp, self.env.sp)
        self._compile(self.ast, self.env)
        return self.contract

    def _compile(self, node_ast, e: Optional[Env] = None, instructions = None, current_type: Optional[t.Type] = None) -> List[Instr]:
        e = self.get_init_env() if not e else e
        self.env = e  # saving as attribute for debug purposes

        if not instructions:
            instructions = []

        if type(node_ast) == ast.Module:
            instructions += self.compile_module(node_ast, e)
            if self.isDebug:
                self.print_instructions(instructions)
        elif type(node_ast) == ast.Assign:
            instructions += self.compile_assign(node_ast, e)
        elif type(node_ast) == ast.AnnAssign:
            instructions += self.compile_ann_assign(node_ast, e)
        elif type(node_ast) == ast.Attribute:
            instructions += self.compile_attribute(node_ast, e)
        elif type(node_ast) == ast.Expr:
            instructions += self.compile_expr(node_ast, e)
        elif type(node_ast) == ast.If:
            instructions += self.compile_if(node_ast, e)
        elif type(node_ast) == ast.Constant:
            instructions += self.compile_constant(node_ast, e)
        elif type(node_ast) == ast.Compare:
            instructions += self.compile_compare(node_ast, e)
        elif type(node_ast) == ast.Name:
            instructions += self.compile_name(node_ast, e)
        elif type(node_ast) == ast.BinOp:
            instructions += self.compile_binop(node_ast, e)
        elif type(node_ast) == ast.Add:
            instructions += self.compile_add(node_ast, e)
        elif type(node_ast) == ast.Sub:
            instructions += self.compile_sub(node_ast, e)
        elif type(node_ast) == ast.List:
            instructions += self.compile_list(node_ast, e)
        elif type(node_ast) == ast.FunctionDef:
            instructions += self.compile_defun(node_ast, e)
        elif type(node_ast) == ast.Return:
            instructions += self.compile_return(node_ast, e)
        elif type(node_ast) == ast.Raise:
            instructions += self.compile_raise(node_ast, e)
        elif type(node_ast) == ast.Call:
            instructions += self.compile_fcall(node_ast, e)
        elif type(node_ast) == ast.ClassDef:
            if node_ast.name == "Contract":
                instructions += self._compile_contract(node_ast, e, instructions)
            elif "dataclass" in [decorator.id for decorator in node_ast.decorator_list]:
                instructions += self.compile_record(node_ast, e)
            else:
                raise NotImplementedError
        elif type(node_ast) == ast.Dict:
            instructions += self.compile_dict(node_ast, current_type.key_type, current_type.value_type, e)
        elif type(node_ast) == ast.Subscript:
            instructions += self.compile_subscript(node_ast, e)
        elif type(node_ast) == ast.UnaryOp:
            instructions += self.compile_unary_op(node_ast, e)
        elif type(node_ast) == ast.ImportFrom:
            # Skip for now
            pass
        elif type(node_ast) == ast.BoolOp:
            instructions += self.compile_bool_op(node_ast, e)
            pass
        else:
            breakpoint()
            raise NotImplementedError

        if self.isDebug:
            print(e)

        return instructions

    def compile_bool_op(self, node: ast.BoolOp, e: Env) -> List[Instr]:
        # AND / x : y : S  =>  (x & y) : S
        # y is pushed first
        y = self._compile(node.values[1], e)
        x = self._compile(node.values[0], e)
        operand_instr = None
        if type(node.op) == ast.And:
            operand_instr = Instr("AND", [], {})
        elif type(node.op) == ast.Or:
            operand_instr = Instr("OR", [], {})
        else:
            raise NotImplementedError
        e.sp -= 1  # account for bool_op
        return y + x + [operand_instr]

    def compile_expression(self, e: Env = None):
        instructions = self._compile(self.ast, e)
        return CompilerBackend().compile_instructions(instructions)

    def compile_contract(self):
        self.ast = macro_expander(self.ast)
        self.compile()
        return CompilerBackend().compile_contract(self.contract)

    @staticmethod
    def print_instructions(instructions):
        print("\n".join([f"{i.name} {i.args} {i.kwargs}" for i in instructions]))


class VM:
    def __init__(self, sender ="tz3M4KAnKF2dCSjqfa1LdweNxBGQRqzvPL88"):
        self.reset_stack()
        self.context = ExecutionContext()
        self.set_sender(sender)

    def execute(self, micheline):
        self.result = InterpreterResult(stdout=[])
        code_section = CodeSection.match(micheline)
        code_section.args[0].execute(self.stack, self.result.stdout, self.context)
        return self

    def load_contract(self, micheline):
        self.contract = ContractInterface.from_micheline(micheline)
        return self

    def reset_stack(self):
        self.stack = MichelsonStack()
        return self

    def set_sender(self, sender):
        self.context.sender = sender
        return self

    def stdout(self):
        print("\n".join(self.result.stdout))
        return self


class TestDict(unittest.TestCase):
    def test_dict_get_with_default(self):
        source = """
@dataclass
class Storage:
    accounts: Dict[str, int]
    counter: int

store = Storage({}, 0)
#store.accounts
store.accounts.get("my_key", "some_default")
"""
        vm = VM()
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().value, 'some_default')
        self.assertEqual(compiler.env.sp, 1)

        source = """
@dataclass
class Storage:
    accounts: Dict[str, int]
    counter: int

store = Storage({}, 0)
store.accounts["my_key"] = "my_value"
store.accounts.get("my_key", "some_default")
"""
        vm = VM()
        micheline = Compiler(source).compile_expression()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().value, 'my_value')
        self.assertEqual(compiler.env.sp, 1)

    def test_record_valued_dict(self):
        vm = VM()
        source = """
@dataclass
class Account:
    name: str
    balance: int

@dataclass
class Storage:
    accounts: Dict[str, Account]
    owner: str

store = Storage({}, "pymich")
store.accounts["my_hash"] = Account("pymich", 1000)
store.accounts["my_hash"].name
"""
        micheline = Compiler(source).compile_expression()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().value, 'pymich')

    def test_record_key_dict(self):
        vm = VM()
        source = """
@dataclass
class Key:
    a: int
    b: int

@dataclass
class Storage:
    accounts: Dict[Key, str]
    owner: str

store = Storage({}, "pymich")
key = Key(10, 20)
store.accounts[key] = "hello"
store.accounts[key]
"""
        micheline = Compiler(source).compile_expression()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().value, 'hello')

    def test_get_dict_no_key_error(self):
        source = """
@dataclass
class Storage:
    balances: Dict[str, int]
    owner: str

storage = Storage({}, "owner")
balances = storage.balances
user = 'Mr. Foobar'
balances[user] = 100
balances[user]
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        expected_int = IntType(100)
        expected_string = StringType('Mr. Foobar')
        expected_map = MapType([(expected_string, expected_int)])
        expected_pair = PairType((MapType([]), StringType("owner")))
        expected_stack = [expected_int, expected_string, expected_map, expected_pair]
        self.assertEqual(vm.stack.items, expected_stack)


    def test_key_in_dict(self):
        source = """
@dataclass
class Storage:
    balances: Dict[str, int]
    owner: str

storage = Storage({}, "owner")
balances = storage.balances
user = 'Mr. Foobar'
balances[user] = 100
user in balances
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        expected_string = StringType('Mr. Foobar')
        expected_int = IntType(100)
        expected_map = MapType([(expected_string, expected_int)])
        expected_pair = PairType((MapType([]), StringType("owner")))
        expected_bool = BoolType(True)
        expected_stack = [expected_bool, expected_string, expected_map, expected_pair]
        self.assertEqual(vm.stack.items, expected_stack)

    def test_get_dict_key_error(self):
        source = """
@dataclass
class Storage:
    balances: Dict[str, int]
    owner: str

storage = Storage({}, "owner")
balances = storage.balances
user = 'Mr. Foobar'
balances[user] = 100
balances['user']
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        try:
            vm.execute(micheline)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Key does not exist'")


class TestRecord(unittest.TestCase):
    def test_nesting_records(self):
        vm = VM()
        source = """
@dataclass
class KYC:
    address: str
    name: str

@dataclass
class Account:
    name: str
    balance: int
    kyc: KYC

@dataclass
class Storage:
    counter: int
    owner: Account

store = Storage(0, Account("pymich", 1000, KYC("kyc_address", "kyc_name")))
"""
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        attr_ast = ast.parse("store.owner").body[0].value
        instructions = compiler._compile_attribute(attr_ast, compiler.env)
        micheline2 = CompilerBackend().compile_instructions(instructions)
        vm.execute(micheline)
        vm.execute(micheline2)
        self.assertEqual(vm.stack.items[0].__repr__(), "('pymich' * (1000 * ('kyc_address' * 'kyc_name')))")

        vm = VM()
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        attr_ast = ast.parse("store.owner.kyc").body[0].value
        instructions = compiler._compile_attribute(attr_ast, compiler.env)
        micheline2 = CompilerBackend().compile_instructions(instructions)
        vm.execute(micheline)
        vm.execute(micheline2)
        self.assertEqual(vm.stack.peek().__repr__(), "('kyc_address' * 'kyc_name')")

        vm = VM()
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        attr_ast = ast.parse("store.owner.kyc.name").body[0].value
        instructions = compiler._compile_attribute(attr_ast, compiler.env)
        micheline2 = CompilerBackend().compile_instructions(instructions)
        vm.execute(micheline)
        vm.execute(micheline2)
        self.assertEqual(vm.stack.peek().value, "kyc_name")

    def test_nested_record_create(self):
        source = """
@dataclass
class SubStorage:
    a: int
    b: int
    c: int

@dataclass
class Storage:
    info: SubStorage
    counter: int

storage = Storage(SubStorage(3, 2, 1), 4)
storage.info
a = storage.info.a
b = storage.info.b
c = storage.info.c
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        for i in range(3):
            self.assertEqual(vm.stack.items[i].value, i + 1)
        # self.assertEqual(vm.stack.peek().__repr__(), '((1 * (2 * 3)) * 4)')

    def test_doubly_nested_record_update(self):
        source = """
@dataclass
class SubStorage:
    a: int
    b: int
    c: int

@dataclass
class Storage:
    info: SubStorage
    counter: int

storage = Storage(SubStorage(1, 2, 3), 4)
storage.info.a = storage.info.a + 10
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().__repr__(), '((11 * (2 * 3)) * 4)')
        self.assertEqual(len(vm.stack), 1)

    def test_triply_nested_record_update(self):
        source = """
@dataclass
class SubSubStorage:
    a: int
    b: int
    c: int

@dataclass
class SubStorage:
    a: int
    sub_storage: SubSubStorage
    c: int

@dataclass
class Storage:
    info: SubStorage
    counter: int

storage = Storage(SubStorage(1, SubSubStorage(10, 20, 30), 2), 3)
storage.info.sub_storage.b = 1000
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().__repr__(), '((1 * ((10 * (1000 * 30)) * 2)) * 3)')
        self.assertEqual(len(vm.stack), 1)

    def test_record_create(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

storage = Storage(1, 2, 3)
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().to_python_object(), (1, 2, 3))

    def test_record_dict_attr(self):
        source = """
@dataclass
class Storage:
    ages: Dict[str, int]
    b: int
    c: int

storage = Storage({}, 2, 3)
foo = "yolo"
storage.ages["foobar"] = 10
storage.ages["foobar"]
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items[0].value, 10)
        self.assertEqual(vm.stack.items[1].value, "yolo")
        self.assertEqual(vm.stack.items[2].__repr__(), "({'foobar': 10} * (2 * 3))")

    def test_nested_record_dict_attr(self):
        source = """
@dataclass
class SubStorage:
    ages: Dict[str, int]
    b: int
    c: int

@dataclass
class Storage:
    sub_storage: SubStorage
    b: int

storage = Storage(SubStorage({}, 1, 2), 3)
foo = "yolo"
storage
storage.sub_storage.ages["foobar"] = 10
storage.sub_storage.ages["foobar"]
"""
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items[0].value, 10)
        self.assertEqual(vm.stack.items[1].__repr__(), "(({} * (1 * 2)) * 3)")
        self.assertEqual(vm.stack.items[2].value, "yolo")
        self.assertEqual(vm.stack.items[3].__repr__(), "(({'foobar': 10} * (1 * 2)) * 3)")

    def test_record_get(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

storage = Storage(1, 2, 3)
storage.a
storage.b
storage.c
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual([item.value for item in vm.stack.items[:-1]], [3, 2, 1])

    def test_record_set(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

storage = Storage(1, 2, 3)
storage.a = 10
storage.b = 20
storage.c = 30
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().to_python_object(), (10, 20, 30))

    def test_record_as_function_argument(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

def add(storage: Storage) -> int:
    return storage.a + storage.b + storage.c

add(Storage(1, 2, 3))
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek(), IntType(6))

    def test_compile_get_record_attribute(self):
        def test(attribute_name, stack_top_value):
            source = f"""
@dataclass
class Storage:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int

a = 2
b = 4
c = 6
my_storage = Storage(1, a, 3, b, 5, c)
my_storage.{attribute_name}
"""
            micheline = Compiler(source).compile_expression()
            vm = VM()
            vm.execute(micheline)
            self.assertEqual(vm.stack.peek().value, stack_top_value)

        test("a", 1)
        test("b", 2)
        test("c", 3)
        test("d", 4)
        test("e", 5)
        test("f", 6)

    def test_compile_create_record(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int

a = 2
b = 4
c = 6
my_storage = Storage(1, a, 3, b, 5, c)
d = 7
my_storage # get storage
"""
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().to_python_object(), (1, 2, 3, 4, 5, 6))

    def test_get_record_entry(self):
        attribute_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        attribute_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        def make_source(attribute_to_get):
            source = "@dataclass\nclass Record:\n"
            for attribute_name in attribute_names:
                source += f"    {attribute_name}: int\n"
            source += "record = Record("
            for attribute_value in attribute_values:
                source += f"{attribute_value}, "
            source += f")\nrecord.{attribute_to_get}"
            return source

        for attribute_name, attribute_value in zip(attribute_names, attribute_values):
            source = make_source(attribute_name)
            micheline = Compiler(source).compile_expression()
            vm = VM()
            vm.execute(micheline)
            self.assertEqual(vm.stack.peek().value, attribute_value)


class TestContract(unittest.TestCase):
    def test_storage_inside_contract(self):
        source = f"""
class Contract:
    counter: int
    admin: address

    def update_counter(self, new_counter: int) :
        self.counter = new_counter

    def update_admin(self, new_admin: address):
        self.admin = new_admin
        """
        compiler = Compiler(source)
        micheline = compiler.compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        self.assertEqual(vm.contract.update_counter(10).interpret().storage['counter'], 10)
        self.assertEqual(vm.contract.update_admin(vm.context.sender).interpret().storage['admin'], vm.context.sender)

    def test_final(self):
        source = f"""
from dataclasses import dataclass
from typing import Dict
from . stubs import *


def require(condition: bool, message: str) -> int:
    if not condition:
        raise Exception(message)

    return 0

@dataclass
class Contract:
    balances: Dict[address, int]
    total_supply: int
    admin: address

    def mint(self, to: address, amount: int):
        require(SENDER == self.admin, "Only admin can mint")

        self.total_supply = self.total_supply + amount

        if to in self.balances:
            self.balances[to] = self.balances[to] + amount
        else:
            self.balances[to] = amount

    def transfer(self, to: address, amount: int):
        require(amount > 0, "You need to transfer a positive amount of tokens")
        require(self.balances[SENDER] >= amount, "Insufficient sender balance")

        self.balances[SENDER] = self.balances[SENDER] - amount

        if to in self.balances:
            self.balances[to] = self.balances[to] + amount
        else:
            self.balances[to] = amount
        """
        compiler = Compiler(source)
        micheline = compiler.compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        new_storage = vm.contract.mint({"to": vm.context.sender, "amount": 10}).interpret(storage=init_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['balances'], {vm.context.sender: 10})

        try:
            vm.contract.mint({"to": vm.context.sender, "amount": 10}).interpret(storage=init_storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only admin can mint'")

        investor = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        new_storage = vm.contract.transfer({"to": investor, "amount": 4}).interpret(storage=new_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage['balances'], {vm.context.sender: 6, investor: 4})

        try:
            vm.contract.transfer({"to": investor, "amount": -10}).interpret(storage=new_storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'You need to transfer a positive amount of tokens'")

        try:
            vm.contract.transfer({"to": investor, "amount": 10}).interpret(storage=new_storage, sender=vm.context.sender).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Insufficient sender balance'")


    def test_multi_arg_entrypoint_or_function(self):
        source = f"""
def require(condition: bool, message: str) -> int:
    if not condition:
        raise Exception(message)

    return 0

class Contract:
    def deploy():
        return 0

    def add_positives(self, x: int, y: int) -> int:
        require(x > 0, "x is not a positive integer")
        return x + y

    def sub(self, x: int) -> int:
        return x
        """
        compiler = Compiler(source)
        micheline = compiler.compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        expected_storage = 5
        actual_storage = vm.contract.add_positives({"x": 2, "y": 3}).interpret(storage=init_storage).storage
        self.assertEqual(actual_storage, expected_storage)

    def test_dataclass_entrypoint_param_1(self):
        source = f"""
@dataclass
class AddParam:
    x: int
    y: int

class Contract:
    def deploy():
        return 0

    def add(param: AddParam) -> int:
        return param.x + param.y

    def sub(x: int) -> int:
        return x
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        expected_storage = 5
        actual_storage = vm.contract.add({"x": 2, "y": 3}).interpret(storage=init_storage).storage
        self.assertEqual(actual_storage, expected_storage)

    def test_dataclass_entrypoint_param_2(self):
        source = """
@dataclass
class Storage:
    balances: Dict[address, int]
    total_supply: int

@dataclass
class ChangeSupplyParam:
    to: address
    amount: int

class Contract:
    def deploy():
        return Storage({}, 0)

    def mint(param: ChangeSupplyParam) -> Storage:
        self.storage.total_supply = self.storage.total_supply + param.amount

        balances = self.storage.balances

        if param.to in balances:
            balances[param.to] = balances[param.to] + param.amount
        else:
            balances[param.to] = param.amount

        self.storage.balances = balances

        return self.storage

    def burn(param: ChangeSupplyParam) -> Storage:
        self.storage.total_supply = self.storage.total_supply - param.amount
        return self.storage
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        tr_to, amount = vm.context.sender, 10
        new_storage = vm.contract.mint({"to": tr_to, "amount": amount}).interpret(storage=init_storage).storage
        expected_storage = {
            "balances": {
                tr_to: amount,
            },
            "total_supply": 10,
        }
        self.assertEqual(new_storage, expected_storage)

    def test_condition_in_function(self):
        source = f"""
@dataclass
class AddParam:
    x: int
    y: int

def foo(param: AddParam) -> int:
    x = param.x
    y = param.y
    if x == param.y:
        x = x + x + x
    else:
        x = x + 10
    return x + y

class Contract:
    def deploy():
        return 0

    def add(param: AddParam) -> int:
        return foo(param)

    def sub(x: int) -> int:
        return x
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        new_storage = vm.contract.add({"x": 2, "y": 2}).interpret(storage=init_storage).storage
        self.assertEqual(new_storage, 8)
        new_storage = vm.contract.add({"x": 2, "y": 3}).interpret(storage=init_storage).storage
        self.assertEqual(new_storage, 15)

    def test_function(self):
        vm = VM()
        admin = vm.context.sender
        source = f"""
@dataclass
class Storage:
    admin: address
    counter: int

@dataclass
class RequireArg:
    condition: bool
    message: str

def require(param: RequireArg) -> int:
    if not param.condition:
        raise Exception(param.message)

    return 0

def double(x: int) -> int:
    return x + x

def triple(x: int) -> int:
    return x + x + x

class Contract:
    def deploy():
        return Storage("{admin}", 0)

    def add(param: int) -> Storage:
        _ = require(RequireArg(SENDER == self.storage.admin, "Only owner can call open"))

        self.storage.counter = self.storage.counter + param
        return self.storage

    def sub(param: int) -> Storage:
        if SENDER != self.storage.admin:
            raise Exception("Only owner can call open")

        self.storage.counter = self.storage.counter - param
        return self.storage

    def quintuple(param: int) -> Storage:
        _ = require(RequireArg(SENDER == self.storage.admin, "Only owner can call open"))

        self.storage.counter = double(self.storage.counter) + triple(self.storage.counter)
        return self.storage
        """
        micheline = Compiler(source).compile_contract()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = admin
        new_storage = vm.contract.add(1).interpret(storage=init_storage, sender=vm.context.sender).storage
        expected_storage = {
            'admin': admin,
            'counter': 1,
        }
        self.assertEqual(new_storage, expected_storage)

        init_storage = expected_storage
        new_storage = vm.contract.sub(1).interpret(storage=init_storage, sender=vm.context.sender).storage
        expected_storage = {
            'admin': admin,
            'counter': 0,
        }
        self.assertEqual(new_storage, expected_storage)

        init_storage = expected_storage
        init_storage['counter'] = 1

        new_storage = vm.contract.quintuple(1).interpret(storage=init_storage, sender=vm.context.sender).storage
        expected_storage = {
            'admin': admin,
            'counter': 5,
        }
        self.assertEqual(new_storage, expected_storage)

    def test_election(self):
        admin =  "tzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        source = f"""
@dataclass
class Storage:
    admin: address
    manifest_url: str
    manifest_hash: str
    open: str
    close: str
    artifacts_url: str
    artifacts_hash: str

@dataclass
class OpenArg:
    open: str
    manifest_url: str
    manifest_hash: str

@dataclass
class ArtifactsArg:
    artifacts_url: str
    artifacts_hash: str

class Contract:
    def deploy():
        return Storage("{admin}", '', '', '', '', '', '')

    def open(params: OpenArg) -> Storage:
        if SENDER != self.storage.admin:
            raise Exception("Only owner can call open")

        self.storage.open = params.open
        self.storage.manifest_url = params.manifest_url
        self.storage.manifest_hash = params.manifest_hash

        return self.storage

    def close(params: str) -> Storage:
        self.storage.close = params

        return self.storage

    def artifacts(params: ArtifactsArg) -> Storage:
        self.storage.artifacts_url = params.artifacts_url
        self.storage.artifacts_hash = params.artifacts_hash

        return self.storage
        """
        compiler = Compiler(source)
        micheline = compiler.compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        init_storage['admin'] = vm.context.sender

        try:
            vm.contract.open({"open": "foo", "manifest_url": "bar", "manifest_hash": "baz"}).interpret(storage=init_storage)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'Only owner can call open'")

        new_storage = vm.contract.open({"open": "foo", "manifest_url": "bar", "manifest_hash": "baz"}).interpret(storage=init_storage, sender=vm.context.sender).storage
        expected_storage = init_storage.copy()
        expected_storage["open"] = "foo"
        expected_storage["manifest_url"] = "bar"
        expected_storage["manifest_hash"] = "baz"
        self.assertEqual(new_storage, expected_storage)

        expected_storage["close"] = "foobar"
        new_storage = vm.contract.close("foobar").interpret(storage=new_storage, sender=vm.context.sender).storage
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.artifacts({"artifacts_url": "1", "artifacts_hash": "2"}).interpret(storage=new_storage, sender=vm.context.sender).storage
        expected_storage["artifacts_url"] = "1"
        expected_storage["artifacts_hash"] = "2"
        self.assertEqual(new_storage, expected_storage)

    def test_attribute_reassign(self):
        source = """
@dataclass
class Storage:
    a: int
    b: int
    c: int

class Contract:
    def deploy():
        return Storage(1, 2, 3)

    def set_a(new_a: int) -> int:
        self.storage.a = new_a
        return self.storage

    def set_b(new_b: int) -> int:
        self.storage.b = new_b
        return self.storage

    def set_c(new_c: int) -> int:
        self.storage.c = new_c
        return self.storage
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()
        new_storage = vm.contract.set_a(10).interpret(storage=init_storage).storage
        expected_storage = init_storage
        expected_storage["a"] = 10
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.set_b(20).interpret(storage=expected_storage).storage
        expected_storage["b"] = 20
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.set_c(30).interpret(storage=expected_storage).storage
        expected_storage["c"] = 30
        self.assertEqual(new_storage, expected_storage)


    def test_contract_final(self):
        vm = VM()
        owner = vm.context.sender
        source = f"""
@dataclass
class Storage:
    owner: address
    name: str
    counter: int

class Contract:
    def deploy():
        return Storage("{owner}", "foo", 0)

    def add(a: int) -> int:
        if a < 10:
            raise Exception('input smaller than 10')
        else:
            a = a + a
            self.storage.counter = self.storage.counter + a
            return self.storage

    def update_owner(new_owner: address) -> int:
        self.storage.owner = new_owner
        return self.storage

    def update_name(new_name: str) -> int:
        self.storage.name = new_name
        return self.storage
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()

        try:
            vm.contract.add(1).interpret(storage=init_storage).storage
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'input smaller than 10'")

        new_storage = vm.contract.add(10).interpret(storage=init_storage).storage
        expected_storage = init_storage
        expected_storage["counter"] = 20
        self.assertEqual(new_storage, expected_storage)

        ZERO_ADDRESS = "tz1burnburnburnburnburnburnburjAYjjX"
        new_storage = vm.contract.update_owner(ZERO_ADDRESS).interpret(storage=new_storage).storage
        expected_storage["owner"] = ZERO_ADDRESS
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.update_name("bar").interpret(storage=new_storage).storage
        expected_storage["name"] = "bar"
        self.assertEqual(new_storage, expected_storage)

    def contract_multitype_storage(self):
        source = """
@dataclass
class Storage:
    owner: str
    counter: int

class Contract:
    def deploy():
        return Storage("foo", 0)

    def add(a: int) -> int:
        return Storage(self.storage.owner, self.storage.counter + a)

    def update_owner(new_owner: str) -> int:
        return Storage(new_owner, self.storage.counter)
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()

        new_storage = vm.contract.add(10).interpret(storage=init_storage).storage
        expected_storage = init_storage
        expected_storage["counter"] = 10
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.update_owner("foo").interpret(storage=init_storage).storage
        expected_storage["owner"] = "foo"
        self.assertEqual(new_storage, expected_storage)

    def test_contract_storage(self):
        source = """
@dataclass
class Storage:
    owner_id: int
    counter: int

class Contract:
    def deploy():
        return Storage(1, 0)

    def add(a: int) -> int:
        b = 10
        new_storage = Storage(self.storage.owner_id, self.storage.counter + a + b)
        return new_storage

    def update_owner_id(new_id: int) -> int:
        return Storage(new_id, self.storage.counter)
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()

        new_storage = vm.contract.add(10).interpret(storage=init_storage).storage
        expected_storage = init_storage
        expected_storage["counter"] = 20
        self.assertEqual(new_storage, expected_storage)

        new_storage = vm.contract.update_owner_id(10).interpret(storage=init_storage).storage
        expected_storage["owner_id"] = 10
        self.assertEqual(new_storage, expected_storage)

    def test_multi_entrypoint_contract(self):
        source = """
class Contract:
    def deploy():
        return 0

    def incrementByTwo(a: int) -> int:
        b = 1
        return a + b + 1

    def bar(b: int) -> int:
        return b
        """
        micheline = Compiler(source).compile_contract()
        vm = VM()
        vm.load_contract(micheline)
        init_storage = vm.contract.storage.dummy()

        new_storage = vm.contract.incrementByTwo(10).interpret(storage=init_storage).storage
        self.assertEqual(new_storage, 12)

        new_storage = vm.contract.bar(10).interpret(storage=new_storage).storage
        self.assertEqual(new_storage, 10)


class TestCompilerList(unittest.TestCase):
    def test_create_list(self):
        source = "[]"
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual([el.value for el in vm.stack.peek().items], [])

    def test_list_instanciation(self):
        source = """
[1, 2, 3]
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual([el.value for el in vm.stack.peek().items], [1, 2, 3])

class TestCompilerAssign(unittest.TestCase):
    def test_reassign(self):
        source = """
a = 1
b = 2
a = a + 2
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual([item.value for item in vm.stack.items], [2, 3])
        self.assertEqual(compiler.env.vars["a"], 0)
        self.assertEqual(compiler.env.vars["b"], 1)


class TestCompilerDefun(unittest.TestCase):
    def test_func_def(self):
        source = """baz = 1
def foo(a: int) -> int:
    b = 2
    return a + b + 3
bar = foo(baz)
fff = foo(bar)
foo(foo(bar))
"""
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek().value, 16)
        fun = vm.stack.items[3]
        self.assertEqual([arg.prim for arg in fun.args], ['int', 'int'])
        self.assertEqual(len(vm.stack), 5)

    def todo_test_multiple_args_func(self):
        source = """
def add(a, b):
    return a + b
foo(1, 2)
"""
        pass
        #vm = OldVM(isDebug=False)
        #c = Compiler(source, isDebug=False)
        #instructions = c._compile(c.ast)
        #vm._run_instructions(instructions)
        #self.assertEqual(vm.stack[-1], 16)
        #self.assertEqual(len(vm.stack), 5)


class TestCompilerIntegration(unittest.TestCase):
    def test_push_address(self):
        user_address = "tz1S792fHX5rvs6GYP49S1U58isZkp2bNmn6"
        contract_address = "KT1EwUrkbmGxjiRvmEAa8HLGhjJeRocqVTFi"
        regular_string = "foobar"
        source = f"""
user_address = "{user_address}"
contract_address = "{contract_address}"
regular_string = "{regular_string}"
        """
        c = Compiler(source, isDebug=False)
        instructions = [instr for instr in c._compile(c.ast) if instr.name != "COMMENT"]
        expected_instructions = [
            Instr("PUSH", [t.Address(), user_address], {}),
            Instr("PUSH", [t.Address(), contract_address], {}),
            Instr("PUSH", [t.String(), regular_string], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

    def test_store_vars_and_add(self):
        source = """
a = 1
b = 2
c = a + b + b
a + b + c
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [IntType(i) for i in [8, 5, 2, 1]])

    def test_push_string(self):
        source = "'foobar'"
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [StringType("foobar")])

    def test_compare(self):
        source = "1 < 2"
        c = Compiler(source, isDebug=False)
        instructions = c._compile(c.ast)
        expected_instructions = [
            Instr("PUSH", [t.Int(), 2], {}),
            Instr("PUSH", [t.Int(), 1], {}),
            Instr("COMPARE", [], {}),
            Instr("LT", [], {}),
        ]
        self.assertEqual(instructions, expected_instructions)

        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [BoolType(True)])

    def test_if(self):
        source = """
if 1 < 2:
    "foo"
else:
    "bar"
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [StringType("foo")])

    def test_if_reassign(self):
        source = """
foo = "foo"
if 1 < 2:
    foo = "bar"
else:
    foo = "baz"
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [StringType("bar")])
        self.assertEqual(compiler.env.vars["foo"], 0)

    def test_if_failwith(self):
        source = """
foo = "foo"
if 1 < 2:
    raise Exception("my error")
else:
    foo = "baz"
        """
        compiler = Compiler(source)
        micheline = compiler.compile_expression()
        vm = VM()
        try:
            vm.execute(micheline)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'my error'")

    def test_raise(self):
        source = "raise Exception('foobar')"
        compiler = Compiler(source, isDebug=False)
        instructions = compiler._compile(compiler.ast)
        expected_instructions = [
            Instr("PUSH", [t.String(), 'foobar'], {}),
            Instr("FAILWITH", [], {})
        ]
        self.assertEqual(instructions, expected_instructions)

        micheline = compiler.compile_expression()
        vm = VM()
        try:
            vm.execute(micheline)
            assert 0
        except MichelsonRuntimeError as e:
            self.assertEqual(e.format_stdout(), "FAILWITH: 'foobar'")

    def test_reassign_in_condition(self):
        def get_source(a):
            return f"""
a = {a}
if a > 0:
    a = 11
else:
    a = 12
            """
        micheline = Compiler(get_source(10)).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [IntType(11)])

        micheline = Compiler(get_source(0)).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.items, [IntType(12)])

    def test_sender(self):
        source = "SENDER"
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek(), AddressType.from_value(vm.context.sender))

    def test_booleans(self):
        vm = VM()

        for boolean in (True, False):
            source = str(boolean)
            micheline = Compiler(source).compile_expression()
            vm.execute(micheline)
            self.assertEqual(vm.stack.peek(), BoolType(boolean))

    def test_not(self):
        source = "not True"
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek(), BoolType(False))

    def test_and(self):
        sources = [
            ("True and False", False),
            ("True and True", True),
            ("False and False", False),
            ("False and True", False),
        ]
        for source, answer in sources:
            micheline = Compiler(source).compile_expression()
            vm = VM()
            vm.execute(micheline)
            self.assertEqual(vm.stack.peek().to_python_object(), answer)

    def test_import(self):
        source = """
from pymich import *
False
        """
        micheline = Compiler(source).compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek(), BoolType(False))

    def test_callback(self):
        source = """
def apply(f: Callable[int, int], x: int) -> int:
    return f(x)

def increment(x: int) -> int:
    return x + 1

apply(increment, 10)
        """
        from macro_expander import TuplifyFunctionArguments
        macro_expander = TuplifyFunctionArguments()
        compiler = Compiler(source)
        new_ast = macro_expander.visit(compiler.ast)
        new_ast.body = macro_expander.dataclasses + compiler.ast.body
        compiler.ast = new_ast
        micheline = compiler.compile_expression()
        vm = VM()
        vm.execute(micheline)
        self.assertEqual(vm.stack.peek(), IntType(11))


for TestSuite in [
    TestDict,
    TestContract,
    TestRecord,
    TestCompilerList,
    TestCompilerAssign,
    TestCompilerDefun,
    TestCompilerIntegration,
]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSuite)
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    unittest.main()
