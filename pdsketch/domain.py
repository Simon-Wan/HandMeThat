#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : domain.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/30/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Callable

import torch

import jacinle
from jacinle.utils.printing import indent_text, stprint
from pdsketch.expr import FunctionArgumentType, FunctionReturnType, FunctionDef, PredicateDef, PredicateApplication
from pdsketch.expr import AssignOp, ConditionalAssignOp, DeicticAssignOp
from pdsketch.expr import Expression, ExpressionDefinitionContext
from pdsketch.expr import ValueOutputExpression
from pdsketch.expr import cvt_expression_list, flatten_expression, get_used_state_variables
from pdsketch.generator import GeneratorDef
from pdsketch.operator import Precondition, Effect, Operator
from pdsketch.state import State, ValueDict
from pdsketch.value import BOOL, BasicValueType, FLOAT32, INT64, NamedValueType, ObjectType, PyObjValueType, Value, Variable, VectorValueType

logger = jacinle.get_logger(__file__)

__all__ = ['Domain', 'Problem']


class PythonFunctionRef(object):
    """A reference to a Python function.

    This class is used to wrap external function implementations in domains.
    """

    def __init__(self, function: Callable, function_quantized=None, auto_broadcast=True):
        self.function = function
        self.function_quantized = function_quantized
        self.auto_broadcast = auto_broadcast

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        return f'PythonFunctionRef({self.function}, fq={self.function_quantized}, auto_broadcast={self.auto_broadcast})'

    def __repr__(self):
        return self.__str__()


class _TypedVariableView(object):
    """Use `domain.typed_variable['type_name']('variable_name')`"""

    def __init__(self, domain):
        self.domain = domain

    def __getitem__(self, typename):
        def function(string):
            return Variable(string, self.domain.types[typename])
        return function


class Domain(object):
    def __init__(self):
        super().__init__()

        self.name: str = ''
        self.types: Dict[str, Union[ObjectType, NamedValueType]] = dict()
        self.predicates: Dict[str, PredicateDef] = dict()
        self.operators: Dict[str, Operator] = dict()
        self.operator_templates: Dict[str, Operator] = dict()
        self.axioms: Dict[str, Operator] = dict()
        self.external_functions: Dict[str, FunctionDef] = dict()
        self.external_functions_implementation: Dict[str, PythonFunctionRef] = dict()
        self.generators: Dict[str, GeneratorDef] = dict()

        """domain.tv is a helper function that returns a variable with the given type.
        For example, domain.tv['object']('x') returns a variable of type 'object' with name 'x'.
        """
        self.tv = self.typed_variable = _TypedVariableView(self)

    def __getattr__(self, item):
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError

        # NB(Jiayuan Mao @ 09/03): PDDL definition convention.
        item = item.replace('_', '-')

        if item.startswith('t-'):
            return self.types[item[2:]]
        elif item.startswith('p-') or item.startswith('f-'):
            return self.predicates[item[2:]]
        elif item.startswith('op-'):
            return self.operators[item[3:]]
        elif item.startswith('gen-'):
            return self.generators[item[4:]]
        raise NameError('Unknown attribute: {}.'.format(item))

    def set_name(self, name: str):
        self.name = name

    BUILTIN_TYPES = ['object', 'pyobject', 'bool', 'int64', 'float32']

    def define_type(self, typename, parent_name: Optional[Union[VectorValueType, BasicValueType, str]] = None):
        if typename in type(self).BUILTIN_TYPES:
            raise ValueError('Typename {} is a builtin type.'.format(typename))

        assert isinstance(parent_name, (str, VectorValueType)), f'Currently only support inheritance from builtin types: {type(self).BUILTIN_TYPES}.'

        if parent_name == 'object':
            self.types[typename] = ObjectType(typename)
        elif parent_name == 'pyobject':
            dtype = PyObjValueType(typename)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
        elif parent_name == 'int64':
            dtype = NamedValueType(typename, INT64)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
            self.implement_external_function(f'type::{typename}::equal', lambda x, y: Value(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor), x.batch_dims, quantized=True))
        elif parent_name == 'float32':
            dtype = NamedValueType(typename, FLOAT32)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
            self.implement_external_function(f'type::{typename}::equal', lambda x, y: Value(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor), x.batch_dims, quantized=True))
        elif isinstance(parent_name, VectorValueType):
            dtype = NamedValueType(typename, parent_name)
            self.types[typename] = dtype
            self.declare_external_function(f'type::{typename}::equal', [dtype, dtype], BOOL, {})
            self.implement_external_function(f'type::{typename}::equal', lambda x, y: Value(BOOL, x.batch_variables, torch.eq(x.tensor, y.tensor).all(dim=-1), x.batch_dims, quantized=True))
        else:
            raise ValueError(f'Unknown parent type: {parent_name}.')

    def define_predicate(
        self, name: str, arguments: Sequence[FunctionArgumentType], return_type: Optional[FunctionReturnType] = None,
        observation: Optional[bool] = None, state: Optional[bool] = None,
        generator_placeholder: bool = False
    ):
        predicate = PredicateDef(name, arguments, return_type, observation=observation, state=state, generator_placeholder=generator_placeholder)
        self._define_predicate_inner(name, predicate)
        return predicate

    def define_derived(self, name: str, arguments: Sequence[FunctionArgumentType], return_type: Optional[FunctionReturnType] = None, expr: ValueOutputExpression = None, state: bool = False, generator_placeholder: bool = False):
        predicate_def = PredicateDef(name, arguments, return_type, observation=False, state=state, generator_placeholder=generator_placeholder, expr=expr)
        return self._define_predicate_inner(name, predicate_def)

    def _define_predicate_inner(self, name, predicate_def):
        self.predicates[name] = predicate_def

        # NB(Jiayuan Mao @ 07/21): a non-cacheable function is basically an external function.
        if not predicate_def.is_cacheable and predicate_def.expr is None:
            identifier = f'predicate::{name}'
            self.external_functions[identifier] = predicate_def

        return predicate_def

    def define_operator(self, name: str, parameters: Sequence[Variable], preconditions: Sequence[Precondition], effects: Sequence[Effect], template=False, extends=None) -> Operator:
        self.operators[name] = op = Operator(self, name, parameters, preconditions, effects, extends=extends, is_template=template)
        return op

    def define_axiom(self, name: Optional[str], parameters: Sequence[Variable], preconditions: Sequence[Precondition], effects: Sequence[Effect]) -> Operator:
        if name is None:
            name = f'axiom_{len(self.axioms)}'
        self.axioms[name] = op = Operator(self, name, parameters, preconditions, effects, is_axiom=True)
        return op

    def declare_external_function(self, identifier: str, arguments: Sequence[FunctionArgumentType], return_type: FunctionReturnType, kwargs: Optional[Mapping[str, Any]] = None):
        function_def = FunctionDef(identifier, arguments, return_type, **kwargs)
        self.external_functions[identifier] = function_def
        return function_def

    def implement_external_function(self, identifier: str, function, function_quantized=None, auto_broadcast=True, notexists_ok=True):
        if identifier not in self.external_functions:
            if notexists_ok:
                logger.warning('Unknown external function: {}.'.format(identifier))
                return
            raise NameError('Unknown external function: {}.'.format(identifier))
        self.external_functions_implementation[identifier] = PythonFunctionRef(
            function,
            function_quantized,
            auto_broadcast=auto_broadcast
        )

    def define_generator(self, name, parameters, certifies, context, generates, priority=0, unsolvable=False):
        if unsolvable:
            priority = int(1e9)

        identifier = f'generator::{name}'
        context: List[ValueOutputExpression] = cvt_expression_list(context)
        generates: List[ValueOutputExpression] = cvt_expression_list(generates)

        arguments = [Variable(f'?c{i}', c.return_type) for i, c in enumerate(context)]
        return_type = [target.return_type for target in generates]
        output_vars = [Variable(f'?g{i}', g.return_type) for i, g in enumerate(generates)]
        function_def = FunctionDef(identifier, arguments, return_type)

        all_variables = {c: cv for c, cv in zip(context, arguments)}
        all_variables.update({g: gv for g, gv in zip(generates, output_vars)})
        ctx = ExpressionDefinitionContext(*arguments, *output_vars, domain=self)
        flatten_certifies = flatten_expression(certifies, all_variables, ctx)

        if not unsolvable:
            self.external_functions[identifier] = function_def

        if name in self.generators:
            raise ValueError(f'Duplicate generator: {name}.')
        self.generators[name] = generator = GeneratorDef(name, parameters, certifies, context, generates, function_def, output_vars, flatten_certifies, priority=priority, unsolvable=unsolvable)
        return generator

    def has_external_function(self, name):
        return name in self.external_functions_implementation

    def get_external_function(self, name):
        if name in self.external_functions_implementation:
            return self.external_functions_implementation[name]
        if name in self.predicates:
            return self.external_functions_implementation['predicate::' + name]
        raise KeyError('Unknown external function: {}.'.format(name))

    def parse(self, string: Union[str, Expression], variables: Optional[Sequence[Variable]] = None) -> Expression:
        if isinstance(string, Expression):
            return string

        from .parser import parse_expression
        return parse_expression(self, string, variables)

    def print_summary(self):
        print(f'Domain <{self.name}>')
        stprint(key='Types: ', data=self.types, indent=1)
        stprint(key='Predicates: ', data=self.predicates, indent=1)
        stprint(key='External Functions: ', data=self.external_functions, indent=1)
        stprint(key='External Function Implementations: ', data=self.external_functions_implementation, indent=1)
        stprint(key='Generators: ', data=self.generators, indent=1)
        print('  Operators:')
        if len(self.operators) > 0:
            for op in self.operators.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')
        print('  Axioms:')
        if len(self.axioms) > 0:
            for op in self.axioms.values():
                print(indent_text(op.pddl_str(), level=2))
        else:
            print('    <Empty>')

    def post_init(self):
        self.analyze_static_predicates()

    def analyze_static_predicates(self):
        dynamic = set()
        for op in itertools.chain(self.operators.values(), self.axioms.values()):
            for eff in op.effects:
                if isinstance(eff.assign_expr, (AssignOp, ConditionalAssignOp)):
                    dynamic.add(eff.assign_expr.predicate.predicate_def.name)
                elif isinstance(eff.assign_expr, DeicticAssignOp):
                    expr = eff.unwrapped_assign_expr
                    assert isinstance(expr, (AssignOp, ConditionalAssignOp))
                    dynamic.add(expr.predicate.predicate_def.name)
                else:
                    raise TypeError(f'Unknown effect type: {eff.assign_expr}.')

        # propagate the static predicates.
        for p in self.predicates.values():
            if p.is_state_variable:
                if p.name not in dynamic:
                    p.mark_static()
            else:
                if p.is_cacheable and p.expr is not None:
                    used_predicates = get_used_state_variables(p.expr)
                    static = True
                    for predicate_def in used_predicates:
                        if not predicate_def.is_static:
                            static = False
                            break
                    if static:
                        p.mark_static()


class Problem(object):
    def __init__(self):
        self.objects: Dict[str, str] = dict()
        self.predicates: List[PredicateApplication] = list()
        self.goal: Optional[ValueOutputExpression] = None

    def add_object(self, name: str, typename: str) -> None:
        self.objects[name] = typename

    def add_predicate(self, predicate: PredicateApplication) -> None:
        self.predicates.append(predicate)

    def set_goal(self, goal: ValueOutputExpression) -> None:
        self.goal = goal

    def to_state(self, domain) -> State:
        object_names = list(self.objects.keys())
        object_types = [domain.types[self.objects[name]] for name in object_names]
        state = State(object_types, ValueDict(), object_names)

        ctx = state.define_context(domain)
        predicates = list()
        for p in self.predicates:
            predicates.append(ctx.get_pred(p.predicate_def.name)(*[arg.name for arg in p.arguments]))
        ctx.define_predicates(predicates)

        return state

