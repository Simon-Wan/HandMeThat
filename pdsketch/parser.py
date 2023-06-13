#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : parser.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/30/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import os.path as osp
import itertools
import collections
from typing import Optional, Union, Sequence, Tuple, Set, List
from lark import Lark, Tree, Transformer, v_args

import jacinle
import pdsketch.expr as E
from .value import ObjectType, ValueType, BasicValueType, VectorValueType, BOOL, Variable, StringConstant, AUTO
from .state import State
from .expr import ExpressionDefinitionContext, get_definition_context, PredicateDef
from .operator import Precondition, Effect
from .domain import Domain, Problem

__all__ = ['PDSketchParser', 'load_domain_file', 'load_domain_string', 'parse_expression', 'load_problem_file', 'PDDLTransformer']

logger = jacinle.get_logger(__file__)

# lark.v_args
inline_args = v_args(inline=True)
DEBUG_LOG_COMPOSE = False


def _log_function(func):
    if DEBUG_LOG_COMPOSE:
        return jacinle.log_function(func)
    return func


class PDSketchParser(object):
    grammar_file = osp.join(osp.dirname(__file__), 'pdsketch-v2.grammar')

    def __init__(self):
        with open(type(self).grammar_file) as f:
            self.lark = Lark(f)

    def load(self, file):
        with open(file) as f:
            return self.lark.parse(f.read())

    def loads(self, string):
        return self.lark.parse(string)

    def make_domain(self, tree: Tree) -> Domain:
        assert tree.children[0].data == 'definition'
        transformer = PDDLTransformer(Domain())
        transformer.transform(tree)
        domain = transformer.domain
        domain.post_init()
        return domain

    def make_problem(self, tree: Tree, domain: Domain, **kwargs) -> Problem:
        assert tree.children[0].data == 'definition'
        transformer = PDDLTransformer(domain, **kwargs)
        transformer.transform(tree)
        problem = transformer.problem
        return problem

    def make_expression(self, domain: Domain, tree: Tree, variables: Optional[Sequence[Variable]] = None) -> E.Expression:
        if variables is None:
            variables = list()
        transformer = PDDLTransformer(domain, allow_object_constants=True)
        node = transformer.transform(tree).children[0]
        assert isinstance(node, (_FunctionApplicationImm, _QuantifierApplicationImm))
        with ExpressionDefinitionContext(*variables, domain=domain).as_default():
            return node.compose()


_parser = PDSketchParser()


def load_domain_file(filename: str) -> Domain:
    tree = _parser.load(filename)
    domain = _parser.make_domain(tree)
    return domain


def load_domain_string(domain_string: str) -> Domain:
    tree = _parser.loads(domain_string)
    domain = _parser.make_domain(tree)
    return domain


def load_problem_file(filename: str, domain: Domain, **kwargs) -> Tuple[State, E.ValueOutputExpression]:
    tree = _parser.load(filename)
    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_problem(tree, domain, **kwargs)
    return problem.to_state(domain), problem.goal


def parse_problem_string(problem_string: str, domain: Domain, **kwargs) -> Tuple[State, E.ValueOutputExpression]:
    tree = _parser.loads(problem_string)
    with ExpressionDefinitionContext(domain=domain).as_default():
        problem = _parser.make_problem(tree, domain, **kwargs)
    return problem.to_state(domain), problem.goal


def parse_expression(domain: Domain, string: str, variables: Sequence[Variable]) -> E.Expression:
    tree = _parser.loads(string)
    expr = _parser.make_expression(domain, tree, variables)
    return expr


class PDDLTransformer(Transformer):
    def __init__(self, init_domain: Domain = None, allow_object_constants: bool = True, ignore_unknown_predicates: bool = False):
        super().__init__()

        self.domain = init_domain
        self.problem = Problem()
        self.allow_object_constants = allow_object_constants
        self.ignore_unknown_predicates = ignore_unknown_predicates
        self.ignored_predicates: Set[str] = set()

    @inline_args
    def definition_decl(self, definition_type, definition_name):
        if definition_type.value == 'domain':
            self.domain.name = definition_name.value

    def type_definition(self, args):
        # Very ugly hack to handle multi-line definition in PDDL.
        # In PDDL, type definition can be separated by newline.
        # This kinds of breaks the parsing strategy that ignores all whitespaces.
        # More specifically, consider the following two definitions:
        # ```
        # (:types
        #   a
        #   b - a
        # )
        # ```
        # and
        # ```
        # (:types
        #   a b - a
        # )
        if isinstance(args[-1], Tree) and args[-1].data == "parent_type_name":
            parent_line, parent_name = args[-1].children[0]
            args = args[:-1]
        else:
            parent_line, parent_name = -1, 'object'

        for lineno, typedef in args:
            assert typedef is not AUTO, 'AUTO type is not allowed in type definition.'

        for arg in args:
            arg_line, arg_name = arg
            if arg_line == parent_line:
                self.domain.define_type(arg_name, parent_name)
            else:
                self.domain.define_type(arg_name, parent_name)

    @inline_args
    def constants_definition(self, *args):
        raise NotImplementedError()

    @inline_args
    def predicate_definition(self, name, *args):
        name, kwargs = name

        return_type = kwargs.pop('return_type', None)
        self._predicate_definition_inner(name, args, return_type, kwargs)

    @inline_args
    def predicate_definition2(self, name, *args):
        name, kwargs = name
        assert 'return_type' not in kwargs
        args, return_type = args[:-1], self.domain.types[args[-1]]
        self._predicate_definition_inner(name, args, return_type, kwargs)

    def _predicate_definition_inner(self, name, args, return_type, kwargs):
        generators = kwargs.pop('generators', None)
        predicate_def = self.domain.define_predicate(name, args, return_type, **kwargs)

        if generators is not None:
            generators: List[str]
            for target_variable_name in generators:
                assert target_variable_name.startswith('?')
                parameters, certifies, context, generates = _canonize_inline_generator_def_predicate(self.domain, target_variable_name, predicate_def)
                generator_name = f'gen-{predicate_def.name}-{target_variable_name[1:]}' if len(generators) > 1 else f'gen-{predicate_def.name}'
                self.domain.define_generator(generator_name, parameters, certifies, context, generates)

    @inline_args
    def predicate_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def type_name(self, name):
        if name.value == 'auto':
            return name.line, AUTO
        # propagate the "lineno" of the type definition up.
        return name.line, name.value

    @inline_args
    def object_type_name(self, typedef):
        return typedef

    @inline_args
    def value_type_name(self, typedef):
        lineno, typedef = typedef
        if typedef is AUTO:
            return lineno, AUTO
        if isinstance(typedef, VectorValueType):
            return lineno, typedef
        assert isinstance(typedef, str)
        if typedef in self.domain.types:
            return lineno, self.domain.types[typedef]
        return lineno, BasicValueType(typedef)

    @inline_args
    def vector_type_name(self, dtype, dim, choices, kwargs=None):
        choices = choices.children[0] if len(choices.children) > 0 else 0
        if kwargs is None:
            kwargs = dict()
        lineno, dtype = dtype
        return lineno, VectorValueType(dtype, dim, choices, **kwargs)

    @inline_args
    def object_type_name_unwrapped(self, typedef):
        return typedef[1]

    @inline_args
    def value_type_name_unwrapped(self, typedef):
        return typedef[1]

    @inline_args
    def predicate_group_definition(self, *args):
        raise NotImplementedError()

    @inline_args
    def action_definition(self, name, *defs):
        name, kwargs = name

        parameters = tuple()
        precondition = None
        effect = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters = def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper):
                effect = def_.effect
            else:
                raise TypeError('Unknown definition type: {}.'.format(type(def_)))

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                precondition = _canonize_precondition(precondition)
        else:
            precondition = list()

        if effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", is_effect_definition=True).as_default():
                effect = _canonize_effect(effect)
        else:
            effect = list()

        self.domain.define_operator(name, parameters, precondition, effect, **kwargs)

    @inline_args
    def action_definition2(self, name, extends, *defs):
        name, kwargs = name

        assert 'extends' not in kwargs, 'Instantiation cannot be set using decorators. Use :extends instead.'
        kwargs['extends'] = extends

        template_op = self.domain.operators[extends]
        parameters = template_op.arguments

        precondition = None
        effect = None

        for def_ in defs:
            if isinstance(def_, _ParameterListWrapper):
                parameters += def_.parameters
            elif isinstance(def_, _PreconditionWrapper):
                precondition = def_.precondition
            elif isinstance(def_, _EffectWrapper):
                effect = def_.effect
            else:
                raise TypeError('Unknown definition type: {}.'.format(type(def_)))

        if precondition is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}").as_default():
                precondition = tuple(_canonize_precondition(precondition))
        else:
            precondition = tuple()
        if effect is not None:
            with ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"action::{name}", is_effect_definition=True).as_default():
                effect = tuple(_canonize_effect(effect))
        else:
            effect = tuple()

        self.domain.define_operator(name, parameters, template_op.preconditions + precondition, template_op.effects + effect, **kwargs)

    def action_parameters(self, args):
        return _ParameterListWrapper(tuple(args))

    @inline_args
    def action_precondition(self, function_call):
        return _PreconditionWrapper(function_call)

    @inline_args
    def action_effect(self, function_call):
        return _EffectWrapper(function_call)

    @inline_args
    def action_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def action_instantiates(self, name):
        return name.value

    @inline_args
    def axiom_definition(self, decorator, vars, context, implies):
        kwargs = dict() if len(decorator.children) == 0 else decorator.children[0]
        vars = tuple(vars.children)
        precondition = context.children[0]
        effect = implies.children[0]

        name = kwargs.pop('name', None)
        scope = None if name is None else f"axiom::{name}"

        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope).as_default():
            precondition = _canonize_precondition(precondition)
        with ExpressionDefinitionContext(*vars, domain=self.domain, scope=scope, is_effect_definition=True).as_default():
            effect = _canonize_effect(effect)
        self.domain.define_axiom(name, vars, precondition, effect, **kwargs)

    @inline_args
    def derived_definition(self, signature, expr):
        name, args, kwargs = signature
        expr = expr

        return_type = kwargs.pop('return_type', BOOL)
        with ExpressionDefinitionContext(*args, domain=self.domain, scope=f"derived::{name}").as_default():
            if return_type is AUTO:
                expr = expr.compose()
                assert isinstance(expr, (E.VariableExpression, E.ValueOutputExpression))
                return_type = expr.return_type
            else:
                expr = expr.compose(return_type)
        self.domain.define_derived(name, args, return_type, expr=expr, **kwargs)

    @inline_args
    def derived_signature1(self, name, *args):
        name, kwargs = name
        return name, args, kwargs

    @inline_args
    def derived_signature2(self, name, *args):
        name, kwargs = name
        assert 'return_type' not in kwargs, 'Return type cannot be set using decorators.'
        kwargs['return_type'] = args[-1]
        return name, args[:-1], kwargs

    @inline_args
    def derived_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def generator_definition(self, name, parameters, certifies, context, generates):
        name, kwargs = name
        parameters = tuple(parameters.children)
        certifies = certifies.children[0]
        context = context.children[0]
        generates = generates.children[0]

        ctx = ExpressionDefinitionContext(*parameters, domain=self.domain, scope=f"generator::{name}")
        with ctx.as_default():
            certifies = certifies.compose(BOOL)
            assert context.name == 'and'
            context = [_compose(ctx, c) for c in context.arguments]
            assert generates.name == 'and'
            generates = [_compose(ctx, c) for c in generates.arguments]

        self.domain.define_generator(name, parameters, certifies, context, generates, **kwargs)

    @inline_args
    def generator_name(self, name, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return name.value, kwargs

    @inline_args
    def object_definition(self, constant):
        self.problem.add_object(constant.name, constant.typename)

    @inline_args
    def init_definition_item(self, function_call):
        if function_call.name not in self.domain.predicates:
            if self.ignore_unknown_predicates:
                if function_call.name not in self.ignored_predicates:
                    logger.warning(f"Unknown predicate: {function_call.name}.")
                    self.ignored_predicates.add(function_call.name)
            else:
                raise ValueError(f"Unknown predicate: {function_call.name}.")
            return
        self.problem.add_predicate(function_call.compose())

    @inline_args
    def goal_definition(self, function_call):
        self.problem.set_goal(function_call.compose())

    @inline_args
    def variable(self, name) -> Variable:
        return Variable(name.value)

    @inline_args
    def typedvariable(self, name, typename):
        # name is of type `Variable`.
        if typename is AUTO:
            return Variable(name.name, AUTO)
        return Variable(name.name, self.domain.types[typename])

    @inline_args
    def constant(self, name) -> StringConstant:
        assert self.allow_object_constants
        return StringConstant(name.value)

    @inline_args
    def typedconstant(self, name, typename):
        return StringConstant(name.name, self.domain.types[typename])

    @inline_args
    def bool(self, v):
        return v.value == 'true'

    @inline_args
    def int(self, v):
        return int(v.value)

    @inline_args
    def float(self, v):
        return float(v.value)

    @inline_args
    def string(self, v):
        return v.value[1:-1]

    @inline_args
    def list(self, *args):
        return list(args)

    @inline_args
    def decorator_k(self, k):
        return k.value

    @inline_args
    def decorator_v(self, v):
        return v

    @inline_args
    def decorator_kwarg(self, k, v):
        return (k, v)

    def decorator_kwargs(self, args):
        return {k: v for k, v in args}

    @inline_args
    def slot(self, _, name, kwargs=None):
        return _Slot(name.children[0].value, kwargs)

    @inline_args
    def function_name(self, name, kwargs=None):
        return name, kwargs

    @inline_args
    def method_name(self, predicate_name, _, method_name):
        return _MethodName(predicate_name, method_name), None

    @inline_args
    def function_call(self, name, *args):
        name, kwargs = name
        assert isinstance(name, (str, _MethodName, _Slot))
        return _FunctionApplicationImm(name, args, kwargs=kwargs)

    @inline_args
    def simple_function_call(self, name, *args):
        name, kwargs = name
        return _FunctionApplicationImm(name.value, args, kwargs=kwargs)

    @inline_args
    def pm_function_call(self, pm_sign, function_call):
        if pm_sign.value == '+':
            return function_call
        else:
            return _FunctionApplicationImm('not', [function_call])

    @inline_args
    def quantified_function_call(self, quantifier, variable, expr):
        return _QuantifierApplicationImm(quantifier, variable, expr)


class _ParameterListWrapper(collections.namedtuple('_ParameterListWrapper', 'parameters')):
    pass


class _PreconditionWrapper(collections.namedtuple('_PreconditionWrapper', 'precondition')):
    pass


class _EffectWrapper(collections.namedtuple('_EffectWrapper', 'effect')):
    pass


class _FunctionApplicationImm(object):
    def __init__(self, name, arguments, kwargs=None):
        self.name = name
        self.arguments = arguments
        self.kwargs = kwargs

        if self.kwargs is None:
            self.kwargs = dict()

    def __str__(self):
        arguments_str = ', '.join([str(arg) for arg in self.arguments])
        return f'IMM::{self.name}({arguments_str})'

    __repr__ = jacinle.repr_from_str

    @_log_function
    def compose(self, expect_value_type: Optional[Union[ObjectType, ValueType]] = None):
        ctx = get_definition_context()
        if isinstance(self.name, _Slot):
            assert ctx.scope is not None, 'Cannot define slots inside anonymous actino/axioms.'

            name = ctx.scope + '::' + self.name.name
            arguments = self._compose_arguments(ctx, self.arguments)
            argument_types = [arg.variable if isinstance(arg, E.VariableExpression) else arg.return_type for arg in arguments]
            return_type = self.name.kwargs.pop('return_type', None)
            if return_type is None:
                assert expect_value_type is not None, f'Cannot infer return type for function {name}; please specify by [return_type=Type]'
                return_type = expect_value_type
            else:
                if expect_value_type is not None:
                    assert return_type == expect_value_type, f'Return type mismatch for function {name}: expect {expect_value_type}, got {return_type}.'
            function_def = ctx.domain.declare_external_function(name, argument_types, return_type, kwargs=self.name.kwargs)
            return E.ExternalFunctionApplication(function_def, *arguments)
        elif isinstance(self.name, _MethodName):
            assert self.name.predicate_name in ctx.domain.predicates, 'Unkwown feature: {}.'.format(self.name.predicate_name)
            predicate_def = ctx.domain.predicates[self.name.predicate_name]

            if self.name.method_name == 'equal':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'assign':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-select':
                nr_index_arguments = len(self.arguments) - 1
            elif self.name.method_name == 'cond-assign':
                nr_index_arguments = len(self.arguments) - 2
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))

            arguments = self._compose_arguments(ctx, self.arguments[:nr_index_arguments], predicate_def.arguments, is_variable_list=True)
            with ctx.mark_is_effect_definition(False):
                value = self._compose_arguments(ctx, [self.arguments[-1]], predicate_def.return_type.assignment_type())[0]

            feature = E.PredicateApplication(predicate_def, *arguments)

            if self.name.method_name == 'equal':
                return E.PredicateEqualOp(feature, value)
            elif self.name.method_name == 'assign':
                return E.AssignOp(feature, value)
            elif self.name.method_name == 'cond-select':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-1]], BOOL)[0]
                return E.ConditionalSelectOp(feature, condition)
            elif self.name.method_name == 'cond-assign':
                with ctx.mark_is_effect_definition(False):
                    condition = self._compose_arguments(ctx, [self.arguments[-2]], BOOL)[0]
                return E.ConditionalAssignOp(feature, value, condition)
            else:
                raise NameError('Unknown method name: {}.'.format(self.name.method_name))
        elif self.name == 'and':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.AndOp(*arguments)
        elif self.name == 'or':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.OrOp(*arguments)
        elif self.name == 'not':
            arguments = [arg.compose(expect_value_type) for arg in self.arguments]
            return E.NotOp(*arguments)
        elif self.name == 'equal':
            assert len(self.arguments) == 2, 'FeatureEqualOp takes two arguments, got: {}.'.format(len(self.arguments))
            feature = self.arguments[0]
            feature = _compose(ctx, feature, None)
            value = self.arguments[1]
            value = _compose(ctx, value, feature.return_type.assignment_type())
            return E.PredicateEqualOp(feature, value)
        elif self.name == 'assign':
            assert len(self.arguments) == 2, 'AssignOp takes two arguments, got: {}.'.format(len(self.arguments))
            assert isinstance(self.arguments[0], _FunctionApplicationImm)
            feature = self.arguments[0].compose(None)
            assert isinstance(feature, E.PredicateApplication)
            with ctx.mark_is_effect_definition(False):
                value = _compose(ctx, arguments[1], feature.return_type.assignment_type())
            return E.AssignOp(feature, value)
        else:  # the name is a predicate name.
            if self.name in ctx.domain.predicates or ctx.allow_auto_predicate_def:
                if self.name not in ctx.domain.predicates:
                    arguments: List[Union[E.ValueOutputExpression, E.VariableExpression]] = self._compose_arguments(ctx, self.arguments, None)
                    for arg in arguments:
                        assert isinstance(arg, (E.ValueOutputExpression, E.VariableExpression)), f'Cannot infer argument type for predicate {self.name}.'
                    argument_types = [arg.return_type for arg in arguments]
                    argument_defs = [Variable(f'arg{i}', arg_type) for i, arg_type in enumerate(argument_types)]
                    self.kwargs.setdefault('state', False)
                    self.kwargs.setdefault('observation', False)
                    generators = self.kwargs.pop('generators', None)
                    predicate_def = ctx.domain.define_predicate(self.name, argument_defs, BOOL, **self.kwargs)
                    rv = E.PredicateApplication(predicate_def, *arguments)
                    logger.info(f'Auto-defined predicate {self.name} with arguments {argument_defs} and return type {BOOL}.')

                    # create generators inline
                    if generators is not None:
                        generators: List[str]
                        for i, target_variable_name in enumerate(generators):
                            assert target_variable_name.startswith('?')
                            parameters, context, generates = _canonize_inline_generator_def(ctx, target_variable_name, arguments)
                            generator_name = f'gen-{self.name}-{target_variable_name[1:]}' if len(generators) > 1 else f'gen-{self.name}'
                            ctx.domain.define_generator(generator_name, parameters=parameters, certifies=rv, context=context, generates=generates)
                    return rv
                else:
                    assert len(self.kwargs) == 0, 'Cannot specify decorators for non-auto predicate definition.'
                    predicate_def = ctx.domain.predicates[self.name]
                    arguments = self._compose_arguments(ctx, self.arguments, predicate_def.arguments, is_variable_list=True)
                    return E.PredicateApplication(predicate_def, *arguments)
            else:
                raise ValueError('Unknown function: {}.'.format(self.name))

    def _compose_arguments(self, ctx, arguments, expect_value_type=None, is_variable_list: bool = False) -> List[E.Expression]:
        if isinstance(expect_value_type, (tuple, list)):
            assert len(expect_value_type) == len(arguments), 'Mismatched number of arguments: expect {}, got {}. Expression: {}.'.format(len(expect_value_type), len(arguments), self)

            if is_variable_list:
                output_list = list()
                for arg, var in zip(arguments, expect_value_type):
                    rv = _compose(ctx, arg, var.dtype if var.dtype is not AUTO else None)
                    if var.dtype is AUTO:
                        var.dtype = rv.return_type
                    output_list.append(rv)
                return output_list

            return [_compose(ctx, arg, evt) for arg, evt in zip(arguments, expect_value_type)]
        return [_compose(ctx, arg, expect_value_type) for arg in arguments]


def _compose(ctx, arg, evt=None):
    if isinstance(arg, Variable):
        return ctx[arg.name]
    elif isinstance(arg, StringConstant):
        return E.ObjectConstantExpression(arg)
    else:
        return arg.compose(evt)


class _Slot(object):
    def __init__(self, name, kwargs=None):
        self.scope = None
        self.name = name
        self.kwargs = kwargs

        if self.kwargs is None:
            self.kwargs = dict()

    def set_scope(self, scope):
        self.scope = scope

    def __str__(self):
        kwargs = ', '.join([f'{k}={v}' for k, v in self.kwargs.items()])
        return f'??{self.name}[{kwargs}]'


class _MethodName(object):
    def __init__(self, predicate_name, method_name):
        self.predicate_name = predicate_name
        self.method_name = method_name

    def __str__(self):
        return f'{self.predicate_name}::{self.method_name}'


class _QuantifierApplicationImm(object):
    def __init__(self, quantifier, darg: Variable, expr: _FunctionApplicationImm):
        self.quantifier = quantifier
        self.darg = darg
        self.expr = expr

    def __str__(self):
        return f'QIMM::{self.quantifier}({self.darg}: {self.expr})'

    @_log_function
    def compose(self, expect_value_type: Optional[ValueType] = None):
        ctx = get_definition_context()

        with ctx.new_arguments(self.darg):
            if ctx.is_effect_definition:
                expr = _canonize_effect(self.expr)
            else:
                expr = self.expr.compose(expect_value_type)

        if self.quantifier in ('foreach', 'forall') and ctx.is_effect_definition:
            outputs = list()
            for e in expr:
                assert isinstance(e, Effect)
                outputs.append(E.DeicticAssignOp(self.darg, e.assign_expr))
            return outputs
        if self.quantifier == 'foreach':
            assert E.is_value_output_expression(expr)
            return E.DeicticSelectOp(self.darg, expr)

        assert E.is_value_output_expression(expr)
        return E.QuantificationOp(E.QuantifierType.from_string(self.quantifier), self.darg, expr)


@_log_function
def _canonize_precondition(precondition: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(precondition, _FunctionApplicationImm) and precondition.name == 'and':
        return list(itertools.chain(*[_canonize_precondition(pre) for pre in precondition.arguments]))
    return [Precondition(precondition.compose(BOOL))]


@_log_function
def _canonize_effect(effect: Union[_FunctionApplicationImm, _QuantifierApplicationImm]):
    if isinstance(effect, _QuantifierApplicationImm):
        effect = effect.compose()
        if isinstance(effect, list):
            effect = [Effect(e) for e in effect]
    else:
        assert isinstance(effect, _FunctionApplicationImm)

        if effect.name == 'and':
            return list(itertools.chain(*[_canonize_effect(eff) for eff in effect.arguments]))

        if isinstance(effect.name, _MethodName):
            effect = effect.compose()
        elif effect.name == 'assign':
            effect = effect.compose()
        elif effect.name == 'not':
            assert len(effect.arguments) == 1, 'NotOp only takes 1 argument, got {}.'.format(len(effect.arguments))
            feat = effect.arguments[0].compose()
            assert feat.return_type == BOOL
            effect = E.AssignOp(feat, E.ConstantExpression.FALSE)
        elif effect.name == 'when':
            assert len(effect.arguments) == 2, 'WhenOp takes two arguments, got: {}.'.format(len(effect.arguments))
            condition = effect.arguments[0].compose(BOOL)
            if effect.arguments[1].name == 'and':
                inner_effects = effect.arguments[1].arguments
            else:
                inner_effects = [effect.arguments[1]]
            inner_effects = list(itertools.chain(*[_canonize_effect(arg) for arg in inner_effects]))
            effect = list()
            for e in inner_effects:
                assert isinstance(e.assign_expr, E.AssignOp)
                effect.append(Effect(E.ConditionalAssignOp(e.assign_expr.predicate, e.assign_expr.value, condition)))
            return effect
        else:
            feat = effect.compose()
            assert isinstance(feat, E.PredicateApplication) and feat.return_type == BOOL
            effect = E.AssignOp(feat, E.ConstantExpression.TRUE)

    if isinstance(effect, list):
        return effect

    assert isinstance(effect, E.VariableAssignmentExpression)
    return [Effect(effect)]


def _canonize_inline_generator_def(ctx: ExpressionDefinitionContext, variable_name: str, arguments: List[Union[E.VariableExpression, E.ValueOutputExpression]]):
    parameters, context, generates = list(), list(), list()
    used_parameters = set()

    for arg in arguments:
        if isinstance(arg, E.VariableExpression):
            if arg.name not in used_parameters:
                used_parameters.add(arg.name)
                parameters.append(arg.variable)
            if arg.name == variable_name:
                generates.append(arg)
            else:
                context.append(arg)
        else:
            context.append(arg)
            assert isinstance(arg, E.ValueOutputExpression)
            for sub_expr in E.iter_exprs(arg):
                if isinstance(sub_expr, E.VariableExpression):
                    if sub_expr.name not in used_parameters:
                        used_parameters.add(sub_expr.name)
                        parameters.append(sub_expr.variable)

    assert len(generates) == 1, f'Generator must generate exactly one variable, got {len(generates)}.'
    return parameters, context, generates


def _canonize_inline_generator_def_predicate(domain: Domain, variable_name: str, predicate_def: PredicateDef):
    parameters = predicate_def.arguments
    context, generates = list(), list()

    if predicate_def.return_type != BOOL:
        for arg in parameters:
            assert arg.name != '?rv', 'Arguments cannot be named ?rv.'
        parameters = parameters + (Variable('?rv', predicate_def.return_type),)

    ctx = ExpressionDefinitionContext(*parameters, domain=domain)
    with ctx.as_default():
        for arg in predicate_def.arguments:
            assert isinstance(arg, Variable)
            if arg.name == variable_name:
                generates.append(ctx[arg.name])
            else:
                context.append(ctx[arg.name])
        certifies = E.PredicateApplication(predicate_def, *[ctx[arg.name] for arg in predicate_def.arguments])
        if predicate_def.return_type != BOOL:
            context.append(ctx['?rv'])
            certifies = E.PredicateEqualOp(certifies, ctx['?rv'])
    return parameters, certifies, context, generates

