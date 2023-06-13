#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : expr.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/28/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import itertools
import contextlib
from abc import ABC
from typing import Optional, Union, Iterable, Tuple, Sequence, List, Set, Mapping, Dict, Callable, ForwardRef

import torch
import jacinle
import jactorch
from jacinle.utils.enum import JacEnum
from jacinle.utils.printing import indent_text
from jacinle.utils.defaults import wrap_custom_as_default, gen_get_default
from .value import ObjectType, ValueType, NamedValueTypeSlot, NamedValueType, PyObjValueType, BOOL, INT64, FLOAT32, RUNTIME_BINDING, is_intrinsically_quantized
from .value import QINDEX, Variable, StringConstant, QuantizedTensorValue, Value
from .optimistic import OPTIM_MAGIC_NUMBER, is_optimistic_value, OptimisticConstraint, EqualOptimisticConstraint, OptimisticValueContext, cvt_opt_value
from .state import StateLike, SingleStateLike

__all__ = [
    'FunctionArgumentType', 'FunctionReturnType', 'FunctionDef', 'PredicateDef',
    'ExpressionDefinitionContext', 'get_definition_context',
    'BoundedVariablesDict', 'BoundedVariablesDictCompatibleKeyType', 'BoundedVariablesDictCompatibleValueType', 'BoundedVariablesDictCompatible', 'compose_bvdict', 'compose_bvdict_args',
    'ExpressionExecutionContext', 'get_execution_context',
    'Expression', 'ExpressionCompatible', 'cvt_expression', 'cvt_expression_list',
    'VariableExpression', 'ObjectConstantExpression', 'ConstantExpression',
    'ValueOutputExpression', 'is_value_output_expression',
    'FunctionApplicationError', 'FunctionArgumentValueType', 'forward_args', 'has_partial_execution_value', 'expand_argument_values',
    'FunctionApplication', 'PredicateApplication', 'ExternalFunctionApplication', 'is_predicate_application', 'is_external_function_application',
    'DeicticSelectOp',
    'BoolOp', 'BoolOpType', 'AndOp', 'OrOp', 'NotOp', 'is_constant_bool_expr', 'is_simple_bool', 'split_simple_bool', 'get_simple_bool_def', 'is_and_expr', 'is_or_expr', 'is_not_expr',
    'QuantificationOp', 'QuantifierType', 'ForallOp', 'ExistsOp', 'is_forall_expr', 'is_exists_expr',
    'PredicateEqualOp',
    'VariableAssignmentExpression', 'is_variable_assignment_expression',
    'AssignOp', 'ConditionalAssignOp', 'DeicticAssignOp',
    'flatten_expression', 'iter_exprs', 'get_used_state_variables'
]


Domain = ForwardRef('Domain')
Session = ForwardRef('Session')

FunctionArgumentType = Union[ObjectType, ValueType, Variable]
FunctionReturnType = Union[ValueType, Tuple[ValueType, ...]]


class FunctionDef(object):
    """Generic function definition."""

    def __init__(self, name: str, arguments: Sequence[FunctionArgumentType], return_type: Optional[FunctionReturnType] = None, generator_placeholder: bool = False, **kwargs):
        self.name = name
        self.arguments = tuple(arguments)
        self.return_type = return_type if return_type is not None else BOOL
        self.kwargs = kwargs

        self.is_cacheable = self._guess_is_cacheable()
        self.is_derived = False
        self.is_static = False
        self.is_generator_placeholder = generator_placeholder

        self._check_arguments()

    def _check_arguments(self):
        pass

    def _guess_is_cacheable(self) -> bool:
        """Return whether the function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it can be statically evaluated."""
        for arg_def in self.arguments:
            if isinstance(arg_def, ValueType) or (isinstance(arg_def, Variable) and isinstance(arg_def.dtype, ValueType)):
                return False
        return True

    def __str__(self) -> str:
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.name}({arguments}) -> {self.return_type}'

    __repr__ = jacinle.repr_from_str


class PredicateDef(FunctionDef):
    """Predicate definition.

    A predicate contains the following properties.

    - name: the name of the predicate.
    - return_type: the output type of the predicate.
    - arguments: a tuple of arguments of the predicate. Each argument can either be an object type or a value type.
    - expr: the expression of the predicate. If not none, the predicate is essentially a derived predicate.

    - is_observation_variable: whether the predicate is an observation variable.
    - is_state_variable: whether the predicate is a state variable.
    - is_generator_placeholder: whether the predicate is a generator placeholder.

    - is_cacheable: whether the grounded of the predicate can be "cahced" in the state represenation.
    - is_derived: a predicate is derived if its expression is not none.
    - is_static: whether the grounded of the predicate is static (i.e., it will never change).

      This flag is useful for "generator_placeholder" predicates.
    """

    def __init__(
        self, name: str, arguments: Sequence[FunctionArgumentType], return_type: Optional[FunctionReturnType] = None,
        expr: Optional['ValueOutputExpression'] = None,
        observation: Optional[bool] = None,
        state: Optional[bool] = None,
        generator_placeholder: bool = False
    ):
        super().__init__(name, arguments, return_type, generator_placeholder)
        self.expr = expr

        self.is_derived = expr is not None
        # NB(Jiayuan Mao @ 09/03): In __init__, we always mark the predicate as non-static. This property will be set later when the entire domain has been loaded.
        self.is_static = False

        self.is_observation_variable = observation if observation is not None else self.is_cacheable
        self.is_state_variable = state if state is not None else self.is_cacheable

        from .ao_discretization import AOFeatureCodebook
        self.ao_discretization: Optional[AOFeatureCodebook] = None  # for AODiscretization

        self._check_flags_sanity()

    def mark_static(self, flag: bool = True):
        """Mark a predicate as static (i.e., its grounded value will never change)."""
        self.is_static = flag

    def _guess_is_cacheable(self) -> bool:
        """Return whether the function can be cached. Specifically, if it contains only "ObjectTypes" as arguments, it can be statically evaluated."""
        for arg_def in self.arguments:
            if isinstance(arg_def.dtype, ValueType):
                return False
        return True

    def _check_arguments(self):
        for arg_def in self.arguments:
            assert isinstance(arg_def, Variable)

    def _check_flags_sanity(self):
        if self.is_observation_variable:
            assert self.is_cacheable and not self.is_derived
        if self.is_state_variable:
            assert self.is_cacheable
            if self.is_derived:
                for predicate_def in get_used_state_variables(self.expr):
                    assert predicate_def.is_observation_variable and not predicate_def.is_state_variable
            else:
                assert self.is_cacheable
        if self.is_generator_placeholder:
            assert not self.is_derived and not self.is_state_variable and not self.is_observation_variable

    def __call__(self, *arguments):
        return PredicateApplication(self, *cvt_expression_list(arguments))

    def __str__(self) -> str:
        flags = list()

        if self.is_observation_variable:
            flags.append('observation')
        if self.is_state_variable:
            flags.append('state')
        if self.is_generator_placeholder:
            flags.append('gen')
        if self.is_cacheable:
            flags.append('cacheable')
        if self.is_static:
            flags.append('static')
        flags_string = '[' + ', '.join(flags) + ']' if len(flags) > 0 else ''
        arguments = ', '.join([str(arg) for arg in self.arguments])
        fmt = f'{self.name}{flags_string}({arguments}) -> {self.return_type}'
        if self.expr is not None:
            fmt += ' {\n'
            fmt += '  ' + str(self.expr)
            fmt += '\n}'
        return fmt

    __repr__ = jacinle.repr_from_str


class ExpressionDefinitionContext(object):
    """The context for defining a PDSketch expression. During definition, it will only keep track
    of the type of the object."""

    def __init__(self, *variables: Variable, domain: Optional[Domain] = None, scope: Optional[str] = None, is_effect_definition: bool = False, allow_auto_predicate_def: bool = True):
        """Initialize the context.

        Args:
            variables: the variables in the expression.
            domain: the domain of the expression.
            scope: the current definition scope (e.g., in a function). This variable will be used to generate unique names for the functions.
            is_effect_definition: whether the expression is defined in an effect of an operator.
            allow_auto_predicate_def: whether to enable automatic predicate definition.
        """

        self.variables = list(variables)
        self.variable_name2obj = {v.name: v for v in self.variables}
        self.domain = domain
        self.scope = scope
        self.name_counter = itertools.count()
        self.is_effect_definition_stack = [is_effect_definition]
        self.allow_auto_predicate_def = allow_auto_predicate_def

    def __getitem__(self, variable_name) -> 'VariableExpression':
        if variable_name == '??':
            return VariableExpression(Variable('??', RUNTIME_BINDING))
        if variable_name not in self.variable_name2obj:
            raise ValueError('Unknown variable: {}; available variables: {}.'.format(variable_name, self.variables))
        return VariableExpression(self.variable_name2obj[variable_name])

    def generate_random_named_variable(self, dtype) -> Variable:
        """Generate a variable expression with a random name.

        This utility is useful in "flatten_expression". See the doc for that
        function for details.
        """
        name = '_t' + str(next(self.name_counter))
        return Variable(name, dtype)

    @contextlib.contextmanager
    def new_arguments(self, *args: Variable):
        """Adding a list of new variables."""
        for arg in args:
            assert arg.name not in self.variable_name2obj, 'Variable name {} already exists.'.format(arg.name)
            self.variables.append(arg)
            self.variable_name2obj[arg.name] = arg
        yield self
        for arg in reversed(args):
            self.variables.pop()
            del self.variable_name2obj[arg.name]

    @contextlib.contextmanager
    def mark_is_effect_definition(self, is_effect_definition: bool):
        self.is_effect_definition_stack.append(is_effect_definition)
        yield self
        self.is_effect_definition_stack.pop()

    @property
    def is_effect_definition(self) -> bool:
        return self.is_effect_definition_stack[-1]

    @wrap_custom_as_default
    def as_default(self):
        yield self


get_definition_context: Callable[[], ExpressionDefinitionContext] = gen_get_default(ExpressionDefinitionContext)


BoundedVariablesDict = Dict[str, Dict[str, Union[str, int, slice, Value]]]
BoundedVariablesDictCompatibleKeyType = Union[str, Variable]
BoundedVariablesDictCompatibleValueType = Union[str, int, slice, bool, float, torch.Tensor, Value]
BoundedVariablesDictCompatible = Union[
    None, Sequence[Variable],
    Dict[BoundedVariablesDictCompatibleKeyType, BoundedVariablesDictCompatibleValueType],
    BoundedVariablesDict
]


def compose_bvdict(input_dict: BoundedVariablesDictCompatible, state: Optional[StateLike] = None, session: Optional[Session] = None) -> BoundedVariablesDict:
    if input_dict is None:
        return dict()

    if isinstance(input_dict, dict):
        if len(input_dict) == 0:
            return input_dict

        sample_value = next(iter(input_dict.values()))
        if isinstance(sample_value, dict):
            return input_dict

        from .session import get_session
        session = get_session(session)

        output_dict = dict()
        for var, value in input_dict.items():
            if isinstance(var, Variable):
                if isinstance(var.dtype, ObjectType):
                    if isinstance(value, int):
                        output_dict.setdefault(var.typename, dict()).setdefault(var.name, value)
                    elif isinstance(value, str):
                        assert state is not None
                        value = state.get_typed_index(value)
                        output_dict.setdefault(var.typename, dict()).setdefault(var.name, value)
                    else:
                        raise TypeError(f'Invalid value type for variable {var}: {type(value)}.')
                elif isinstance(var.dtype, PyObjValueType):
                    if not isinstance(value, Value):
                        value = session.pyobj_store.make_batched_value(var.dtype.typename, value)
                    typename = var.dtype.typename
                    output_dict.setdefault(typename, {})[var.name] = value
                elif isinstance(var.dtype, NamedValueType):
                    if isinstance(value, NamedValueTypeSlot):
                        value = Value(var.dtype, [], torch.tensor(OPTIM_MAGIC_NUMBER, dtype=torch.int64))
                    elif isinstance(value, (bool, int, float, torch.Tensor)):
                        value = Value(var.dtype, [], torch.tensor(value, dtype=var.dtype.dtype), quantized=isinstance(value, (bool, int)))
                    elif isinstance(value, Value):
                        pass
                    else:
                        raise TypeError(f'Invalid value type for variable {var}: {type(value)}.')
                    output_dict.setdefault(var.dtype.typename, {})[var.name] = value
                else:
                    raise TypeError(f'Invalid variable type: {var.dtype}.')
            elif isinstance(var, str) and isinstance(value, str):
                assert state is not None
                typename, value_index = state.get_typename(value), state.get_typed_index(value)
                output_dict.setdefault(typename, dict()).setdefault(var, value_index)
            else:
                raise TypeError('Invalid input type: {}.'.format(type(input_dict)))
        return output_dict
    else:
        # The input dict is a list of variables.
        assert isinstance(input_dict, (list, tuple))
        output_dict = dict()
        for var in input_dict:
            assert isinstance(var, Variable)
            output_dict.setdefault(var.typename, dict()).setdefault(var.name, QINDEX)
        return output_dict


def compose_bvdict_args(arguments_def: Sequence[Variable], arguments: Sequence[BoundedVariablesDictCompatibleValueType], state: Optional[StateLike] = None, session: Optional[Session] = None) -> BoundedVariablesDict:
    return compose_bvdict(dict(zip(arguments_def, arguments)), state=state, session=session)


class ExpressionExecutionContext(object):
    def __init__(self, session: Session, state: Optional[StateLike] = None, bounded_variables: Optional[BoundedVariablesDict] = None, optimistic_context: Optional[OptimisticValueContext] = None):
        """The expression execution context.

        Args:
            session (Session): The session.
            state (State): The input state.
            bounded_variables (Mapping[str, Mapping[str, int]]): A mapping for bounded variables. Stored in the following form:

                ```
                {
                    'item': {'?obj1': 0}
                    'location': {'?loc': 1}
                }
                ```

                The key to the outer mapping is the typename. The key to the inner mapping is the variable type.
                The values are "typed_index".
        """

        # NB(Jiayuan Mao @ 08/04): Sanity check at 08/04 because we have migrated to Session-based.
        from .session import Session
        assert session is None or isinstance(session, Session)

        self.session = session
        self.state = state
        self.bounded_variables = bounded_variables if bounded_variables is not None else dict()
        self.optimistic_context = optimistic_context

    def __str__(self) -> str:
        fmt = f'ExpressionExecutionContext(domain={self.domain.name},\n'
        fmt += f'  bounded_variables={indent_text(str(self.bounded_variables)).lstrip()},\n'
        fmt += f'  optimistic_context={indent_text(str(self.optimistic_context)).lstrip()}\n'
        fmt += ')'
        return fmt

    __repr__ = jacinle.repr_from_str

    @property
    def domain(self):
        return self.session.domain

    @property
    def value_quantizer(self):
        return self.session.value_quantizer

    @property
    def is_optimistic_execution(self) -> bool:
        return self.optimistic_context is not None

    def get_external_function(self, name, default='ERROR'):
        if self.domain.has_external_function(name):
            return self.domain.get_external_function(name)
        if default == 'ERROR':
            raise ValueError(f'External function {name} not found.')
        return default

    @contextlib.contextmanager
    def new_bounded_variables(self, variable2index: Mapping[Variable, Union[int, slice]]):
        for arg, index in variable2index.items():
            assert arg.dtype.typename not in self.bounded_variables or arg.name not in self.bounded_variables[arg.dtype.typename], 'Variable name {} already exists.'.format(arg.name)
            if arg.dtype.typename not in self.bounded_variables:
                self.bounded_variables[arg.dtype.typename] = dict()
            self.bounded_variables[arg.dtype.typename][arg.name] = index
        yield self
        for arg, index in variable2index.items():
            del self.bounded_variables[arg.dtype.typename][arg.name]

    def get_bounded_variable(self, variable: Variable) -> Union[int, slice, Value]:
        if variable.name == '??':
            return QINDEX
        return self.bounded_variables[variable.dtype.typename][variable.name]

    @wrap_custom_as_default
    def as_default(self):
        yield self


get_execution_context: Callable[[], ExpressionExecutionContext] = gen_get_default(ExpressionExecutionContext)


class Expression(ABC):
    # @jacinle.log_function
    def forward(self, ctx: ExpressionExecutionContext) -> Optional[Value]:
        return self._forward(ctx)

    def _forward(self, ctx: ExpressionExecutionContext) -> Optional[Union[int, slice, Value]]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


ExpressionCompatible = Union[Expression, Variable, str, StringConstant, bool, int, float, torch.Tensor, Value]


def cvt_expression(expr: ExpressionCompatible) -> Expression:
    if isinstance(expr, Expression):
        return expr
    elif isinstance(expr, Variable):
        return VariableExpression(expr)
    elif isinstance(expr, str):
        return ObjectConstantExpression(StringConstant(expr))
    elif isinstance(expr, StringConstant):
        return ObjectConstantExpression(expr)
    elif isinstance(expr, bool):
        return ConstantExpression(BOOL, torch.tensor(int(expr), dtype=torch.int64), quantized=True)
    elif isinstance(expr, int):
        return ConstantExpression(INT64, torch.tensor(expr, dtype=torch.int64), quantized=True)
    elif isinstance(expr, float):
        return ConstantExpression(FLOAT32, torch.tensor(expr, dtype=torch.float32), quantized=False)
    elif isinstance(expr, torch.Tensor):
        assert expr.dtype in [torch.int64, torch.float32]
        expr_dtype = INT64 if expr.dtype == torch.int64 else FLOAT32
        return ConstantExpression(expr_dtype, expr, quantized=False)
    elif isinstance(expr, Value):
        return ConstantExpression(expr.dtype, expr, quantized=False)
    else:
        raise TypeError(f'Non-compatible expression type {type(expr)} for expression "{expr}".')


def cvt_expression_list(arguments: Sequence[Union[Expression, Variable, str, StringConstant, bool, int, float, torch.Tensor, Value]]) -> List[Expression]:
    arguments = [cvt_expression(arg) for arg in arguments]
    return arguments


class VariableExpression(Expression):
    def __init__(self, variable: Variable):
        self.variable = variable

    @property
    def name(self):
        return self.variable.name

    @property
    def type(self):
        return self.variable.dtype

    @property
    def return_type(self):
        return self.variable.dtype

    def _forward(self, ctx: ExpressionExecutionContext) -> Union[int, slice, torch.Tensor, Value]:
        return ctx.get_bounded_variable(self.variable)

    def __str__(self) -> str:
        return f'V::{self.name}'


class ObjectConstantExpression(Expression):
    def __init__(self, constant: StringConstant):
        self.constant = constant

    @property
    def name(self):
        return self.constant.name

    @property
    def type(self):
        return self.constant.dtype if self.constant.dtype is not None else RUNTIME_BINDING

    @property
    def return_type(self):
        return self.constant.dtype if self.constant.dtype is not None else RUNTIME_BINDING

    def _forward(self, ctx: ExpressionExecutionContext) -> Union[int, torch.Tensor]:
        if ctx.state.batch_dims > 0:  # Batched state.
            return torch.tensor(ctx.state.get_typed_index(self.name), dtype=torch.int64)
        assert isinstance(ctx.state, SingleStateLike)
        assert self.name in ctx.state.object_names, f'Object {self.name} does not exist.'
        return ctx.state.get_typed_index(self.name)

    def __str__(self):
        return f'OBJ::{self.name}'


class ValueOutputExpression(Expression):
    @property
    def return_type(self):
        raise NotImplementedError()

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


def is_value_output_expression(expr: Expression) -> bool:
    return isinstance(expr, ValueOutputExpression)


class ConstantExpression(ValueOutputExpression):
    def __init__(self, dtype: ValueType, value: Union[bool, int, float, torch.Tensor, Value], quantized=False):
        if isinstance(value, Value):
            self.value = value
        else:
            self.value = Value(dtype, [], value, quantized=quantized)

    @property
    def return_type(self):
        return self.value.dtype

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        return self.value

    @classmethod
    def true(cls):
        return cls(BOOL, torch.tensor(1, dtype=torch.int64), quantized=True)

    @classmethod
    def false(cls):
        return cls(BOOL, torch.tensor(0, dtype=torch.int64), quantized=True)

    @classmethod
    def int64(cls, value):
        return cls(INT64, torch.tensor(value, dtype=torch.int64), quantized=True)

    def __str__(self):
        return str(self.value)


ConstantExpression.TRUE = ConstantExpression.true()
ConstantExpression.FALSE = ConstantExpression.false()


class FunctionApplicationError(Exception):
    def __init__(self, index, expect, got):
        msg = f'Argument #{index} type does not match: expect {expect}, got {got}.'
        super().__init__(msg)

        self.index = index
        self.expect = expect
        self.got = got


FunctionArgumentValueType = Union[int, str, slice, Value]


def forward_args(ctx: ExpressionExecutionContext, *args: Union[Value, Expression, int], force_tuple: bool = False, keep_name: bool = False) -> Union[Tuple[FunctionArgumentValueType, ...], FunctionArgumentValueType]:
    rv = list()
    for arg in args:
        if isinstance(arg, Value):
            rv.append(arg)
        elif isinstance(arg, Expression):
            if keep_name:
                if isinstance(arg, VariableExpression):
                    value = arg.forward(ctx)
                    if isinstance(value, Value):
                        rv.append(value)
                    elif value == QINDEX:
                        rv.append(StringConstant(QINDEX, arg.variable.dtype))
                    else:
                        assert isinstance(value, int)
                        rv.append(StringConstant(ctx.state.object_type2names[arg.variable.typename][value], arg.variable.dtype))
                elif isinstance(arg, ObjectConstantExpression):
                    rv.append(arg.constant)
                else:
                    rv.append(arg.forward(ctx))
            else:
                rv.append(arg.forward(ctx))
        elif isinstance(arg, int):  # object index.
            rv.append(arg)
        else:
            raise TypeError('Unknown argument type: {}.'.format(type(arg)))
    if len(rv) == 1 and not force_tuple:
        return rv[0]
    return tuple(rv)


def has_partial_execution_value(argument_values: Sequence[Value]) -> bool:
    for argv in argument_values:
        if isinstance(argv, Value):
            if argv.has_optimistic_value():
                return True
        elif isinstance(argv, (int, slice)):
            pass
        else:
            raise TypeError('Unknown argument value type: {}.'.format(type(argv)))
    return False


def expand_argument_values(argument_values: Sequence[Value]) -> List[Value]:
    has_slot_var = False
    for arg in argument_values:
        if isinstance(arg, Value):
            for var in arg.batch_variables:
                if var == '??':
                    has_slot_var = True
                    break
    if has_slot_var:
        return list(argument_values)

    if len(argument_values) < 2:
        return list(argument_values)

    argument_values = list(argument_values)
    batch_variables = list()
    batch_sizes = list()
    for arg in argument_values:
        if isinstance(arg, Value):
            for var in arg.batch_variables:
                if var not in batch_variables:
                    batch_variables.append(var)
                    batch_sizes.append(arg.get_variable_size(var))
        else:
            assert isinstance(arg, (int, slice)), arg

    masks = list()
    for i, arg in enumerate(argument_values):
        if isinstance(arg, Value):
            argument_values[i] = arg.expand(batch_variables, batch_sizes)
            if argument_values[i].tensor_mask is not None:
                masks.append(argument_values[i].tensor_mask)

    if len(masks) > 0:
        final_mask = torch.stack(masks, dim=-1).amin(dim=-1)
        for arg in argument_values:
            if isinstance(arg, Value):
                arg.tensor_mask = final_mask
                arg._mask_certified_flag = True  # now we have corrected the mask.
    return argument_values


class FunctionApplication(ValueOutputExpression, ABC):
    def __init__(self, function_def: FunctionDef, *arguments: Expression):
        self.function_def = function_def
        self.arguments = arguments
        self._check_arguments()

    def _check_arguments(self):
        try:
            if len(self.function_def.arguments) != len(self.arguments):
                raise TypeError('Argument number mismatch: expect {}, got {}.'.format(len(self.function_def.arguments), len(self.arguments)))
            for i, (arg_def, arg) in enumerate(zip(self.function_def.arguments, self.arguments)):
                if isinstance(arg_def, Variable):
                    if isinstance(arg, VariableExpression):
                        if arg_def.dtype != arg.type and not isinstance(arg.type, type(RUNTIME_BINDING)):
                            raise FunctionApplicationError(i, arg_def.dtype, arg.type)
                    elif isinstance(arg, ObjectConstantExpression):
                        if arg_def.dtype != arg.type:
                            raise FunctionApplicationError(i, arg_def.dtype, arg.type)
                    elif isinstance(arg, FunctionApplication):
                        if arg_def.dtype != arg.return_type:
                            raise FunctionApplicationError(i, arg_def.dtype, arg.return_type)
                    else:
                        raise FunctionApplicationError(i, 'VariableExpression or FunctionApplication', type(arg))
                elif isinstance(arg_def, (ValueType, NamedValueType)):
                    if isinstance(arg, ValueOutputExpression):
                        pass
                    elif isinstance(arg, VariableExpression) and isinstance(arg.return_type, (ValueType, NamedValueType)):
                        pass
                    else:
                        raise FunctionApplicationError(i, 'ValueOutputExpression', type(arg))
                    if arg_def != arg.return_type:
                        raise FunctionApplicationError(i, arg_def, arg.return_type)
                else:
                    raise TypeError('Unknown argdef type: {}.'.format(type(arg_def)))
        except (TypeError, FunctionApplicationError) as e:
            error_header = 'Error during applying {}.\n'.format(str(self.function_def))
            try:
                arguments_str = ', '.join(str(arg) for arg in self.arguments)
                error_header += ' Arguments: {}\n'.format(arguments_str)
            except:
                pass
            raise TypeError(error_header + str(e)) from e

    @property
    def return_type(self):
        return self.function_def.return_type

    def __str__(self):
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.function_def.name}({arguments})'

    def _forward_external_function(self, ctx: ExpressionExecutionContext, external_function: Callable, argument_values: List[FunctionArgumentValueType], force_quantized: bool = False, force_non_optimistic: bool = False) -> Value:
        """Forward an external function.

        Args:
            ctx (ExpressionExecutionContext): The expression execution context.
            external_function (Callable): The external function to be called. Should be a bare function that supports direct __call__.
            argument_values (List[FunctionArgumentValueType]): A list of argument values.
            force_quantized (bool, optional): Internal flag. If true, the return value will always be quantized.
            force_non_optimistic (bool, optional): Internal flag. If true, we will always call the non-optimistic-execution branch.

        Returns:
            Value: The return value.
        """
        if not force_non_optimistic and ctx.is_optimistic_execution:  # optimistic execution branch.
            argument_values = expand_argument_values(argument_values)
            optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values if argv.quantized]
            if len(optimistic_masks) > 0:
                optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
                if optimistic_mask.sum().item() == 0:
                    pass  # just do the standard execution.
                else:
                    retain_mask = torch.logical_not(optimistic_mask)
                    rv = torch.zeros(
                        argument_values[0].tensor.shape,
                        dtype=torch.int64,
                        device=argument_values[0].tensor.device
                    )

                    if retain_mask.sum().item() > 0:
                        argument_values_r = [Value(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=argv.quantized) for argv in argument_values]
                        rv_r = self._forward_external_function(ctx, external_function, argument_values_r, force_quantized=True, force_non_optimistic=True)
                        rv[retain_mask] = rv_r.tensor

                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)
                        new_identifier = ctx.optimistic_context.new_var(self.return_type)
                        rv[ind] = new_identifier
                        ctx.optimistic_context.add_constraint(OptimisticConstraint.from_function_def(
                            self.function_def,
                            [argv.tensor[ind].item() if argv.quantized else argv.tensor[ind] for argv in argument_values],
                            new_identifier
                        ), note=str(self))

                    return Value(
                        self.function_def.return_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                        rv,
                        batch_dims=ctx.state.batch_dims, quantized=True
                    )

        if isinstance(self.function_def, PredicateDef) and self.function_def.is_generator_placeholder:  # always true branch
            argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
            argument_values = expand_argument_values(argument_values)
            rv = torch.ones(argument_values[0].tensor.shape[:argument_values[0].total_batch_dims], dtype=torch.int64, device=argument_values[0].tensor.device)
            return Value(
                self.function_def.return_type, argument_values[0].batch_variables,
                rv, batch_dims=ctx.state.batch_dims, quantized=True
            )

        # Standard execution branch.
        quantized = False
        all_quantized = all([v.quantized for v in argument_values if isinstance(v, Value)])
        if all_quantized and external_function.function_quantized is not None:
            if external_function.auto_broadcast:
                argument_values = expand_argument_values(argument_values)
            rv = external_function.function_quantized(*argument_values)
            quantized = True
        else:
            argument_values = list(argument_values)
            for i, argv in enumerate(argument_values):
                if isinstance(argv, Value) and argv.quantized and not is_intrinsically_quantized(argv.dtype):
                    argument_values[i] = ctx.value_quantizer.unquantize_value(argv)
            if external_function.auto_broadcast:
                argument_values = expand_argument_values(argument_values)
            rv = external_function(*argument_values)

        if torch.is_tensor(rv):
            rv = Value(
                self.function_def.return_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                QuantizedTensorValue(rv, None, argument_values[0].tensor_mask if len(argument_values) > 0 else None),
                batch_dims=ctx.state.batch_dims, quantized=quantized
            )
        elif isinstance(rv, QuantizedTensorValue):
            rv = Value(
                self.function_def.return_type, argument_values[0].batch_variables if len(argument_values) > 0 else [],
                rv,
                batch_dims=ctx.state.batch_dims, quantized=quantized
            )
        elif isinstance(self.function_def.return_type, PyObjValueType) and not isinstance(rv, Value):
            rv = ctx.session.pyobj_store.make_batched_value(self.function_def.return_type.typename, rv)
        else:
            assert isinstance(rv, Value), 'Expect external function return Tensor, Value, or QuantizedTensorValue objects, got {}.'.format(type(rv))

        if not rv.quantized and force_quantized:
            return ctx.value_quantizer.quantize_value(rv)
        return rv


class PredicateApplication(FunctionApplication):
    RUNTIME_BINDING_CHECK = False
    function_def: PredicateDef

    def _check_arguments(self):
        assert isinstance(self.function_def, PredicateDef)
        super()._check_arguments()

    def _check_arguments_runtime(self, ctx):
        if ctx.state.batch_dims == 0:  # Does not work for BatchState.
            for arg_def, arg in zip(self.function_def.arguments, self.arguments):
                if isinstance(arg, ObjectConstantExpression):
                    if arg.type == RUNTIME_BINDING:
                        got_type = ctx.state.get_typename(arg.name)
                        exp_type = arg_def.dtype.typename
                        if got_type != exp_type:
                            error_header = 'Error during applying {}.\n'.format(str(self.function_def))
                            raise TypeError(error_header + f'Runtime type check for argument {arg_def.name}: expect {exp_type}, got {got_type}.')

    @property
    def predicate_def(self) -> PredicateDef:
        return self.function_def

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        if self.RUNTIME_BINDING_CHECK:
            self._check_arguments_runtime(ctx)

        if self.predicate_def.is_cacheable and self.predicate_def.name in ctx.state.features:  # i.e., all variables are ObjectType
            argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
            batch_variables = [arg.name for arg, value in zip(self.arguments, argument_values) if value == QINDEX]
            value = ctx.state.features[self.predicate_def.name][argument_values]
            return value.rename_batch_variables(batch_variables)
        elif self.predicate_def.is_derived:
            argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
            return self._forward_expr_internal(ctx, argument_values)
        else:
            # dynamic predicate is exactly the same thing as a pre-defined external function.
            external_function = None
            if not self.function_def.is_generator_placeholder:
                external_function = ctx.get_external_function('predicate::' + self.predicate_def.name)
            argument_values = forward_args(ctx, *self.arguments, force_tuple=True, keep_name=True)
            return self._forward_external_function(ctx, external_function, argument_values)

    def _forward_expr_internal(self, ctx, argument_values):
        bounded_variables = dict()
        for var, value in zip(self.predicate_def.arguments, argument_values):
            if var.dtype.typename not in bounded_variables:
                bounded_variables[var.dtype.typename] = dict()
            bounded_variables[var.dtype.typename][var.name] = value

        nctx = ExpressionExecutionContext(ctx.session, ctx.state, bounded_variables=bounded_variables, optimistic_context=ctx.optimistic_context)
        with nctx.as_default():
            rv = self.predicate_def.expr.forward(nctx)

        guess_new_names = [arg.name for arg, value in zip(self.arguments, argument_values) if isinstance(arg, VariableExpression) and value == QINDEX]
        if len(guess_new_names) > 0:
            return rv.rename_batch_variables(guess_new_names)
        return rv


class ExternalFunctionApplication(FunctionApplication):
    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        external_function = None
        if isinstance(self.function_def, PredicateDef) and not self.function_def.is_generator_placeholder:
            external_function = ctx.get_external_function(self.function_def.name)
        if not isinstance(self.function_def, PredicateDef):
            # NB(Jiayuan Mao @ 09/29): unfound external function is not an Error. You don't need it for optimistic execution.
            external_function = ctx.get_external_function(self.function_def.name, default=None)
            if external_function is None and not ctx.is_optimistic_execution:
                raise ValueError(f'Cannot find external function {self.function_def.name}.')

        argument_values = forward_args(ctx, *self.arguments, force_tuple=True, keep_name=True)
        return self._forward_external_function(ctx, external_function, argument_values)


def is_predicate_application(expr: Expression) -> bool:
    return isinstance(expr, PredicateApplication)


def is_external_function_application(expr: Expression) -> bool:
    return isinstance(expr, ExternalFunctionApplication) or (
        isinstance(expr, PredicateApplication) and
        expr.predicate_def.expr is None and
        not expr.predicate_def.is_cacheable
    )


class ConditionalSelectOp(ValueOutputExpression):
    def __init__(self, predicate: ValueOutputExpression, condition: ValueOutputExpression):
        self.predicate = predicate
        self.condition = condition

    def _check_arguments(self):
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.return_type == BOOL

    @property
    def return_type(self):
        return self.predicate.return_type

    def _forward(self, ctx: ExpressionExecutionContext):
        if ctx.is_optimistic_execution:
            raise RuntimeError('ConditionalSelectOp is not supported in optimistic execution.')
        value, condition = forward_args(ctx, self.predicate, self.condition)
        value = value.clone()
        if value.tensor_mask is None:
            value.tensor_mask = torch.ones(value.tensor.shape[:value.total_batch_dims], device=value.tensor.device, dtype=torch.int64)
        value.tensor_mask = torch.min(value.tensor_mask, condition.tensor)
        return value

    def __str__(self):
        return f'cond-select({self.predicate} if {self.condition})'


class DeicticSelectOp(ValueOutputExpression):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        self.variable = variable
        self.expr = expr
        self._check_arguments()

    @property
    def return_type(self):
        return self.expr.return_type

    def _check_arguments(self):
        assert isinstance(self.variable.dtype, ObjectType)

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            return self.expr.forward(ctx)

    def __str__(self):
        return f'deictic-select({self.variable}: {self.expr})'


class BoolOpType(JacEnum):
    AND = 'and'
    OR = 'or'
    NOT = 'not'


class BoolOp(ValueOutputExpression):
    def __init__(self, boolean_op_type: BoolOpType, arguments: Sequence[ValueOutputExpression]):
        self.bool_op_type = boolean_op_type
        self.arguments = arguments
        self._check_arguments()

    def _check_arguments(self):
        if self.bool_op_type is BoolOpType.NOT:
            assert len(self.arguments) == 1, f'Number of arguments for NotOp should be 1, got: {len(self.arguments)}.'
        for i, arg in enumerate(self.arguments):
            assert isinstance(arg, ValueOutputExpression), f'BooOp only accepts ValueOutputExpressions, got argument #{i} of type {type(arg)}.'

    @property
    def return_type(self):
        return self.arguments[0].return_type

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        argument_values = forward_args(ctx, *self.arguments, force_tuple=True)
        argument_values = list(expand_argument_values(argument_values))

        if ctx.is_optimistic_execution:
            optimistic_masks = [is_optimistic_value(argv.tensor) for argv in argument_values if argv.quantized]
            if len(optimistic_masks) > 0:
                optimistic_mask = torch.stack(optimistic_masks, dim=-1).any(dim=-1)
                if optimistic_mask.sum().item() > 0:
                    for i, argv in enumerate(argument_values):
                        # TODO (Jiayuan Mao @ 03/31):: Whether we want to issue a warning or just do the quantization?
                        if not argv.quantized:
                            argument_values[i] = argv.make_quantized()
                        # assert argv.quantized, 'Found optimistic values in BoolOp, but at least one of the arguments is not quantized.'
                    retain_mask = torch.logical_not(optimistic_mask)
                    rv = torch.zeros(
                        argument_values[0].tensor.shape,
                        dtype=torch.int64,
                        device=argument_values[0].tensor.device
                    )
                    if retain_mask.sum().item() > 0:
                        argument_values_r = [Value(argv.dtype, ['?x'], argv.tensor[retain_mask], 0, quantized=True) for argv in argument_values]
                        rv_r = self._forward_inner(ctx, argument_values_r)
                        rv[retain_mask] = rv_r.tensor

                    for ind in optimistic_mask.nonzero().tolist():
                        ind = tuple(ind)

                        this_argv = [argv.tensor[ind].item() for argv in argument_values]
                        determined = None
                        if self.return_type == BOOL:
                            if self.bool_op_type is BoolOpType.NOT:
                                pass  # nothing we can do.
                            elif self.bool_op_type is BoolOpType.AND:
                                if 0 in this_argv:
                                    determined = False
                            elif self.bool_op_type is BoolOpType.OR:
                                if 1 in this_argv:
                                    determined = True
                            this_argv = list(filter(is_optimistic_value, this_argv))
                        else:  # generalized boolean operations.
                            pass

                        if determined is None:
                            new_identifier = ctx.optimistic_context.new_var(self.return_type)
                            rv[ind] = new_identifier
                            ctx.optimistic_context.add_constraint(OptimisticConstraint(
                                self.bool_op_type,
                                [cvt_opt_value(v, self.return_type) for v in this_argv],
                                cvt_opt_value(new_identifier, self.return_type),
                            ), note=str(self))
                        else:
                            rv[ind] = determined
                    return Value(self.return_type, argument_values[0].batch_variables, rv, argument_values[0].batch_dims, quantized=True)
                else:
                    return self._forward_inner(ctx, argument_values)
            else:  # if len(optimistic_masks) == 0
                return self._forward_inner(ctx, argument_values)
        else:
            return self._forward_inner(ctx, argument_values)

    def _forward_inner(self, ctx: ExpressionExecutionContext, argument_values) -> Value:
        for value in argument_values:
            assert value.tensor_indices is None, 'Does not support quantification over values with tensor_indices.'

        if len(argument_values) == 0:
            return Value(BOOL, [], torch.tensor(1, dtype=torch.int64), quantized=True)

        all_quantized = all([argv.quantized for argv in argument_values])
        # NB(Jiayuan Mao @ 03/12): when not all quantized, the tensors should be casted to float32.
        # Interestingly, PyTorch automatically handles that.

        dtype = argument_values[0].dtype
        batch_variables = argument_values[0].batch_variables
        if self.bool_op_type is BoolOpType.NOT:
            rv = argument_values[0].tensor
            rv = torch.logical_not(rv) if rv.dtype == torch.bool else 1 - rv
        elif self.bool_op_type is BoolOpType.AND:
            if len(argument_values) == 1:
                return argument_values[0]
            rv = torch.stack([arg.tensor for arg in argument_values], dim=-1).amin(dim=-1)
        elif self.bool_op_type is BoolOpType.OR:
            if len(argument_values) == 1:
                return argument_values[0]
            rv = torch.stack([arg.tensor for arg in argument_values], dim=-1).amax(dim=-1)
        else:
            raise ValueError('Unknown Boolean op type: {}.'.format(self.bool_op_type))
        return Value(
            dtype, batch_variables,
            QuantizedTensorValue(rv, None, argument_values[0].tensor_mask),
            batch_dims=argument_values[0].batch_dims, quantized=all_quantized
        )

    def __str__(self):
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'{self.bool_op_type.value}({arguments})'


class AndOp(BoolOp):
    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.AND, arguments)


class OrOp(BoolOp):
    def __init__(self, *arguments: ValueOutputExpression):
        super().__init__(BoolOpType.OR, arguments)


class NotOp(BoolOp):
    def __init__(self, arg: ValueOutputExpression):
        super().__init__(BoolOpType.NOT, [arg])


def is_and_expr(expr: Expression) -> bool:
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.AND


def is_or_expr(expr: Expression) -> bool:
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.OR


def is_not_expr(expr: Expression) -> bool:
    return isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT


def is_constant_bool_expr(expr: Expression) -> bool:
    if isinstance(expr, ConstantExpression) and expr.return_type == BOOL:
        return True
    return False


def is_simple_bool(expr: Expression) -> bool:
    if isinstance(expr, PredicateApplication) and expr.predicate_def.is_state_variable:
        return True
    if isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT:
        return is_simple_bool(expr.arguments[0])
    return False


def split_simple_bool(expr: Expression, initial_negated: bool = False) -> Tuple[Optional[PredicateApplication], bool]:
    """
    If the expression is a state variable predicate, it returns the feature definition and a boolean indicating whether the expression is negated.

    Args:
        expr (Expression): the expression to be checked.
        initial_negated (bool, optional): whether outer context of the feature expression is a negated function.

    Returns:
        (PredicateApplication, bool): A tuple of the feature application and a boolean indicating whether the expression is negated.
        The first element is None if the feature is not a simple Boolean feature application.
    """
    if isinstance(expr, PredicateApplication) and expr.predicate_def.is_state_variable:
        return expr, initial_negated
    if isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT:
        return split_simple_bool(expr.arguments[0], not initial_negated)
    return None, initial_negated


def get_simple_bool_def(expr: Expression):
    if isinstance(expr, PredicateApplication):
        return expr.predicate_def
    assert isinstance(expr, BoolOp) and expr.bool_op_type is BoolOpType.NOT
    return expr.arguments[0].predicate_def


class QuantifierType(JacEnum):
    FORALL = 'forall'
    EXISTS = 'exists'


class QuantificationOp(ValueOutputExpression):
    def __init__(self, quantifier_type: QuantifierType, variable: Variable, expr: ValueOutputExpression):
        self.quantifier_type = quantifier_type
        self.variable = variable
        self.expr = expr

        self._check_arguments()

    def _check_arguments(self):
        assert isinstance(self.expr, ValueOutputExpression), f'QuantificationOp only accepts ValueOutputExpressions, got type {type(self.expr)}.'
        assert isinstance(self.variable.dtype, ObjectType)

    @property
    def return_type(self):
        return self.expr.return_type

    def _forward(self, ctx: ExpressionExecutionContext) -> Value:
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            value = self.expr.forward(ctx)
        assert self.variable.name in value.batch_variables, f'Quantified argument is not in batch_variables: expect {self.variable.name}, got {value.batch_variables}.'
        rv_r = self._forward_inner(ctx, value)

        if ctx.is_optimistic_execution and value.quantized:
            dim = value.batch_variables.index(self.variable.name) + value.batch_dims
            value_transformed = value.tensor
            if dim != value.tensor.ndim - 1:
                value_transformed = value.tensor.transpose(dim, -1)  # put the target dimension last.
            optimistic_mask = is_optimistic_value(value_transformed).any(dim=-1)

            if optimistic_mask.sum().item() == 0:
                return rv_r

            rv = rv_r.tensor.clone()
            for ind in optimistic_mask.nonzero().tolist():
                ind = tuple(ind)

                this_argv = value_transformed[ind].tolist()
                determined = None
                if self.return_type == BOOL:
                    if self.quantifier_type is QuantifierType.FORALL:
                        if 0 in this_argv:
                            determined = False
                    else:
                        if 1 in this_argv:
                            determined = True
                    this_argv = list(filter(is_optimistic_value, this_argv))
                else:  # generalized quantization.
                    pass

                if determined is None:
                    new_identifier = ctx.optimistic_context.new_var(self.return_type)
                    rv[ind] = new_identifier
                    ctx.optimistic_context.add_constraint(OptimisticConstraint(
                        self.quantifier_type,
                        [cvt_opt_value(v, value.dtype) for v in this_argv],
                        cvt_opt_value(new_identifier, value.dtype)
                    ), note=f'{self}::{ind}')
                else:
                    rv[ind] = determined
            return Value(self.return_type, rv_r.batch_variables, rv, batch_dims=rv_r.batch_dims, quantized=True)
        else:
            return rv_r

    def _forward_inner(self, ctx: ExpressionExecutionContext, value):
        dim = value.batch_variables.index(self.variable.name) + value.batch_dims
        batch_variables = list(value.batch_variables)
        batch_variables.remove(self.variable.name)

        assert value.tensor_indices is None, 'Does not support quantification over values with tensor_indices.'
        if value.tensor_mask is None:
            masked_tensor = value.tensor
            tensor_mask = None
        else:
            if self.quantifier_type is QuantifierType.FORALL:
                masked_tensor = (value.tensor * value.tensor_mask + (1 - value.tensor_mask)).to(value.tensor.dtype)
            elif self.quantifier_type is QuantifierType.EXISTS:
                masked_tensor = (value.tensor * value.tensor_mask)
            else:
                raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))
            tensor_mask = value.tensor_mask.narrow(dim, 0, 1).squeeze(dim)

        if self.quantifier_type is QuantifierType.FORALL:
            return Value(
                value.dtype, batch_variables, QuantizedTensorValue(masked_tensor.amin(dim=dim), None, tensor_mask),
                batch_dims=value.batch_dims, quantized=value.quantized
            )
        elif self.quantifier_type is QuantifierType.EXISTS:
            return Value(
                value.dtype, batch_variables, QuantizedTensorValue(masked_tensor.amax(dim=dim), None, tensor_mask),
                batch_dims=value.batch_dims, quantized=value.quantized
            )
        else:
            raise ValueError('Unknown quantifier type: {}.'.format(self.quantifier_type))

    def __str__(self):
        return f'{self.quantifier_type.value}({self.variable}: {self.expr})'


class ForallOp(QuantificationOp):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantifierType.FORALL, variable, expr)


class ExistsOp(QuantificationOp):
    def __init__(self, variable: Variable, expr: ValueOutputExpression):
        super().__init__(QuantifierType.EXISTS, variable, expr)


def is_forall_expr(expr: Expression) -> bool:
    return isinstance(expr, QuantificationOp) and expr.quantifier_type is QuantifierType.FORALL


def is_exists_expr(expr: Expression) -> bool:
    return isinstance(expr, QuantificationOp) and expr.quantifier_type is QuantifierType.EXISTS


class _PredicateValueExpression(Expression, ABC):
    def __init__(self, predicate: Union[VariableExpression, PredicateApplication], value: ValueOutputExpression):
        self.predicate = predicate
        self.value = value
        self._check_arguments()

    def _check_arguments(self):
        try:
            if isinstance(self.predicate.return_type, (ValueType, NamedValueType)):
                if self.predicate.return_type.assignment_type() != self.value.return_type:
                    raise FunctionApplicationError(0, f'{self.predicate.return_type}(assignment type is {self.predicate.return_type.assignment_type()})', self.value.return_type)
            else:
                raise TypeError('Unknown argdef type: {}.'.format(type(self.predicate.return_type)))
        except TypeError as e:
            raise e
        except FunctionApplicationError as e:
            error_header = 'Error during _PredicateValueExpression checking: feature = {} value = {}.\n'.format(str(self.predicate), str(self.value))
            raise TypeError(
                error_header +
                f'Value type does not match: expect: {e.expect}, got {e.got}.'
            ) from e


class PredicateEqualOp(ValueOutputExpression, _PredicateValueExpression):
    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.predicate, (VariableExpression, PredicateApplication)), 'FeatureEqualOp only support dest type VariableExpression or PredicateApplication or PredicateApplication, got {}.'.format(type(self.predicate))

    @property
    def return_type(self):
        return BOOL

    def _forward(self, ctx: ExpressionExecutionContext):
        feature, value = forward_args(ctx, self.predicate, self.value)
        feature, value = expand_argument_values([feature, value])

        if ctx.is_optimistic_execution and (feature.quantized or value.quantized):
            return self._forward_optimistic(ctx, feature, value)

        if isinstance(feature.dtype, NamedValueType):
            rv = ctx.domain.get_external_function(f'type::{feature.dtype.typename}::equal')(feature, value)
        elif feature.quantized and value.quantized:
            rv = torch.eq(feature.tensor, value.tensor)
        elif feature.tensor_indices is not None and value.tensor_indices is not None:
            rv = torch.eq(feature.tensor_indices, value.tensor_indices)
        else:
            raise NotImplementedError('Unsupported FeatureEqual computation for dtype {}.'.format(feature, value))

        if not isinstance(rv, Value):
            rv = Value(BOOL, feature.batch_variables, QuantizedTensorValue(rv, None, feature.tensor_mask), batch_dims=feature.batch_dims, quantized=True)
        return rv

    def _forward_optimistic(self, ctx: ExpressionExecutionContext, feature: Value, value: Value) -> Value:
        feature, value = ctx.value_quantizer.quantize_value(feature), ctx.value_quantizer.quantize_value(value)
        rv_r = torch.eq(feature.tensor, value.tensor)
        optimistic_mask = torch.logical_or(is_optimistic_value(feature.tensor), is_optimistic_value(value.tensor))

        if optimistic_mask.sum().item() == 0:
            return Value(BOOL, feature.batch_variables, rv_r, batch_dims=feature.batch_dims, quantized=True)

        rv = rv_r.clone().to(torch.int64)
        for ind in optimistic_mask.nonzero().tolist():
            ind = tuple(ind)
            this_argv = feature.tensor[ind].item(), value.tensor[ind].item()
            new_identifier = ctx.optimistic_context.new_var(BOOL)
            rv[ind] = new_identifier
            ctx.optimistic_context.add_constraint(EqualOptimisticConstraint(
                *[cvt_opt_value(v, feature.dtype) for v in this_argv],
                cvt_opt_value(new_identifier, BOOL)
            ), note=f'{self}::{ind}')

        return Value(BOOL, feature.batch_variables, rv, batch_dims=feature.batch_dims, quantized=True)

    def __str__(self):
        return f'equal({self.predicate}, {self.value})'


class VariableAssignmentExpression(Expression, ABC):
    pass


def is_variable_assignment_expression(expr: Expression) -> bool:
    return isinstance(expr, VariableAssignmentExpression)


class AssignOp(_PredicateValueExpression, VariableAssignmentExpression):
    def __init__(self, feature: PredicateApplication, value: Expression):
        _PredicateValueExpression.__init__(self, feature, value)

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.predicate, (PredicateApplication, PredicateApplication)), 'AssignOp only support dest type PredicateApplication or PredicateApplication, got {}.'.format(type(self.predicate))

    def _forward(self, ctx: ExpressionExecutionContext):
        argument_values = forward_args(ctx, *self.predicate.arguments, force_tuple=True)
        value = forward_args(ctx, self.value)

        if ctx.state.features[self.predicate.function_def.name].quantized:
            if not value.quantized:
                value = ctx.value_quantizer.quantize_value(value)
        else:
            if value.quantized:
                value = ctx.value_quantizer.unquantize_value(value)
        ctx.state.features[self.predicate.predicate_def.name][argument_values] = value

    def __str__(self):
        return f'assign({self.predicate}: {self.value})'


class ConditionalAssignOp(_PredicateValueExpression, VariableAssignmentExpression):
    OPTIONS = {'quantize': False}

    @staticmethod
    def set_options(**kwargs):
        ConditionalAssignOp.OPTIONS.update(kwargs)

    def __init__(self, feature: PredicateApplication, value: ValueOutputExpression, condition: ValueOutputExpression):
        self.condition = condition
        _PredicateValueExpression.__init__(self, feature, value)

    def _check_arguments(self):
        super()._check_arguments()
        assert isinstance(self.condition, ValueOutputExpression) and self.condition.return_type == BOOL

    def _forward(self, ctx: ExpressionExecutionContext):
        argument_values = forward_args(ctx, *self.predicate.arguments, force_tuple=True)
        value = forward_args(ctx, self.value)
        condition = forward_args(ctx, self.condition)

        if ctx.state.features[self.predicate.function_def.name].quantized:
            if not value.quantized:
                value = ctx.value_quantizer.quantize_value(value)
        else:
            if value.quantized:
                value = ctx.value_quantizer.unquantize_value(value)

        condition_tensor = jactorch.quantize(condition.tensor) if ConditionalAssignOp.OPTIONS['quantize'] else condition.tensor
        if value.tensor.dim() > condition_tensor.dim():
            condition_tensor = condition_tensor.unsqueeze(-1)

        origin_tensor = ctx.state.features[self.predicate.predicate_def.name].tensor[argument_values]
        assert value.tensor.dim() == condition_tensor.dim()

        if ctx.is_optimistic_execution and value.quantized:
            optimistic_mask = is_optimistic_value(condition_tensor)

            if optimistic_mask.sum().item() == 0:
                pass
            else:
                dtype = self.predicate.predicate_def.return_type
                for ind in optimistic_mask.nonzero().tolist():
                    ind = tuple(ind)

                    new_identifier = ctx.optimistic_context.new_var(dtype, wrap=True)
                    neg_condition_identifier = ctx.optimistic_context.new_var(BOOL, wrap=True)
                    eq_1_identifier = ctx.optimistic_context.new_var(BOOL, wrap=True)
                    eq_2_identifier = ctx.optimistic_context.new_var(BOOL, wrap=True)

                    ctx.optimistic_context.add_constraint(EqualOptimisticConstraint(
                        new_identifier,
                        cvt_opt_value(value.tensor[ind].item(), dtype),
                        eq_1_identifier,
                    ), note=f'{self}::{ind}::eq-1')
                    ctx.optimistic_context.add_constraint(EqualOptimisticConstraint(
                        new_identifier,
                        cvt_opt_value(origin_tensor[ind].item(), dtype),
                        eq_2_identifier
                    ), note=f'{self}::{ind}::eq-2')

                    ctx.optimistic_context.add_constraint(OptimisticConstraint(
                        BoolOpType.NOT,
                        [cvt_opt_value(condition_tensor[ind].item(), BOOL)],
                        neg_condition_identifier
                    ), note=f'{self}::{ind}::neg-cond')

                    ctx.optimistic_context.add_constraint(OptimisticConstraint(
                        BoolOpType.OR,
                        [neg_condition_identifier, eq_1_identifier],
                        cvt_opt_value(True, BOOL)
                    ), note=f'{self}::{ind}::implies-1')

                    ctx.optimistic_context.add_constraint(OptimisticConstraint(
                        BoolOpType.OR,
                        [cvt_opt_value(condition_tensor[ind].item(), BOOL), eq_2_identifier],
                        cvt_opt_value(True, BOOL)
                    ), note=f'{self}::{ind}::implies-2')

                    condition_tensor[ind] = 1
                    value.tensor[ind] = new_identifier.identifier

        if value.quantized:
            ctx.state.features[self.predicate.predicate_def.name][argument_values] = (
                value.tensor * condition_tensor +
                origin_tensor * (1 - condition_tensor)
            )
        else:
            ctx.state.features[self.predicate.predicate_def.name][argument_values] = (
                value.tensor * condition_tensor.float() +
                origin_tensor * (1 - condition_tensor.float())
            )

    def __str__(self):
        return f'cond-assign({self.predicate}: {self.value} if {self.condition})'


class DeicticAssignOp(VariableAssignmentExpression):
    def __init__(self, variable: Variable, expr: Union[VariableAssignmentExpression]):
        self.variable = variable
        self.expr = expr
        self._check_arguments()

    def _check_arguments(self):
        assert isinstance(self.variable.dtype, ObjectType)

    def _forward(self, ctx: ExpressionExecutionContext):
        with ctx.new_bounded_variables({self.variable: QINDEX}):
            self.expr.forward(ctx)

    def __str__(self):
        return f'deictic-assign({self.variable}: {self.expr})'


def flatten_expression(
    expr: Expression,
    mappings: Optional[Dict[Union[PredicateApplication, VariableExpression], Union[Variable, ValueOutputExpression]]] = None,
    ctx: Optional[ExpressionDefinitionContext] = None,
    flatten_cacheable_bool: bool = True,
) -> Union[AssignOp, ConditionalAssignOp, DeicticAssignOp, VariableExpression, ValueOutputExpression]:
    if ctx is None:
        ctx = ExpressionDefinitionContext()
    if mappings is None:
        mappings = {}

    with ctx.as_default():
        return _flatten_expression_inner(expr, mappings, flatten_cacheable_bool=flatten_cacheable_bool)


# @jacinle.log_function
def _flatten_expression_inner(
    expr: Expression,
    mappings: Dict[Union[PredicateApplication, VariableExpression], Union[Variable, ValueOutputExpression]],
    flatten_cacheable_bool: bool,
) -> Union[VariableExpression, ValueOutputExpression, VariableAssignmentExpression]:
    ctx = get_definition_context()

    if isinstance(expr, BoolOp):
        return BoolOp(expr.bool_op_type, [_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
    elif isinstance(expr, QuantificationOp):
        with ctx.new_arguments(expr.variable):
            dummy_variable = ctx.generate_random_named_variable(expr.variable.dtype)
            mappings_inner = mappings.copy()
            mappings_inner[VariableExpression(expr.variable)] = dummy_variable
            return QuantificationOp(expr.quantifier_type, dummy_variable, _flatten_expression_inner(expr.expr, mappings_inner, flatten_cacheable_bool))
    elif isinstance(expr, PredicateEqualOp):
        return PredicateEqualOp(_flatten_expression_inner(expr.predicate, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool))
    elif isinstance(expr, PredicateApplication):
        for k, v in mappings.items():
            if not isinstance(k, PredicateApplication):
                continue
            if expr.predicate_def.name == k.predicate_def.name and all(
                isinstance(a1, VariableExpression) and isinstance(a2, VariableExpression) and a1.name == a2.name for a1, a2 in zip(expr.arguments, k.arguments)
            ):
                return VariableExpression(v)
        if expr.predicate_def.expr is None or expr.predicate_def.is_state_variable or (not flatten_cacheable_bool and expr.predicate_def.is_cacheable and expr.return_type == BOOL):
            return type(expr)(expr.function_def, *[_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
        else:
            for arg in expr.function_def.arguments:
                assert isinstance(arg, Variable)
            mappings_inner = mappings.copy()
            argvs = [_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments]
            nctx = ExpressionDefinitionContext(*expr.function_def.arguments)
            with nctx.as_default():
                for arg, argv in zip(expr.predicate_def.arguments, argvs):
                    if isinstance(arg, Variable):
                        mappings_inner[VariableExpression(arg)] = argv
                return _flatten_expression_inner(expr.predicate_def.expr, mappings_inner, flatten_cacheable_bool)
    elif isinstance(expr, FunctionApplication):
        return type(expr)(expr.function_def, *[_flatten_expression_inner(e, mappings, flatten_cacheable_bool) for e in expr.arguments])
    elif isinstance(expr, ConditionalSelectOp):
        return type(expr)(_flatten_expression_inner(expr.predicate, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.condition, mappings, flatten_cacheable_bool))
    elif isinstance(expr, VariableExpression):
        rv = expr
        for k, v in mappings.items():
            if isinstance(k, VariableExpression):
                if k.name == expr.name:
                    if isinstance(v, Variable):
                        rv = VariableExpression(v)
                    else:
                        rv = v
        return rv
    elif isinstance(expr, (ConstantExpression, ObjectConstantExpression)):
        return expr
    elif isinstance(expr, AssignOp):
        return AssignOp(_flatten_expression_inner(expr.predicate, mappings, flatten_cacheable_bool), _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool))
    elif isinstance(expr, ConditionalAssignOp):
        return ConditionalAssignOp(
            _flatten_expression_inner(expr.predicate, mappings, flatten_cacheable_bool),
            _flatten_expression_inner(expr.value, mappings, flatten_cacheable_bool),
            _flatten_expression_inner(expr.condition, mappings, flatten_cacheable_bool)
        )
    elif isinstance(expr, (DeicticSelectOp, DeicticAssignOp)):
        with ctx.new_arguments(expr.variable):
            dummy_variable = ctx.generate_random_named_variable(expr.variable.dtype)
            mappings_inner = mappings.copy()
            mappings_inner[VariableExpression(expr.variable)] = dummy_variable
            return type(expr)(dummy_variable, _flatten_expression_inner(expr.expr, mappings_inner, flatten_cacheable_bool))
    else:
        raise TypeError('Unknown expression type: {}.'.format(type(expr)))


def iter_exprs(expr: Expression) -> Iterable[Expression]:
    """Iterate over all sub-expressions of the input."""
    yield expr
    if isinstance(expr, BoolOp):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, QuantificationOp):
        yield from iter_exprs(expr.expr)
    elif isinstance(expr, PredicateEqualOp):
        yield from iter_exprs(expr.predicate)
        yield from iter_exprs(expr.value)
    elif isinstance(expr, FunctionApplication):
        for arg in expr.arguments:
            yield from iter_exprs(arg)
    elif isinstance(expr, AssignOp):
        yield from iter_exprs(expr.value)
    elif isinstance(expr, ConditionalSelectOp):
        yield from iter_exprs(expr.predicate)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, ConditionalAssignOp):
        yield from iter_exprs(expr.value)
        yield from iter_exprs(expr.condition)
    elif isinstance(expr, (DeicticSelectOp, DeicticAssignOp)):
        yield from iter_exprs(expr.expr)
    elif isinstance(expr, (PredicateApplication, VariableExpression, ConstantExpression, ObjectConstantExpression)):
        pass
    else:
        raise TypeError('Unknown expression type: {}.'.format(type(expr)))


def get_used_state_variables(expr: ValueOutputExpression) -> Set[PredicateDef]:
    assert isinstance(expr, ValueOutputExpression), (
        'Only value output expression has well-defined used-state-variables.\n'
        'For value assignment expressions, please separately process the targets, conditions, and values.'
    )

    used_svs = set()

    def dfs(this):
        nonlocal used_svs
        for e in iter_exprs(this):
            if isinstance(e, PredicateApplication):
                if e.predicate_def.is_state_variable:
                    used_svs.add(e.predicate_def)
                elif e.predicate_def.expr is not None:
                    dfs(e.predicate_def.expr)

    dfs(expr)
    return used_svs

