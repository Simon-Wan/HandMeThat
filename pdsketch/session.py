#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : session.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/04/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import collections
import jacinle
import torch
from typing import Any, Optional, Union, Tuple, Iterable, Sequence, List, Mapping, Dict, Callable

from .value import ObjectType, NamedValueType, Value, Variable, BOOL, is_intrinsically_quantized, scalar
from .expr import Expression, VariableExpression, ExpressionExecutionContext, PredicateApplication, BoundedVariablesDictCompatible, compose_bvdict, compose_bvdict_args, FunctionArgumentValueType
from .optimistic import OptimisticValueContext, OptimisticConstraint, DeterminedValue
from .state import StateLike, State
from .domain import PredicateDef, Domain


__all__ = ['ValueQuantizer', 'PyObjectStore', 'Session', 'get_session', 'set_default_session', 'TensorDictDefHelper']


class ValueQuantizer(object):
    def __init__(self, domain: Domain):
        self.domain = domain
        self.values: Dict[str, Union[List[Any], Dict[Any, int]]] = dict()

    def quantize(self, typename: str, value: Union[torch.Tensor, Value]) -> int:
        if not isinstance(value, Value):
            value = Value(self.domain.types[typename], [], value)
        use_hash = self.domain.has_external_function(f'type::{typename}::hash')
        if typename not in self.values:
            self.values[typename] = dict() if use_hash else list()

        if use_hash:
            hash_value = self.domain.get_external_function(f'type::{typename}::hash')(value)
            if hash_value not in self.values[typename]:
                self.values[typename][hash_value] = len(self.values[typename])
            return self.values[typename][hash_value]
        else:
            for i, v in enumerate(self.values[typename]):
                if bool(self.domain.get_external_function(f'type::{typename}::equal')(v, value)):
                    return i
            self.values[typename].append(value)
            return len(self.values[typename]) - 1

    def quantize_tensor(self, dtype: NamedValueType, tensor: torch.Tensor) -> torch.Tensor:
        tensor_flatten = tensor.reshape((-1,) + dtype.size_tuple())
        quantized_values = list()
        for value in tensor_flatten:
            quantized_values.append(self.quantize(dtype.typename, value))
        quantized_tensor = torch.tensor(quantized_values, dtype=torch.int64, device=tensor_flatten.device)
        quantized_tensor = quantized_tensor.reshape(tensor.shape[:-dtype.ndim()])
        return quantized_tensor

    def quantize_dict_list(self, continuous_values: Mapping[str, Sequence[int]]) -> Mapping[str, Sequence[Value]]:
        output_dict = dict()
        for typename, values in continuous_values.items():
            output_dict[typename] = set()
            for v in values:
                output_dict[typename].add(self.quantize(typename, v))
            output_dict[typename] = [Value(self.domain.types[typename], [], x, quantized=True) for x in output_dict[typename]]
        return output_dict

    def quantize_value(self, value: Value) -> Value:
        if value.quantized:
            return value
        if is_intrinsically_quantized(value.dtype):
            return value.make_quantized()
        if value.tensor_indices is not None:
            return value.make_quantized()
        assert isinstance(value.dtype, NamedValueType)
        return Value(value.dtype, value.batch_variables, self.quantize_tensor(value.dtype, value.tensor), value.batch_dims, quantized=True)

    def quantize_state(self, state: StateLike, includes=None, excludes=None) -> StateLike:
        state = state.clone()
        for feature_name in state.features.all_feature_names:
            if (includes is not None and feature_name not in includes) or (excludes is not None and feature_name in excludes):
                rv = state.features[feature_name]
            else:
                predicate_def = self.domain.predicates[feature_name]
                if predicate_def.is_observation_variable and not predicate_def.is_state_variable:
                    rv = state.features[feature_name]
                else:
                    rv = self.quantize_value(state.features[feature_name])
            state.features[feature_name] = rv
        return state

    def unquantize(self, typename: str, value: int) -> Value:
        return self.values[typename][value]

    def unquantize_tensor(self, dtype: NamedValueType, tensor: torch.Tensor) -> torch.Tensor:
        assert dtype.typename in self.values
        lookup_table = self.values[dtype.typename]
        output = [lookup_table[x].tensor for x in tensor.flatten().tolist()]
        output = torch.stack(output, dim=0).reshape(tensor.shape + dtype.size_tuple())
        return output

    def unquantize_value(self, value: Value) -> Value:
        dtype = value.dtype
        assert isinstance(dtype, NamedValueType)
        if is_intrinsically_quantized(dtype):
            return Value(dtype, value.batch_variables, value.tensor, value.batch_dims, quantized=False)
        else:
            return Value(dtype, value.batch_variables, self.unquantize_tensor(dtype, value.tensor), value.batch_dims, quantized=False)

    def unquantize_optimistic_context(self, ctx: OptimisticValueContext):
        def _cvt(arg):
            if isinstance(arg, DeterminedValue):
                if not arg.quantized:
                    return arg
                elif is_intrinsically_quantized(arg.dtype):
                    if arg.dtype == BOOL:
                        return DeterminedValue(BOOL, bool(arg.value), True)
                    return DeterminedValue(arg.dtype, int(arg.value), True)
                else:
                    assert isinstance(arg.dtype, NamedValueType) and isinstance(arg.value, int)
                    return DeterminedValue(arg.dtype, self.unquantize(arg.dtype.typename, arg.value), False)
            else:
                return arg

        ctx = ctx.clone()
        for i, c in enumerate(ctx.constraints):
            new_args = tuple(map(_cvt, c.args))
            new_rv = _cvt(c.rv)
            ctx.constraints[i] = OptimisticConstraint(c.func_def, new_args, new_rv, note=c.note)
        return ctx


class PyObjectStore(object):
    def __init__(self, domain: Domain):
        self.domain = domain
        self.objects = collections.defaultdict(list)

    def add(self, typename: str, python_object) -> int:
        self.objects[typename].append(python_object)
        return len(self.objects[typename]) - 1

    def retrieve(self, typename: str, index: int) -> Any:
        return self.objects[typename][index]

    def retrieve_value(self, value: Value) -> Union[Any, List[Any]]:
        return _tensor2pyobj_list(self, value.dtype.typename, value.tensor)

    def make_value(self, typename: str, python_object) -> Value:
        index = self.add(typename, python_object)
        return scalar(index, self.domain.types[typename])

    def make_batched_value(self, typename: str, pyobj_list: List[Any], batch_variables: Optional[Sequence[str]] = None) -> Value:
        sizes = _get_pyobj_list_size(pyobj_list)
        tensor = torch.zeros(sizes, dtype=torch.int64)

        for indices, pyobj in _iterate_pyobj_list(pyobj_list):
            tensor[indices] = self.add(typename, pyobj)

        return Value(
            self.domain.types[typename],
            batch_variables if batch_variables is not None else len(sizes),
            tensor, batch_dims=0, quantized=True
        )


def _tensor2pyobj_list(pyobj_store: PyObjectStore, typename: str, indices_tensor: torch.Tensor) -> Union[Any, List[Any]]:
    if indices_tensor.dim() == 0:
        return pyobj_store.retrieve(typename, indices_tensor.item())
    else:
        return [_tensor2pyobj_list(pyobj_store, typename, indices_tensor[i]) for i in range(indices_tensor.size(0))]


def _get_pyobj_list_size(pyobj_list: Union[Any, List[Any]]) -> int:
    if isinstance(pyobj_list, list):
        assert len(pyobj_list) > 0
        return (len(pyobj_list), ) + _get_pyobj_list_size(pyobj_list[0])
    else:
        return tuple()


def _iterate_pyobj_list(pyobj_list: Union[Any, List[Any]], indices: Tuple[int, ...] = tuple()) -> Iterable[Tuple[Tuple[int, ...], Any]]:
    """Given a nested list of pyobjs, yield a tuple of (indices, pyobj) for each pyobj."""
    if isinstance(pyobj_list, list):
        for i, pyobj in enumerate(pyobj_list):
            yield from _iterate_pyobj_list(pyobj, indices + (i,))
    else:
        yield indices, pyobj_list


class Session(object):
    def __init__(self, domain: Domain):
        self.domain = domain
        self.value_quantizer = ValueQuantizer(self.domain)
        self.pyobj_store = PyObjectStore(self.domain)
        self.environment: Optional[Any] = None

    @jacinle.wrap_custom_as_default
    def as_default(self):
        yield

    def set_as_default(self) -> 'Session':
        set_default_session(self)
        return self

    def set_environment(self, environment: Any):
        self.environment = environment

    def compile(self, expr: Union[str, Expression], variables: Optional[Sequence[Variable]] = None):
        expr = self.domain.parse(expr, variables=variables)

        def func(state, bounded_variables: Optional[BoundedVariablesDictCompatible] = None, optimistic_context: Optional[OptimisticValueContext] = None):
            return self.eval(state, expr, bounded_variables, optimistic_context=optimistic_context)

        return func

    def eval(
        self, state: StateLike, expr: Union[str, Expression], bounded_variables: Optional[BoundedVariablesDictCompatible] = None,
        optimistic_context: Optional[OptimisticValueContext] = None, return_ctx: bool = False
    ) -> Union[Value, Tuple[ExpressionExecutionContext, Value]]:

        bounded_variables = compose_bvdict(bounded_variables, state=state)

        for typename, args in bounded_variables.items():
            typedef = self.domain.types[typename]
            if isinstance(typedef, NamedValueType):
                for name, value in args.items():
                    if not isinstance(value, Value):
                        value = Value(typedef, [], value)
                        args[name] = value
            else:
                assert isinstance(typedef, ObjectType)
                for name, value in args.items():
                    if isinstance(value, str):
                        if state.batch_dims > 0:
                            value = torch.tensor(state.get_typed_index(value), dtype=torch.int64)
                        else:
                            value = state.get_typed_index(value)
                        args[name] = value

        if isinstance(expr, str):
            variables = [Variable(name, self.domain.types[typename]) for typename, args in bounded_variables.items() for name in args]
            expr = self.domain.parse(expr, variables=variables)

        ctx = ExpressionExecutionContext(self, state, bounded_variables=bounded_variables, optimistic_context=optimistic_context)
        with ctx.as_default():
            if return_ctx:
                return ctx, expr.forward(ctx)
            return expr.forward(ctx)

    def eval_predicate(self, state: StateLike, predicate_def: Union[str, PredicateDef], arguments: Sequence[FunctionArgumentValueType]) -> Value:
        if isinstance(predicate_def, str):
            predicate_def = self.domain.predicates[predicate_def]
        bounded_varaibles = compose_bvdict_args(predicate_def.arguments, arguments, state=state)
        expression = PredicateApplication(predicate_def, *[VariableExpression(arg) for arg in predicate_def.arguments])
        return self.eval(state, expression, bounded_variables=bounded_varaibles)

    def forward_predicate(self, state: StateLike, predicate_def: PredicateDef):
        if predicate_def.expr is None:
            assert predicate_def.name in state.features
            return

        result = self.eval(state, predicate_def.expr, predicate_def.arguments)
        state.features.set_feature(predicate_def.name, result)

    def forward_state_variables(self, state: StateLike):
        for predicate_def in self.domain.predicates.values():
            if predicate_def.is_state_variable and not predicate_def.is_observation_variable:
                if predicate_def.is_cacheable:
                    self.forward_predicate(state, predicate_def)

    def forward_derived_predicates(self, state: StateLike):
        for predicate_def in self.domain.predicates.values():
            if predicate_def.is_derived and not predicate_def.is_state_variable:
                if predicate_def.is_cacheable:
                    self.forward_predicate(state, predicate_def)

    def forward_axioms(self, state: StateLike):
        for op in self.domain.axioms.values():
            _, state = op.apply(self, state)
        return state

    def forward_predicates_and_axioms(
        self,
        state: StateLike,
        forward_state_variables: bool = True,
        forward_axioms: bool = True,
        forward_derived: bool = True
    ):
        if forward_state_variables:
            self.forward_state_variables(state)
        if forward_axioms:
            state = self.forward_axioms(state)
        if forward_derived:
            self.forward_derived_predicates(state)
        return state


get_session_inner: Callable[[], Session] = jacinle.gen_get_default(Session)


def get_session(session: Optional[Session] = None) -> Session:
    if session is not None:
        return session
    return get_session_inner()


set_default_session: Callable[[Session], None] = jacinle.gen_set_default(Session)


class _TensorDefPredicate(object):
    def __init__(self, predicate_def, arguments):
        self.predicate_def = predicate_def
        self.arguments = arguments


class _TensorDefPredicateApplier(object):
    def __init__(self, predicate_def):
        self.predicate_def = predicate_def

    def __call__(self, *args):
        return _TensorDefPredicate(self.predicate_def, args)


class TensorDictDefHelper(object):
    def __init__(self, domain: Domain, state: State, session: Optional[Session] = None):
        self.domain = domain
        self.state = state
        self.session = session

        if session is None:
            self.session = get_session()

    def get_pred(self, name):
        if name in self.domain.predicates:
            return _TensorDefPredicateApplier(self.domain.predicates[name])
        elif name.replace('_', '-') in self.domain.predicates:
            return _TensorDefPredicateApplier(self.domain.predicates[name.replace('_', '-')])
        else:
            raise NotImplementedError('Unknown predicate: {}.'.format(name))

    def __getattr__(self, name):
        return self.get_pred(name)

    def define_predicates(self, predicates: Sequence[_TensorDefPredicate]):
        for predicate_def in self.domain.predicates.values():
            if predicate_def.name in self.state.features.all_feature_names:
                continue
            if predicate_def.is_state_variable:
                sizes = list()
                for arg_def in predicate_def.arguments:
                    sizes.append(len(self.state.object_type2names[arg_def.typename]) if arg_def.typename in self.state.object_type2names else 0)
                self.state.features[predicate_def.name] = Value(
                    BOOL, [var.name for var in predicate_def.arguments],
                    torch.zeros(sizes, dtype=torch.int64),
                    batch_dims=0, quantized=True
                )

        for pred in predicates:
            assert isinstance(pred, _TensorDefPredicate)
            assert pred.predicate_def.return_type == BOOL
            name = pred.predicate_def.name
            arguments = [self.state.get_typed_index(arg) for arg in pred.arguments]
            self.state.features[name].tensor[tuple(arguments)] = 1

    def define_feature(self, feature_name, tensor_or_mapping, quantized=False):
        predicate_def = self.domain.predicates[feature_name]
        sizes = list()
        for arg_def in predicate_def.arguments:
            sizes.append(len(self.state.object_type2names[arg_def.typename]) if arg_def.typename in self.state.object_type2names else 0)
        sizes = tuple(sizes)
        if torch.is_tensor(tensor_or_mapping):
            self.state.features[feature_name] = Value(
                predicate_def.return_type, [var.name for var in predicate_def.arguments],
                tensor_or_mapping, batch_dims=0, quantized=quantized
            )
        else:
            if not quantized:
                tensor = torch.zeros(sizes + predicate_def.return_type.size_tuple())
            else:
                tensor = torch.zeros(sizes, dtype=torch.int64)

            for key, value in tensor_or_mapping.items():
                if isinstance(key, tuple):
                    args = [self.state.get_typed_index(arg) for arg in key]
                else:
                    assert isinstance(key, str)
                    args = [self.state.get_typed_index(key)]
                tensor[tuple(args)] = value
            self.state.features[feature_name] = Value(
                predicate_def.return_type, [var.name for var in predicate_def.arguments],
                tensor, batch_dims=0, quantized=quantized
            )

    def define_pyobj_feature(self, feature_name: str, pyobj_list: List[Any]):
        if self.session is None:
            raise ValueError('Session has to be specified to define pyobj features.')

        predicate_def = self.domain.predicates[feature_name]
        value = self.session.pyobj_store.make_batched_value(
            predicate_def.return_type.typename,
            pyobj_list,
            [var.name for var in predicate_def.arguments],
        )

        self.state.features[feature_name] = value

