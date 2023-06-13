#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import itertools
import torch
import jacinle

from typing import Any, Sequence, Tuple, List, Dict, ForwardRef
from .value import NamedValueType, Value, wrap_value
from .state import State
from .expr import FunctionDef, FunctionArgumentType, ValueOutputExpression

__all__ = ['GeneratorDef', 'extract_generator_data', 'ContinuousValueDict', 'generate_continuous_values', 'expand_continuous_values']

Domain = ForwardRef('Domain')
Session = ForwardRef('Session')


class GeneratorDef(object):
    def __init__(
        self,
        name: str,
        parameters: Sequence[FunctionArgumentType],
        certifies: ValueOutputExpression,
        context: Sequence[ValueOutputExpression],
        generates: Sequence[ValueOutputExpression],
        function_def: FunctionDef,
        output_vars: Sequence[FunctionArgumentType],
        flatten_certifies: ValueOutputExpression,
        priority: int = 0,
        unsolvable: bool = False
    ):
        self.name = name
        self.parameters = tuple(parameters)
        self.certifies = certifies
        self.context = tuple(context)
        self.generates = tuple(generates)
        self.function_def = function_def
        self.output_vars = output_vars
        self.flatten_certifies = flatten_certifies
        self.priority = priority
        self.unsolvable = unsolvable

    @property
    def input_vars(self):
        return self.function_def.arguments

    def __str__(self):
        arg_string = ', '.join([str(c) for c in self.context])
        gen_string = ', '.join([str(c) for c in self.generates])
        return (
            f'{self.name}({arg_string}) -> {gen_string}' + ' {\n'
            '  ' + str(self.function_def) + '\n'
            '  parameters: ' + str(self.parameters) + '\n'
            '  certifies:  ' + str(self.flatten_certifies) + '\n'
            '  context:    ' + str(self.context) + '\n'
            '  generates:  ' + str(self.generates) + '\n'
            '}'
        )

    __repr__ = jacinle.repr_from_str


def extract_generator_data(session: Session, state: State, generator_def: GeneratorDef) -> Tuple[List[Value], List[Value]]:
    ctx, result = session.eval(state, generator_def.certifies, generator_def.arguments, generator_def.certifies, return_ctx=True)
    result.tensor = torch.ge(result.tensor, 0.5)
    if result.tensor_mask is not None:
        result.tensor = torch.logical_and(result.tensor, torch.ge(result.tensor_mask, 0.5))

    def _index(value, mask):
        value = value.expand_as(mask)
        return value.tensor[mask.tensor]

    with ctx.as_default():
        contexts = [_index(c.forward(ctx), result) for c in generator_def.context]
        generates = [_index(c.forward(ctx), result) for c in generator_def.generates]
    return contexts, generates


ContinuousValueDict = Dict[str, List[Value]]


def generate_continuous_values(domain: Domain, state: State, nr_iterations: int = 1, nr_samples: int = 5) -> ContinuousValueDict:
    """The function generate_continuous_values and expand_continuous_values jointly implements the incremenetal search
    algorithm for Task and Motion Planning.

    Basically, the algorithm starts from generating a large collection of possible continuous parameters by "expanding"
    from the continuous parameters in the input state. Next, it reduces the TAMP problem into a basic discrete search
    problem. The downside of this approach is that it requires grounding a large collection of possible values,
    but it is in theory probabilistically complete.
    """
    continuous_values = dict()
    for type_def in domain.types.values():
        if isinstance(type_def, NamedValueType):
            continuous_values[type_def.typename] = list()

    for key, value in domain.predicates.items():
        if key in state.features.all_feature_names and isinstance(value.return_type, NamedValueType):
            type_def = value.return_type
            feat = state.features[key].tensor
            feat = feat.reshape((-1, ) + feat.shape[-type_def.ndim():])
            continuous_values[type_def.typename].extend([wrap_value(x, type_def) for x in feat])

    for i in range(nr_iterations):
        expand_continuous_values(domain, continuous_values, nr_samples=nr_samples)
    return continuous_values


def expand_continuous_values(domain: Domain, current: Dict[str, List[Any]], nr_samples: int = 5):
    for gen_name, gen_def in domain.generators.items():
        arguments = list()
        for arg in gen_def.context:
            assert isinstance(arg.return_type, NamedValueType)
            arguments.append(current[arg.return_type.typename])
        for comb in itertools.product(*arguments):
            for i in range(nr_samples):
                outputs = domain.get_external_function(f'generator::{gen_name}')(*comb)
                for output, output_def in zip(outputs, gen_def.generates):
                    assert isinstance(output_def.return_type, NamedValueType)
                    current[output_def.return_type.typename].append(wrap_value(output, output_def.return_type))

