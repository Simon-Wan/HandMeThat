#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : operator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/04/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

from typing import Optional, Sequence, Tuple, Union, ForwardRef

import jacinle
from jacinle.utils.printing import indent_text

from .expr import ExpressionExecutionContext, compose_bvdict_args
from .expr import AssignOp, ConditionalAssignOp, DeicticAssignOp, ValueOutputExpression, VariableAssignmentExpression
from .optimistic import EqualOptimisticConstraint, is_optimistic_value
from .state import State
from .value import NamedValueTypeSlot, Value, Variable

logger = jacinle.get_logger(__file__)

__all__ = ['Precondition', 'Effect', 'Operator', 'OperatorApplier']


Domain = ForwardRef('Domain')
Session = ForwardRef('Session')


class Precondition(object):
    def __init__(self, bool_expr: ValueOutputExpression):
        self.bool_expr = bool_expr
        self.ao_discretization = None

    def forward(self, ctx: ExpressionExecutionContext):
        return self.bool_expr.forward(ctx)

    def __str__(self):
        return str(self.bool_expr)

    __repr__ = jacinle.repr_from_str


class Effect(object):
    def __init__(self, assign_expr: VariableAssignmentExpression):
        self.assign_expr = assign_expr
        self.ao_discretization = None

    def forward(self, ctx: ExpressionExecutionContext):
        return self.assign_expr.forward(ctx)

    @property
    def unwrapped_assign_expr(self) -> Union[AssignOp, ConditionalAssignOp]:
        """Unwrap the DeicticAssignOps and return the innermost AssignOp."""
        expr = self.assign_expr
        if isinstance(expr, DeicticAssignOp):
            expr = expr.expr
        assert isinstance(expr, (AssignOp, ConditionalAssignOp))
        return expr

    def __str__(self):
        return str(self.assign_expr)

    __repr__ = jacinle.repr_from_str


class Operator(object):
    def __init__(
        self,
        domain: Domain,
        name: str,
        arguments: Sequence[Variable],
        preconditions: Sequence[Precondition],
        effects: Sequence[Effect],
        is_axiom: bool = False,
        is_template: bool = False,
        extends: Optional[str] = None,
        policy_identifier: Optional[str] = None,
        policy_arguments: Optional[Sequence[Variable]] = None,
    ):
        self.domain = domain
        self.name = name
        self.arguments = arguments
        self.preconditions = tuple(preconditions)
        self.effects = tuple(effects)
        self.is_axiom = is_axiom
        self.is_template = is_template
        self.extends = extends

        self.policy_identifier = policy_identifier
        self.policy_arguments = tuple(policy_arguments) if policy_arguments is not None else tuple()

    @property
    def nr_arguments(self) -> int:
        return len(self.arguments)

    def __call__(self, *args):
        output_args = list()
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == '??':
                output_args.append(NamedValueTypeSlot(self.arguments[i].dtype))
            else:
                output_args.append(arg)
        return OperatorApplier(self, *output_args)

    def apply_precondition(self, state: State, *args, session: Optional[Session] = None, ctx: Optional[ExpressionExecutionContext] = None, **kwargs) -> bool:
        """Apply the precondition of this operator to the given state.

        Args:
            state (State): The state to apply the precondition.
            *args: The arguments to the operator.
            session (Optional[Session]): The session to use.
            ctx (Optional[ExpressionExecutionContext]): The execution context to use.
            **kwargs: The additional arguments to the execution context.

        Returns:
            bool: Whether the precondition is satisfied.
        """

        from .session import get_session
        session = get_session(session)

        if ctx is None:
            ctx = ExpressionExecutionContext(session, state, bounded_variables=compose_bvdict_args(self.arguments, args, state=state, session=session), **kwargs)

        all_rvs = list()
        with ctx.as_default():
            for pred in self.preconditions:
                try:
                    pred_value = pred.forward(ctx)
                    all_rvs.append(pred_value)
                    rv = pred_value.item()
                    # print(f'{pred} = {rv}')
                except Exception:
                    logger.warning('Precondition evaluation failed: {}.'.format(pred.bool_expr))
                    raise
                if is_optimistic_value(rv):
                    if ctx.optimistic_context is not None:
                        ctx.optimistic_context.add_constraint(EqualOptimisticConstraint.from_bool(rv, True), note=f'precondition_test::{pred.bool_expr}')
                else:
                    if rv < 0.5:
                        return False
        return True

    def apply_effect(self, state: State, *args, clone: bool = True, session: Optional[Session] = None, ctx: Optional[ExpressionExecutionContext] = None, **kwargs) -> State:
        """Apply the effect of this operator to the given state.

        Args:
            state (State): The state to apply the effect.
            *args: The arguments to the operator.
            clone (bool): Whether to clone the state before applying the effect.
            session (Optional[Session]): The session to use.
            ctx (Optional[ExpressionExecutionContext]): The execution context to use.
            **kwargs: The additional arguments to the execution context.

        Returns:
            State: The new state after applying the effect.
        """
        from .session import get_session
        session = get_session(session)

        if clone:
            state = state.clone()
        if ctx is None:
            ctx = ExpressionExecutionContext(session, state, bounded_variables=compose_bvdict_args(self.arguments, args, state=state, session=session), **kwargs)
        with ctx.as_default():
            for effect in self.effects:
                try:
                    effect.forward(ctx)
                except Exception:
                    logger.warning('Effect application failed: {}.'.format(effect.assign_expr))
                    raise
            return state

    def apply(self, state: State, *args, clone: bool = True, session: Optional[Session] = None, **kwargs) -> Tuple[bool, State]:
        from .session import get_session
        session = get_session(session)

        ctx = None
        if not clone:  # if clone = True, the state binded to ctx in apply_effect will be a new state.
            ctx = ExpressionExecutionContext(session, state, bounded_variables=compose_bvdict_args(self.arguments, args, state=state, session=session), **kwargs)
        if self.apply_precondition(state, *args, session=session, ctx=ctx, **kwargs):
            return True, self.apply_effect(state, *args, clone=clone, session=session, ctx=ctx, **kwargs)
        return False, state

    def __str__(self):
        if not self.is_axiom:
            def_name = 'action'
        else:
            def_name = 'axiom'
        arg_string = ', '.join([str(arg) for arg in self.arguments])
        return f'{def_name}::{self.name}({arg_string})'

    __repr__ = jacinle.repr_from_str

    def pddl_str(self) -> str:
        if not self.is_axiom:
            def_name, def_name_a, def_name_p, def_name_e = f'action {self.name}', 'parameters', 'precondition', 'effect'
        else:
            def_name, def_name_a, def_name_p, def_name_e = 'axiom', 'vars', 'context', 'implies'
        arg_string = ' '.join([str(arg) for arg in self.arguments])
        pre_string = '\n'.join([indent_text(str(pre), 2, tabsize=2) for pre in self.preconditions])
        eff_string = '\n'.join([indent_text(str(eff), 2, tabsize=2) for eff in self.effects])
        return f'''(:{def_name}
  :{def_name_a} ({arg_string})
  :{def_name_p} (and
    {pre_string.lstrip()}
  )
  :{def_name_e} (and
    {eff_string.lstrip()}
  )
)'''


class OperatorApplier(object):
    def __init__(self, operator: Operator, *args):
        self.operator = operator
        self.arguments = args

        if len(self.arguments) != len(self.operator.arguments):
            raise ValueError(f'The number of arguments does not match the operator: {self}')

    @property
    def name(self):
        return self.operator.name

    def apply_precondition(self, state: State, session: Optional[Session] = None, **kwargs) -> bool:
        return self.operator.apply_precondition(state, *self.arguments, session=session, **kwargs)

    def apply_effect(self, state: State, clone: bool = True, session: Optional[Session] = None, **kwargs) -> State:
        return self.operator.apply_effect(state, *self.arguments, clone=clone, session=session, **kwargs)

    def __call__(self, state: State, clone: bool = True, session: Optional[Session] = None, **kwargs) -> Tuple[bool, State]:
        return self.operator.apply(state, *self.arguments, clone=clone, session=session, **kwargs)

    def __str__(self):
        if not self.operator.is_axiom:
            def_name = 'action'
        else:
            def_name = 'axiom'
        arg_string = ', '.join([
            arg_def.name + '=' + (arg.short_str() if isinstance(arg, Value) else str(arg))
            for arg_def, arg in zip(self.operator.arguments, self.arguments)
        ])
        return f'{def_name}::{self.operator.name}({arg_string})'

    __repr__ = jacinle.repr_from_str

