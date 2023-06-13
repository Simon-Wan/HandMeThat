#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : state.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/28/2022
#
# This file is part of HACL-PyTorch.
# Distributed under terms of the MIT license.

import torch
import jacinle

from jacinle.utils.printing import indent_text, kvformat
from typing import Optional, Union, Tuple, Sequence, Mapping, Dict
from .value import NamedValueType, NamedValueTypeSlot, ObjectType, PyObjValueType, BOOL, Variable, Value, concat_values

__all__ = [
    'MultidimensionalArrayInterface', 'ValueDict',
    'StateLike', 'SingleStateLike',
    'State', 'BatchState',
    'concat_states', 'concat_batch_states'
]


class MultidimensionalArrayInterface(object):
    """
    A multi-dimensional array inferface. At a high-level, this can be interpreted as a dictionary that maps
    feature names (keys) to multi-diemsntional tensors (value).
    """

    def __init__(self, all_feature_names):
        self.all_feature_names = set(all_feature_names)

    def clone(self):
        raise NotImplementedError()

    def get_feature(self, name: Union[str, int]) -> Value:
        raise NotImplementedError()

    def _set_feature_impl(self, name: str, feature: Value):
        raise NotImplementedError()

    def set_feature(self, name: str, feature: Value):
        if name not in self.all_feature_names:
            self.all_feature_names.add(name)
        self._set_feature_impl(name, feature)

    def update_feature(self, other_tensor_dict: Mapping[str, Value]):
        for key, value in other_tensor_dict.items():
            self.set_feature(key, value)

    def __contains__(self, item: str) -> bool:
        return item in self.all_feature_names

    def __getitem__(self, name: str) -> Value:
        return self.get_feature(name)

    def __setitem__(self, key, value):
        return self.set_feature(key, value)

    def keys(self):
        yield from self.all_feature_names

    def values(self):
        for key in self.all_feature_names:
            yield self.get_feature(key)

    def items(self):
        for key in self.all_feature_names:
            yield key, self.get_feature(key)


class ValueDict(MultidimensionalArrayInterface):
    """Basic tensor dict implementation."""
    def __init__(self, tensor_dict: Optional[Dict[str, Value]] = None):
        if tensor_dict is None:
            tensor_dict = dict()

        all_feature_names = set(tensor_dict.keys())
        super().__init__(all_feature_names)
        self.tensor_dict = tensor_dict

    def clone(self):
        return type(self)({k: v.clone() for k, v in self.tensor_dict.items()})

    def get_feature(self, name: Union[str, int]):
        return self.tensor_dict[name]

    def _set_feature_impl(self, name, feature: Value):
        self.tensor_dict[name] = feature

    def __contains__(self, item):
        return item in self.tensor_dict


class StateTensorAccessor(object):
    def __init__(self, state):
        self.state = state

    def __getitem__(self, item):
        return self.state.features[item].tensor


class StateLike(object):
    @property
    def batch_dims(self) -> int:
        raise NotImplementedError()

    def get_typename(self, name):
        raise NotImplementedError()

    def get_typed_index(self, name):
        raise NotImplementedError()

    def get_nr_objects_by_type(self, typename):
        raise NotImplementedError()

    @property
    def features(self) -> MultidimensionalArrayInterface:
        raise NotImplementedError()

    @property
    def tensors(self) -> StateTensorAccessor:
        return StateTensorAccessor(self)

    def clone(self):
        raise NotImplementedError()

    def index(self, feature_name, arguments):
        return self.features[feature_name][arguments]

    def set_index(self, feature_name, arguments, value):
        self.features[feature_name][arguments] = value

    def get_feature(self, name: str):
        return self.features[name]

    def __getitem__(self, name: str):
        return self.get_feature(name)

    def __str__(self):
        raise NotImplementedError()

    __repr__ = jacinle.repr_from_str


class SingleStateLike(StateLike):
    def __init__(self, object_types: Sequence[ObjectType], features: Optional[MultidimensionalArrayInterface] = None, object_names: Optional[Sequence[str]] = None):
        self._object_types = object_types
        self._features = features
        self._object_names = object_names

        if self._features is None:
            self._features = ValueDict()

        if self._object_names is not None:
            assert len(self._object_names) == len(self._object_types)
            self._object_type2name = dict()
            self._object_name2index = dict()
            for name, type in zip(self._object_names, self._object_types):
                typename = type.typename
                if typename not in self._object_type2name:
                    self._object_type2name[typename] = list()
                self._object_name2index[name] = (typename, len(self._object_type2name[typename]))
                self._object_type2name[typename].append(name)
        else:
            self._object_name2index = dict()
            self._object_type2name = dict()
        self.internals = dict()

    @property
    def batch_dims(self) -> int:
        """The number of batchified dimensions. For the basic State, it should be 0."""
        return 0

    @property
    def nr_objects(self) -> int:
        """The number of objects in the current state."""
        return len(self._object_types)

    @property
    def object_types(self) -> Sequence[ObjectType]:
        """A list of object types."""
        return self._object_types

    @property
    def object_names(self) -> Sequence[str]:
        """A list of object names."""
        return self._object_names

    @property
    def object_type2names(self) -> Mapping[str, Sequence[str]]:
        """Return a mapping from typename (in string) to a list of objects of this type. For example:
            `state.object_type2name['location'][i]` returns the object name of the i-th location in the state.

        Returns:
            Mapping[str, Sequence[str]]: the mapping.
        """
        return self._object_type2name

    @property
    def object_name2index(self) -> Mapping[str, Tuple[str, int]]:
        """Return a mapping from the object name to a tuple of (typename, the index under that typename).
            That is, `state.object_name2index[name] == (typename, index)` iff. `state.object_type2name[typename][index] = name`.
        """
        return self._object_name2index

    def get_typename(self, name: str) -> str:
        return self._object_name2index[name][0]

    def get_typed_index(self, name: str) -> int:
        return self._object_name2index[name][1]

    def get_nr_objects_by_type(self, typename: str) -> int:
        return len(self.object_type2names[typename]) if typename in self.object_type2names else 0

    @property
    def features(self):
        return self._features

    def clone(self):
        return type(self)(self._object_types, self._features.clone(), self._object_names)

    def __str__(self):
        if self.object_names is not None:
            objects_str = [f'{name} - {type.typename}' for name, type in zip(self.object_names, self.object_types)]
        else:
            objects_str = self.object_names
        fmt = f'''{type(self).__name__}{{
  states:
'''
        for p in self.features.all_feature_names:
            tensor = self.features[p]
            fmt += f'    - {p}'
            fmt += ': ' + indent_text(str(tensor), level=2).strip() + '\n'
        fmt += f"  objects: {', '.join(objects_str)}\n"
        fmt += self.extra_state_str()
        fmt += '}'
        return fmt

    def extra_state_str(self):
        return ''


class State(SingleStateLike):
    def clone(self):
        rv = type(self)(self._object_types, self._features.clone(), self._object_names)
        rv.internals = self.clone_internals()
        return rv

    def clone_internals(self):
        return dict()

    def make_quantized(self, domain, features=None):
        assert isinstance(self.features, ValueDict), 'Only TensorDict is supported for automatic quantization.'

        if features is None:
            features = [name for name in self.features.all_feature_names if domain.features[name].is_state_variable]

        new_tensor_dict = ValueDict()
        for feature_name in features:
            new_tensor_dict[feature_name] = self.features[feature_name].make_quantized()
        return type(self)(self.object_types, new_tensor_dict, self.object_names)

    def define_context(self, domain, session=None):
        from .session import TensorDictDefHelper
        return TensorDictDefHelper(domain, self, session=session)

    def generate_tuple_description(self, domain):
        assert isinstance(self.features, ValueDict), 'Only TensorDict is supported for automatic tuple description.'

        rv = list()
        for feature_name in sorted(self.features.all_feature_names):
            if domain.predicates[feature_name].is_state_variable:
                feature = self.features[feature_name]
                assert feature.quantized, 'Can only generate tuple description for quantized states.'
                rv.extend(self.features[feature_name].tensor.flatten().tolist())
        return tuple(rv)


class BatchState(StateLike):
    def __init__(self, nr_objects_per_type, features, max_nr_objects_per_type=None, object_name2index=None):
        self._nr_objects_per_type = nr_objects_per_type
        self._max_nr_objects_per_type = max_nr_objects_per_type
        self._object_name2index = object_name2index
        self._features = features

        if self._max_nr_objects_per_type is None:
            self._max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

    @classmethod
    def from_states(cls, domain, states: Sequence[State]):
        return concat_states(*states)

    @property
    def batch_dims(self) -> int:
        return 1

    @property
    def batch_size(self) -> int:
        x = next(iter(self.features.tensor_dict.values()))
        return x.shape[0]

    @property
    def nr_objects_per_type(self) -> int:
        return self._nr_objects_per_type

    @property
    def max_nr_objects_per_type(self) -> int:
        return self._max_nr_objects_per_type

    @property
    def features(self):
        return self._features

    def clone(self):
        object_name2index = self.object_name2index.copy() if self.object_name2index is not None else None
        return type(self)(self.nr_objects_per_type.copy(), self.features.clone(), max_nr_objects_per_type=self.max_nr_objects_per_type.copy(), object_name2index=object_name2index)

    @property
    def object_name2index(self) -> Mapping[str, Tuple[str, int]]:
        """Return a mapping from the object name to a tuple of (typename, the index under that typename).
            That is, `state.object_name2index[name] == (typename, index)` iff. `state.object_type2name[typename][index] == name`.
        """
        return self._object_name2index

    def get_typename(self, name: str) -> str:
        return [record[name][0] for record in self._object_name2index]

    def get_typed_index(self, name: str) -> int:
        return [record[name][1] for record in self._object_name2index]

    def get_nr_objects_by_type(self, typename: str) -> Sequence[int]:
        return self._nr_objects_per_type[typename]

    def __str__(self):
        fmt = f'''{type(self).__name__}{{
  nr_objects_per_type:
'''
        for typename, number in self.nr_objects_per_type.items():
            fmt += f'    - {typename}: {number.tolist()}\n'
        fmt += '''
  states:
'''
        fmt += indent_text(kvformat(self.features.tensor_dict), 2, tabsize=2).rstrip() + '\n'
        fmt += '}'
        return fmt

    __repr__ = jacinle.repr_from_str


def concat_states(*args: State) -> BatchState:
    """Concatenate a list of states into a batch state."""

    assert len(args) > 0
    all_features = list(args[0].features.all_feature_names)

    # 1. Sanity checks.
    for state in args[1:]:
        assert len(all_features) == len(state.features.all_feature_names)
        for feature in all_features:
            assert feature in state.features.all_feature_names

    # 2. Construct the nr_objects_pre_type dict.
    all_typenames = {t for t in state.object_type2names for state in args}
    nr_objects_per_type = {typename: list() for typename in all_typenames}
    for state in args:
        for typename in all_typenames:
            if typename in state.object_type2names:
                nr_objects_per_type[typename].append(len(state.object_type2names[typename]))
            else:
                nr_objects_per_type[typename].append(0)

    # 3. Compute the max_nr_objects_per_type.
    for typename, nr_list in nr_objects_per_type.items():
        nr_objects_per_type[typename] = torch.tensor(nr_list, dtype=torch.int64)
    max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

    # 4. Put the same feature into a list.
    features = {feature_name: list() for feature_name in all_features}
    for state in args:
        assert isinstance(state.features, ValueDict), 'Only TensorDict is implemented for BatchState.from_states.'
        for key, value in state.features.tensor_dict.items():
            features[key].append(value)

    # 5. Actually, compute the features.
    feature_names = list(features.keys())
    for feature_name in feature_names:
        features[feature_name] = concat_values(*features[feature_name])
    return BatchState(nr_objects_per_type, ValueDict(features), max_nr_objects_per_type=max_nr_objects_per_type, object_name2index=[state.object_name2index for state in args])


def concat_batch_states(*args: BatchState) -> BatchState:
    """Concatenate a list of batch states into a single batch state."""

    assert len(args) > 0
    all_typenames = list(args[0].nr_objects_per_type)
    all_features = list(args[0].features.all_feature_names)

    nr_objects_per_type = {typename: list() for typename in args[0].nr_objects_per_type}
    for arg in args:
        for typename in all_typenames:
            nr_objects_per_type[typename].append(arg.nr_objects_per_type[typename])

    for typename, nr_list in nr_objects_per_type.items():
        nr_objects_per_type[typename] = torch.cat(nr_list, dim=0)
    max_nr_objects_per_type = {key: value.max().item() for key, value in nr_objects_per_type.items()}

    features = {feature_name: list() for feature_name in all_features}
    for arg in args:
        assert isinstance(arg.features, ValueDict), 'Only TensorDict is implemented for BatchState.from_states.'
        for key, value in arg.features.tensor_dict.items():
            features[key].append(value)
    for key, values in features:
        features[key] = concat_values(*values)

    return BatchState(
        nr_objects_per_type,
        ValueDict(features),
        max_nr_objects_per_type=max_nr_objects_per_type,
        object_name2index=sum([state.object_name2index for state in args], start=[])
    )

