"""Module for the Rule class."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from content.setting import FrequencySetting, ParameterSetting

if TYPE_CHECKING:
    from content.automaton import (
        Automaton,
        MaskGen,
        RuleFunc,
        RuleTuple,
        TargetSlice,
    )
    from content.base import GridView, StateBase


class AttributeVisitor(ast.NodeVisitor):
    """Visitor to extract attributes from a function."""

    def __init__(self, grid_arg_name: str) -> None:
        self.grid_arg_name = grid_arg_name
        self.attributes: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Visit an attribute node."""
        if isinstance(node.value, ast.Name) and node.value.id == self.grid_arg_name:
            # Record the consumption of the grid's attribute
            self.attributes.add(node.attr)

        self.generic_visit(node)


@dataclass(kw_only=True, slots=True, unsafe_hash=True)
class Rule:
    """Class for a rule to update cells."""

    target_slice: TargetSlice = field(hash=False)
    """The slice which the rule applies to."""

    rule_func: RuleFunc
    """The function which creates the mask generator."""

    to_state: StateBase | tuple[StateBase, ...]
    """The state to change to."""

    frequency: int | str
    """The frequency of the rule in frames. If a string is provided, it references a `FrequencySetting` by name."""

    predicate: Callable[[Automaton], bool] = field(hash=False, default=lambda _: True)
    """Optional predicate to determine if the rule should be applied."""

    random_multiplier: float | str = 1.0
    """Optional random multiplier for the rule."""

    _frequency_setting: FrequencySetting = field(init=False, repr=False, hash=False)
    """Optional frequency setting for the rule. Only set if `frequency` is a string."""

    target_view: GridView = field(init=False, repr=False, hash=False)
    """The view of the grid which the rule applies to."""

    mask_generator: MaskGen = field(init=False, repr=False)
    """The mask generator for the rule."""

    consumed_parameters: set[str] = field(init=False, repr=False, hash=False)
    """The slugs of the ParameterSettings consumed by the rule function."""

    def active_on_frame(self, i: int, /) -> bool:
        """Return whether the rule is active on the given frame."""
        current_frequency = (
            self.frequency
            if isinstance(self.frequency, int)
            else self._frequency_setting.value
        )

        return bool(current_frequency) and i % current_frequency == 0

    def refresh_mask_generator(self, automaton: Automaton) -> None:
        """Refresh the mask generator for the rule.

        Also sets the consumed_parameters attribute if it hasn't been set yet.
        """
        if not hasattr(self, "consumed_parameters"):
            grid_arg_name = self.rule_func.__code__.co_varnames[0]
            visitor = AttributeVisitor(grid_arg_name)
            visitor.visit(ast.parse(inspect.getsource(self.rule_func)))

            self.consumed_parameters = {
                va
                for va in visitor.attributes
                if isinstance(automaton.settings.get(va), ParameterSetting)
            }

        self.mask_generator = self.rule_func(automaton, self.target_slice)

    def rule_tuple(self, ca: Automaton) -> RuleTuple:
        """Return the rule as a tuple."""
        return _rule_tuple(self, ca)

    @classmethod
    def clear_cached_rule_tuples(cls) -> None:
        """Clear the cached rule tuples."""
        _rule_tuple.cache_clear()


@lru_cache
def _rule_tuple(rule: Rule, ca: Automaton) -> RuleTuple:
    """Return the rule as a tuple."""
    states: int | tuple[int, ...] = (
        rule.to_state.state
        if not isinstance(rule.to_state, tuple)
        else tuple(ts.state for ts in rule.to_state)
    )

    if isinstance(rule.random_multiplier, str):
        rand_mult = float(getattr(ca, rule.random_multiplier))
    else:
        rand_mult = rule.random_multiplier

    return (
        rule.target_slice,
        rule.mask_generator,
        states,
        rule.predicate,
        rand_mult,
    )
