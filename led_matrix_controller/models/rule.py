"""Module for the Rule class."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from content.setting import FrequencySetting  # noqa: TCH002

if TYPE_CHECKING:
    from content.automaton import (
        Automaton,
        MaskGen,
        RuleFunc,
        RuleTuple,
        TargetSlice,
    )
    from content.base import GridView, StateBase


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

    random_multiplier: float = 1.0
    """Optional random multiplier for the rule."""

    _frequency_setting: FrequencySetting = field(init=False, repr=False, hash=False)
    """Optional frequency setting for the rule. Only set if `frequency` is a string."""

    target_view: GridView = field(init=False, repr=False, hash=False)
    """The view of the grid which the rule applies to."""

    mask_generator: MaskGen = field(init=False, repr=False)
    """The mask generator for the rule."""

    def active_on_frame(self, i: int, /) -> bool:
        """Return whether the rule is active on the given frame."""
        current_frequency = (
            self.frequency
            if isinstance(self.frequency, int)
            else self._frequency_setting.value
        )

        return bool(current_frequency) and i % current_frequency == 0

    def rule_tuple(self) -> RuleTuple:
        """Return the rule as a tuple."""
        return _rule_tuple(self)

    @classmethod
    def clear_cached_rule_tuples(cls) -> None:
        """Clear the cached rule tuples."""
        _rule_tuple.cache_clear()


@lru_cache
def _rule_tuple(rule: Rule) -> RuleTuple:
    """Return the rule as a tuple."""
    states: int | tuple[int, ...] = (
        rule.to_state.state
        if not isinstance(rule.to_state, tuple)
        else tuple(ts.state for ts in rule.to_state)
    )

    return (
        rule.target_slice,
        rule.mask_generator,
        states,
        rule.predicate,
        rule.random_multiplier,
    )
