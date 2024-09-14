"""Module for the Rule class."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(kw_only=True, slots=True)
class Rule:
    """Class for a rule to update cells."""

    target_slice: TargetSlice
    """The slice which the rule applies to."""

    rule_func: RuleFunc
    """The function which creates the mask generator."""

    to_state: StateBase | tuple[StateBase, ...]
    """The state to change to."""

    frequency: int | str
    """The frequency of the rule in frames. If a string is provided, it references a `FrequencySetting` by name."""

    predicate: Callable[[Automaton], bool] = lambda _: True
    """Optional predicate to determine if the rule should be applied."""

    random_multiplier: float = 1.0
    """Optional random multiplier for the rule."""

    _frequency_setting: FrequencySetting = field(init=False, repr=False)
    """Optional frequency setting for the rule. Only set if `frequency` is a string."""

    target_view: GridView = field(init=False, repr=False)
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

    @property
    def rule_tuple(self) -> RuleTuple:
        """Return the rule as a tuple."""
        states: int | tuple[int, ...] = (
            self.to_state.state
            if not isinstance(self.to_state, tuple)
            else tuple(ts.state for ts in self.to_state)
        )

        return (
            self.target_slice,
            self.mask_generator,
            states,
            self.predicate,
            self.random_multiplier,
        )
