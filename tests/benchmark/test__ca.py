"""Benchmark the pre-commit hooks."""

from __future__ import annotations

from copy import deepcopy
from itertools import islice, product
from typing import TYPE_CHECKING, Callable

import pytest
from utils.cellular_automata import RainingGrid

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture  # type: ignore[import-untyped]
    from utils.cellular_automata.ca import MaskGen


@pytest.mark.parametrize(
    ("size", "limit"),
    [
        pytest.param(
            size,
            limit,
            id=f"{limit} frame{'s' if limit > 1 else ''} @ {size}x{size}",
            marks=pytest.mark.xdist_group(f"{size}-{limit}"),
        )
        for size, limit in product(
            [8, 16, 32, 64],
            [50, 250, 500],
        )
    ],
)
def test_raining_grid_simulation(
    benchmark: BenchmarkFixture,
    size: int,
    limit: int,
) -> None:
    """Benchmark the CA."""
    grid = RainingGrid(size, size, rain_chance=0.025, rain_speed=1, splash_speed=1)

    @benchmark  # type: ignore[misc]
    def bench() -> None:
        for _ in grid.run(limit=limit):
            pass


@pytest.mark.parametrize(
    ("size", "limit", "rule"),
    [
        pytest.param(
            size,
            limit,
            rule,
            id=f"{rule.__name__} for {limit} frame{'s' if limit > 1 else ''} @ {size}x{size}",
            marks=pytest.mark.xdist_group(f"{size}-{limit}-{rule.__name__}"),
        )
        for size, limit, rule in product(
            [8, 16, 32, 64],
            [50, 250, 500],
            RainingGrid._RULE_FUNCTIONS,
        )
    ],
)
def test_rules(
    benchmark: BenchmarkFixture,
    size: int,
    limit: int,
    rule: Callable[..., MaskGen],
) -> None:
    """Test/benchmark each individual rule."""
    grid = RainingGrid(size, size, rain_chance=0.025, rain_speed=1, splash_speed=1)

    # Discard the first H frames so all rules are effective (e.g. splashing)
    for _ in islice(grid.frames, size + 10):
        pass

    expected_frame_index = size + 9
    assert grid.frame_index == expected_frame_index
    expected_frame_index += 1

    grids_to_eval = [deepcopy(grid) for _ in islice(grid.frames, limit)]

    assert len(grids_to_eval) == limit

    for g in grids_to_eval:
        assert g.frame_index == expected_frame_index
        expected_frame_index += 1

    mask_generators = [rule(grid) for grid in grids_to_eval]

    @benchmark  # type: ignore[misc]
    def bench() -> None:
        for mask_gen in mask_generators:
            mask_gen()
