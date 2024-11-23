"""Benchmark the pre-commit hooks."""

from __future__ import annotations

import dataclasses
from itertools import islice, product
from typing import TYPE_CHECKING, Callable

import pytest
from content import RainingGrid

if TYPE_CHECKING:
    from content.automaton import MaskGen
    from pytest_codspeed import BenchmarkFixture  # type: ignore[import-untyped]

SIZES = [32]
LIMITS = [100]


@pytest.mark.parametrize(
    ("size", "limit", "test_id"),
    [
        pytest.param(
            size,
            limit,
            test_id := f"{limit} frame{'s' if limit > 1 else ''} @ {size}x{size}",
            id=test_id,
        )
        for size, limit in product(
            SIZES,
            LIMITS,
        )
    ],
)
def test_raining_grid_simulation(
    benchmark: BenchmarkFixture,
    size: int,
    limit: int,
    test_id: str,
) -> None:
    """Benchmark the CA."""
    grid = RainingGrid(
        height=size,
        width=size,
        rain_chance=0.025,
        rain_speed=1,
        splash_speed=1,
        id_override=test_id,
    )

    @benchmark  # type: ignore[misc]
    def bench() -> None:
        for _ in grid.islice(limit=limit):
            pass


@pytest.mark.parametrize(
    ("size", "limit", "rule", "test_id"),
    [
        pytest.param(
            size,
            limit,
            rule,
            test_id
            := f"{rule.__name__} for {limit} frame{'s' if limit > 1 else ''} @ {size}x{size}",
            id=test_id,
        )
        for size, limit, rule in product(
            SIZES,
            LIMITS,
            RainingGrid._RULE_FUNCTIONS,
        )
    ],
)
def test_rules(
    benchmark: BenchmarkFixture,
    size: int,
    limit: int,
    rule: Callable[..., MaskGen],
    test_id: str,
) -> None:
    """Test/benchmark each individual rule."""
    grid = RainingGrid(
        height=size,
        width=size,
        rain_chance=0.025,
        rain_speed=1,
        splash_speed=1,
        id_override=test_id,
    )

    # Discard the first `size` frames so all rules are effective (e.g. splashing)
    for _ in islice(grid, size + 10):
        pass

    expected_frame_index = size + 9
    assert grid.frame_index == expected_frame_index
    expected_frame_index += 1

    grids_to_eval = [
        dataclasses.replace(grid, id_override=f"{test_id}-{i}")
        for i, _ in enumerate(islice(grid, limit))
    ]

    mask_generators = [rule(grid) for grid in grids_to_eval]

    @benchmark  # type: ignore[misc]
    def bench() -> None:
        for mask_gen in mask_generators:
            mask_gen(grid.pixels)
