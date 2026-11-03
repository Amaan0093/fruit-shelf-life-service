"""
Convert ripeness-stage labels (1..N) into shelf-life targets in days.

The mapping is taken from the proposal:
    Stage 1 (unripe)         -> longest shelf life (fruit-specific max)
    Stage N (overripe/spoiled) -> 0 days
For fruits without an explicit mapping table we use linear interpolation
between max_days and 0.
"""
from __future__ import annotations

from config import FRUITS


def stage_to_days(fruit: str, stage: int) -> float:
    """Return the predicted remaining shelf life (days) for a given fruit/stage.

    Parameters
    ----------
    fruit : str
        Fruit name, must be a key in config.FRUITS.
    stage : int
        Ripeness stage, 1-indexed (1 = freshest, N = spoiled).

    Returns
    -------
    float
        Estimated remaining shelf life in days.
    """
    if fruit not in FRUITS:
        raise KeyError(f"Unknown fruit: {fruit!r}. Known: {list(FRUITS)}")

    meta = FRUITS[fruit]
    n_stages = meta["stages"]
    max_days = meta["max_days"]

    if not (1 <= stage <= n_stages):
        raise ValueError(
            f"Stage {stage} out of range for {fruit} (expected 1..{n_stages})"
        )

    # Explicit mapping wins where supplied.
    if meta["mapping"] is not None:
        return float(meta["mapping"][stage])

    # Linear schedule: stage 1 -> max_days, stage N -> 0.
    if n_stages == 1:
        return float(max_days)
    return float(max_days) * (n_stages - stage) / (n_stages - 1)


def build_mapping_table() -> dict[str, dict[int, float]]:
    """Return the full {fruit: {stage: days}} table for all configured fruits.

    Useful for sanity checks and for printing in the methods section of the
    write-up.
    """
    table: dict[str, dict[int, float]] = {}
    for fruit, meta in FRUITS.items():
        table[fruit] = {
            stage: stage_to_days(fruit, stage)
            for stage in range(1, meta["stages"] + 1)
        }
    return table


if __name__ == "__main__":
    # Pretty-print the mapping when run directly: python data/label_mapping.py
    table = build_mapping_table()
    for fruit, stages in table.items():
        days = ", ".join(f"S{s}={d:g}d" for s, d in stages.items())
        print(f"{fruit:11s} -> {days}")
