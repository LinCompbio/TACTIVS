import numpy as np

from tactivs.episodes import build_episode_plan, episode_parent_masks
from tactivs.evaluation import retain_complete_protocol_targets


def test_fixed_random_ranks_every_non_support_molecule():
    parents = np.asarray(["a1", "a2", "a3", "a4", "d1", "d2"])
    visible, ranked = episode_parent_masks(
        parents, ["a1", "a2"], protocol="fixed_random"
    )

    assert visible.tolist() == [True] * 6
    assert parents[ranked].tolist() == ["a3", "a4", "d1", "d2"]


def test_series_protocol_removes_unselected_support_series_members():
    parents = np.asarray(["a1", "a2", "a3", "d1", "d2"])
    series = {"a1": "s1", "a2": "s1", "a3": "s2"}
    visible, ranked = episode_parent_masks(
        parents, ["a1"], protocol="series", series_of_parent=series
    )

    assert parents[visible].tolist() == ["a1", "a3", "d1", "d2"]
    assert parents[ranked].tolist() == ["a3", "d1", "d2"]


def test_random_support_order_is_independent_of_manifest_row_order():
    records = {f"a{i}": {} for i in range(6)}
    reversed_records = dict(reversed(list(records.items())))

    first = build_episode_plan({"t": records}, ["fixed_random"], seeds=[0], max_k=3)
    second = build_episode_plan(
        {"t": reversed_records}, ["fixed_random"], seeds=[0], max_k=3
    )

    assert (
        first["fixed_random"]["t"]["0"]["support_order"]
        == second["fixed_random"]["t"]["0"]["support_order"]
    )


def test_incomplete_series_target_is_removed_as_a_complete_cohort():
    rows = []
    for target, ks in (("complete", [1, 2]), ("partial", [1])):
        for config in ("full", "direct_only"):
            for k in ks:
                rows.append(
                    {
                        "config": config,
                        "target_id": target,
                        "protocol": "series",
                        "seed": 0,
                        "K": k,
                    }
                )
    rows.append(
        {
            "config": "full",
            "target_id": "partial",
            "protocol": "fixed_random",
            "seed": 0,
            "K": 1,
        }
    )

    filtered = retain_complete_protocol_targets(rows, "series", [0], [1, 2])

    assert {row["target_id"] for row in filtered if row["protocol"] == "series"} == {
        "complete"
    }
    assert any(row["protocol"] == "fixed_random" for row in filtered)
