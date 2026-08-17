"""Tests for :mod:`raddb.hc_mapping` — the hydrometeor-class constants.

The module declares no functions, only six index-aligned constants.  Their whole value is
the alignment: ``HC_MCH`` and ``HC_PYART`` are both stored in parquet on the same 1-based
scale, so parquet integer ``k`` means ``HC_MAP_DICT[k - 1]``.  An off-by-one here
mislabels every classified gate in every plot, silently.
"""

from __future__ import annotations

from raddb.hc_mapping import (
    HC_CLASSES,
    HC_COLOR_BY_LABEL,
    HC_COLORS,
    HC_MAP_DICT,
    PYART_TO_OPE,
)

N_CLASSES = 9
"""Operational classes 0-8, stored in parquet as 1-9."""


def test_hc_map_dict_is_a_contiguous_zero_based_range():
    """Keys 0..8 with no gaps — the ``k - 1`` indexing depends on it."""
    assert sorted(HC_MAP_DICT) == list(range(N_CLASSES))
    assert HC_MAP_DICT[0] == "None"


def test_hc_classes_is_hc_map_dict_in_order():
    """``HC_CLASSES[i]`` is ``HC_MAP_DICT[i]``; index 0 corresponds to parquet 1."""
    assert HC_CLASSES == [HC_MAP_DICT[k] for k in range(N_CLASSES)]
    assert len(HC_CLASSES) == N_CLASSES


def test_class_labels_are_unique():
    """Two classes sharing a label would make a legend ambiguous."""
    assert len(set(HC_CLASSES)) == N_CLASSES


def test_hc_colors_is_index_aligned_with_hc_classes():
    """One colour per class, same order — this is what the plots zip together."""
    assert len(HC_COLORS) == len(HC_CLASSES)
    assert len(set(HC_COLORS)) == N_CLASSES, "a repeated colour makes two classes indistinguishable"


def test_hc_colors_are_recognised_by_matplotlib():
    """Every entry must actually resolve; a typo only shows up at plot time."""
    from matplotlib.colors import to_rgba

    for colour in HC_COLORS:
        assert len(to_rgba(colour)) == 4


def test_hc_color_by_label_matches_the_two_lists():
    """The convenience lookup is exactly ``zip(HC_CLASSES, HC_COLORS)``."""
    assert HC_COLOR_BY_LABEL == dict(zip(HC_CLASSES, HC_COLORS))
    assert len(HC_COLOR_BY_LABEL) == N_CLASSES


def test_pyart_to_ope_covers_every_pyart_class():
    """Py-ART's native hydro classes are 1-9; all nine must map."""
    assert sorted(PYART_TO_OPE) == list(range(1, 10))


def test_pyart_to_ope_lands_inside_the_operational_scale():
    """Targets are operational 1-8; class 0 (``None``) is never a Py-ART output."""
    assert set(PYART_TO_OPE.values()) == set(range(1, 9))


def test_cr_and_vi_are_the_only_merged_pair():
    """Py-ART separates CR (2) and VI (6); the operational scale merges both into CR/VI."""
    merged = [k for k, v in PYART_TO_OPE.items() if list(PYART_TO_OPE.values()).count(v) > 1]
    assert sorted(merged) == [2, 6]
    assert PYART_TO_OPE[2] == PYART_TO_OPE[6] == 1
    assert HC_MAP_DICT[1] == "CR/VI"


def test_remapped_pyart_values_index_a_real_label():
    """After the remap, ``HC_MAP_DICT[PYART_TO_OPE[k]]`` always resolves."""
    for pyart_class, operational in PYART_TO_OPE.items():
        assert operational in HC_MAP_DICT, f"PyART class {pyart_class} maps outside the label table"
