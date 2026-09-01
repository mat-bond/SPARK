from __future__ import annotations

import pytest

from utils import build_lut_from_contig, chain_offsets_from_lut, global_index, parse_contig


def test_parse_contig_fixed_range():
    assert parse_contig("A10-19", linker_length=8) == 10


def test_parse_contig_link_halflink_and_des():
    assert parse_contig("A1-3/LINK/HALFLINK/DES", linker_length=8) == 16


def test_parse_contig_zero_stops_parsing():
    assert parse_contig("A1-3/0/A4-100", linker_length=8) == 3


def test_parse_contig_rejects_bad_segment():
    with pytest.raises(ValueError, match="Bad contig segment"):
        parse_contig("A1-3/NOT_A_SEGMENT", linker_length=8)


def test_build_lut_simple_linker_mapping():
    lut = build_lut_from_contig(4, ["A10-12/LINK/B20-21"])
    assert lut == [
        ("A10-12", "A1-3"),
        ("LINK1", "A4-7"),
        ("B20-21", "A8-9"),
    ]


def test_build_lut_break_starts_new_output_chain():
    lut = build_lut_from_contig(4, ["A1-2/LINK/BREAK/B1-3"])
    assert lut == [
        ("A1-2", "A1-2"),
        ("LINK1", "A3-6"),
        ("B1-3", "B1-3"),
    ]


def test_build_lut_rejects_design_token():
    with pytest.raises(ValueError, match="Fixed residue contig not accepted"):
        build_lut_from_contig(4, ["A1-2/DESIGN/A3-4"])


def test_chain_offsets_are_based_on_true_chain_lengths():
    lut = [
        ("A1-2", "A1-2"),
        ("LINK1", "A3-5"),
        ("B1-4", "B1-4"),
    ]
    assert chain_offsets_from_lut(lut) == {0: 0, 1: 5}


def test_global_index_maps_chain_local_numbering_to_zero_based_global_index():
    offsets = {0: 0, 1: 5}
    assert global_index(offsets, "A", 1) == 0
    assert global_index(offsets, "A", 5) == 4
    assert global_index(offsets, "B", 1) == 5
    assert global_index(offsets, "B", 4) == 8

def test_build_lut_rejects_multiple_breaks_in_one_input_chain():
    with pytest.raises(ValueError):
        build_lut_from_contig(4, ["A1-2/BREAK/B1-2/BREAK/C1-2"])

def test_build_lut_allows_one_break_per_input_chain():
    lut = build_lut_from_contig(
        4,
        [
            "A1-2/BREAK/B1-2",
            "C1-2/BREAK/D1-2",
        ],
    )

    assert lut == [
        ("A1-2", "A1-2"),
        ("B1-2", "B1-2"),
        ("C1-2", "C1-2"),
        ("D1-2", "D1-2"),
    ]