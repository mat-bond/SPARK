from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from constants import Stats
from stages import analysis
from utils import build_lut_from_contig


CONTIG = "A1-2/LINK/A3-4"
LINKER_LENGTH = 2
LUT = build_lut_from_contig(LINKER_LENGTH, CONTIG.split())


def test_calculate_pae_averages_only_selected_designed_residues(tmp_path):
    matrix = np.arange(36, dtype=float).reshape(6, 6)
    npz = tmp_path / "pae.npz"
    np.savez(npz, predicted_aligned_error=matrix)

    designed, full = analysis.calculate_pae(
        residue_contig="LINK",
        contig=CONTIG,
        linker_length=LINKER_LENGTH,
        predicted_file_path=str(npz),
        lut=LUT,
        desired_quantity="predicted_aligned_error",
        second_name="pae",
    )

    assert designed == pytest.approx(matrix[2:4, 2:4].mean())
    assert full == pytest.approx(matrix.mean())


def test_calculate_average_pde_counts_upper_triangle_once(tmp_path):
    matrix = np.zeros((6, 6), dtype=float)
    matrix[2:4, 2:4] = np.array([[1.0, 2.0], [2.0, 3.0]])

    npz = tmp_path / "pde.npz"
    np.savez(npz, predicted_distance_error=matrix)

    result = analysis.calculate_average_pde(
        residue_contig="LINK",
        contig=CONTIG,
        linker_length=LINKER_LENGTH,
        npz_path=str(npz),
        lut=LUT,
    )
    assert result == pytest.approx(2.0)


def test_calculate_designed_avg_plddt_uses_selected_residues(tmp_path):
    plddt = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    npz = tmp_path / "plddt.npz"
    np.savez(npz, plddt=plddt)

    result = analysis.calculate_designed_avg_plddt(
        residue_contig="LINK",
        contig=CONTIG,
        linker_length=LINKER_LENGTH,
        predicted_file_path=str(npz),
        lut=LUT,
    )
    assert result == pytest.approx(35.0)


def _stats_row(path: str, *, rmsd: float, plddt: float, pae: float, pde: float, length: int = 10):
    row = [0] * len(Stats)
    row[Stats.PATH] = path
    row[Stats.RMSD] = rmsd
    row[Stats.RF_RMSD] = 0.0
    row[Stats.DES_PAE] = pae
    row[Stats.DES_PDE] = pde
    row[Stats.DES_PLDDT] = plddt
    row[Stats.FULL_PAE] = pae
    row[Stats.FULL_PDE] = pde
    row[Stats.FULL_PLDDT] = plddt
    row[Stats.FIX_TOTAL_VOL] = 0.0
    row[Stats.FIX_CAVITYAVG] = 0.0
    row[Stats.FIX_CAVITYCOUNT] = 0
    row[Stats.DES_TOTAL_VOL] = 0.0
    row[Stats.DES_CAVITYAVG] = 0.0
    row[Stats.DES_CAVITYCOUNT] = 0
    row[Stats.LENGTH] = length
    return tuple(row)


def test_run_analysis_applies_cutoffs_and_final_rmsd_ranking(tmp_path, monkeypatch):
    rows = [
        _stats_row("keep_best.pdb", rmsd=1.0, plddt=0.90, pae=3.0, pde=2.0),
        _stats_row("keep_second.pdb", rmsd=1.5, plddt=0.85, pae=4.0, pde=2.5),
        _stats_row("fail_rmsd.pdb", rmsd=2.5, plddt=0.95, pae=2.0, pde=1.0),
        _stats_row("fail_plddt.pdb", rmsd=1.2, plddt=0.50, pae=2.0, pde=1.0),
        _stats_row("fail_pae.pdb", rmsd=1.1, plddt=0.90, pae=8.0, pde=1.0),
        _stats_row("fail_pde.pdb", rmsd=1.1, plddt=0.90, pae=2.0, pde=8.0),
    ]

    captured = {}
    monkeypatch.setattr(analysis, "get_stats_from_file", lambda args: rows.copy())
    monkeypatch.setattr(
        analysis,
        "process_best_designs",
        lambda args, designs: captured.setdefault("designs", list(designs)),
    )
    monkeypatch.setattr(analysis, "create_stats_xlsx", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "create_stats_graphs", lambda *a, **k: None)

    args = SimpleNamespace(
        stats_read_file_only=True,
        rmsd_cutoff=2.0,
        plddt_cutoff=0.70,
        pae_cutoff=5.0,
        pde_cutoff=3.0,
        final_selection_amount=1,
        filtered_designs_folder=str(tmp_path),
    )

    analysis.run_analysis(args)

    assert len(captured["designs"]) == 1
    assert captured["designs"][0][Stats.PATH] == "keep_best.pdb"

def test_none_cutoff_means_filter_is_disabled(tmp_path, monkeypatch):
    rows = [_stats_row("candidate.pdb", rmsd=1.0, plddt=0.5, pae=20.0, pde=20.0)]
    captured = {}

    monkeypatch.setattr(analysis, "get_stats_from_file", lambda args: rows.copy())
    monkeypatch.setattr(
        analysis,
        "process_best_designs",
        lambda args, designs: captured.setdefault("designs", list(designs)),
    )
    monkeypatch.setattr(analysis, "create_stats_xlsx", lambda *a, **k: None)
    monkeypatch.setattr(analysis, "create_stats_graphs", lambda *a, **k: None)

    args = SimpleNamespace(
        stats_read_file_only=True,
        rmsd_cutoff=2.0,
        plddt_cutoff=None,
        pae_cutoff=None,
        pde_cutoff=None,
        final_selection_amount=10,
        filtered_designs_folder=str(tmp_path),
    )

    analysis.run_analysis(args)
    assert captured["designs"][0][Stats.PATH] == "candidate.pdb"
