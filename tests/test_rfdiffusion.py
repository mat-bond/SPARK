from __future__ import annotations

from types import SimpleNamespace
from stages import rfdiffusion


def test_compute_chain_thresholds():
    assert rfdiffusion.compute_chain_thresholds(
        "A1-2/LINK/A3-4 B1-3", linker_length=4
    ) == [8, 11]


def test_compute_segment_info_marks_design_regions_unfixed():
    thresholds, fixed = rfdiffusion.compute_segment_info(
        "A1-2/LINK/A3-4 B1-2",
        linker_length=4,
        pm_redesign_designed_res_seq=False,
        designed_residues_contig="",
    )
    assert thresholds == [2, 6, 8, 10]
    assert fixed == [True, False, True, True]


def test_compute_segment_info_can_redesign_selected_fixed_segment():
    thresholds, fixed = rfdiffusion.compute_segment_info(
        "A1-2/LINK/A3-4",
        linker_length=4,
        pm_redesign_designed_res_seq=True,
        designed_residues_contig="A3-4",
    )
    assert thresholds == [2, 6, 8]
    assert fixed == [True, False, False]


def test_compute_chain_and_local_at_boundaries():
    thresholds = [8, 11]
    assert rfdiffusion.compute_chain_and_local(1, thresholds) == (0, 1)
    assert rfdiffusion.compute_chain_and_local(8, thresholds) == (0, 8)
    assert rfdiffusion.compute_chain_and_local(9, thresholds) == (1, 1)
    assert rfdiffusion.compute_chain_and_local(11, thresholds) == (1, 3)


def test_reassign_chain_and_renumber_changes_pdb_chain_and_residue_number():
    line = "ATOM      1  CA  ALA X 123      10.000  11.000  12.000  1.00 20.00           C\n"
    result = rfdiffusion.reassign_chain_and_renumber(line, chain_idx=1, local_res=7)
    assert result[21] == "B"
    assert int(result[22:26]) == 7


def test_run_rfdiffusion_builds_expected_external_arguments(tmp_path, monkeypatch):
    calls = []
    fixes = []

    monkeypatch.setattr(
        rfdiffusion,
        "run_script_in_env",
        lambda env, script, argv: calls.append((env, script, argv)),
    )
    monkeypatch.setattr(
        rfdiffusion,
        "fix_rfdiffusion_output",
        lambda *args, **kwargs: fixes.append((args, kwargs)),
    )

    args = SimpleNamespace(
        min_length=8,
        max_length=8,
        rf_contig="A1-2/LINK/A3-4",
        rf_output_prefix=str(tmp_path / "rf_output_"),
        rf_input_pdb=str(tmp_path / "input.pdb"),
        rf_num_designs=20,
        rf_inpaint_seq=None,
        rf_symmetry=None,
        rf_env="SE3nv",
        rf_script_path="/opt/RFdiffusion/run_inference.py",
        rf_fixpath=str(tmp_path / "fixed"),
        pm_redesign_designed_res_seq=False,
        designed_residues_contig="LINK",
    )

    rfdiffusion.run_rfdiffusion(args)

    assert len(calls) == 1
    env, script, argv = calls[0]
    assert env == "SE3nv"
    assert script == "/opt/RFdiffusion/run_inference.py"
    assert f"inference.output_prefix={args.rf_output_prefix}l8" in argv
    assert f"inference.input_pdb={args.rf_input_pdb}" in argv
    assert "contigmap.contigs=[A1-2/8/A3-4]" in argv
    assert "inference.num_designs=20" in argv
    assert len(fixes) == 1
