from __future__ import annotations

from types import SimpleNamespace
import pytest
from stages import proteinmpnn


def _base_args():
    return SimpleNamespace(
        pm_seq_per_target=10,
        pm_batch_size=2,
        pm_sampling_temp=0.1,
        pm_seed=123,
        pm_sol=False,
    )


def test_build_proteinmpnn_cli_contains_expected_flags():
    args = _base_args()
    cli = proteinmpnn.build_proteinMPNN_CLI(
        args,
        parsed_jsonl="/tmp/parsed.jsonl",
        fixed_jsonl="/tmp/fixed.jsonl",
        tied_jsonl="/tmp/tied.jsonl",
        output_folder="/tmp/out",
    )
    assert "--jsonl_path=/tmp/parsed.jsonl" in cli
    assert "--fixed_positions_jsonl=/tmp/fixed.jsonl" in cli
    assert "--tied_positions_jsonl=/tmp/tied.jsonl" in cli
    assert "--num_seq_per_target=10" in cli
    assert "--batch_size=2" in cli
    assert "--sampling_temp=0.1" in cli
    assert "--seed=123" in cli
    assert "--out_folder=/tmp/out" in cli
    assert "--use_soluble_model" not in cli
    assert "--unconditional_probs_only=1" not in cli


def test_build_proteinmpnn_cli_adds_optional_flags():
    args = _base_args()
    args.pm_sol = True
    cli = proteinmpnn.build_proteinMPNN_CLI(
        args,
        parsed_jsonl="/tmp/parsed.jsonl",
        fixed_jsonl="/tmp/fixed.jsonl",
        tied_jsonl="/tmp/tied.jsonl",
        output_folder="/tmp/out",
        glycosylate_best_designs=True,
    )
    assert "--use_soluble_model" in cli

def test_single_design_mode_reaches_runner_without_keyword_error(tmp_path, monkeypatch):
    rf_root = tmp_path / "rf"
    rf_len = rf_root / "length_10"
    rf_len.mkdir(parents=True)
    (rf_len / "rf_output_l10_1.pdb").write_text("MODEL\nENDMDL\n")

    output_path = tmp_path / "single"
    pm_root = tmp_path / "ProteinMPNN"
    pm_root.mkdir()

    args = SimpleNamespace(
        rf_fixpath=str(rf_root),
        pm_jsonl=str(tmp_path / "jsonl"),
        pm_fold=str(pm_root),
        pm_output=str(tmp_path / "pm_output"),
        pm_env="proteinmpnn",
        run_pmpnn_for_length=None,
        pm_seq_per_target=10,
        pm_batch_size=2,
        pm_sampling_temp=0.1,
        pm_seed=123,
        pm_sol=False,
    )

    monkeypatch.setattr(proteinmpnn, "generate_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(proteinmpnn, "validate_file_path", lambda p: p)
    monkeypatch.setattr(proteinmpnn, "run_script_in_env", lambda *a, **k: None)

    proteinmpnn.run_proteinMPNN(
        args,
        length=10,
        pdb_name="b_yaml_l10_1_1",
        output_path=str(output_path),
    )
