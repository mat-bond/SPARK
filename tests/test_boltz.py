from __future__ import annotations

from types import SimpleNamespace
import pytest
import yaml
from stages import boltz


def test_subsequent_sequences_skips_original_query(tmp_path):
    fasta = tmp_path / "design.fa"
    fasta.write_text(
        ">original\nAAAA\n"
        ">design_1\nBBBB\n"
        ">design_2\nCC\nCC\n"
    )
    assert boltz._subsequent_sequences(str(fasta)) == ["BBBB", "CCCC"]


def test_derive_output_name_preserves_ids(tmp_path):
    assert boltz._derive_output_name(
        "rf_output_l29_3.fa", str(tmp_path), seq_idx=2
    ) == "b_yaml_l29_3_2.yml"


def test_chain_break_position_is_computed_from_pre_break_contig():
    breaks = boltz._get_new_chain_break_positions(
        "A1-2/LINK/B1-2/BREAK/C1-2 D1-3",
        linker_length=4,
    )
    assert breaks == [8, None]


def test_build_sequences_block_assigns_chain_ids_and_empty_msa():
    block = boltz._build_sequences_block(
        b_use_msa_server=False,
        chains=["AAAA", "BBBB"],
        msa_paths=None,
        cyclic_flags=None,
    )
    assert block == [
        {"protein": {"id": "A", "sequence": "AAAA", "cyclic": False, "msa": "empty"}},
        {"protein": {"id": "B", "sequence": "BBBB", "cyclic": False, "msa": "empty"}},
    ]


def test_boltz_flags_match_config():
    args = SimpleNamespace(
        b_recycling_steps=3,
        b_sampling_steps=50,
        b_diffusion_samples=2,
        b_step_scale=1.5,
        b_output_format="pdb",
        b_devices=2,
        max_parallel_samples=4,
        b_use_msa_server=True,
    )
    flags = boltz._boltz_flags("/tmp/yaml_dir", "/tmp/out", args)
    assert flags[0] == "/tmp/yaml_dir"
    assert "--out_dir=/tmp/out" in flags
    assert "--recycling_steps=3" in flags
    assert "--sampling_steps=50" in flags
    assert "--diffusion_samples=2" in flags
    assert "--step_scale=1.5" in flags
    assert "--output_format=pdb" in flags
    assert "--devices=2" in flags
    assert "--max_parallel_samples=4" in flags
    assert "--use_msa_server" in flags


def test_create_boltz_yaml_converts_each_designed_sequence(tmp_path):
    pm_output = tmp_path / "proteinmpnn"
    seq_dir = pm_output / "length_4" / "seqs"
    seq_dir.mkdir(parents=True)
    (seq_dir / "rf_output_l4_1.fa").write_text(
        ">original\nAAAAAA\n"
        ">design_1\nBBBBBB\n"
        ">design_2\nCCCCCC\n"
    )

    yaml_root = tmp_path / "boltz_yaml"
    args = SimpleNamespace(
        b_yaml=str(yaml_root),
        pm_output=str(pm_output),
        run_boltz_for_length=None,
        boltz_chain_break_contig="A1-2/LINK",
        b_use_fixed_residues_template=False,
        b_use_msa_server=False,
        b_designs_from_pm=2,
    )

    boltz.create_boltz_yaml(args)

    generated = sorted((yaml_root / "length_4").glob("*.yml"))
    assert [p.name for p in generated] == [
        "b_yaml_l4_1_1.yml",
        "b_yaml_l4_1_2.yml",
    ]

    first = yaml.safe_load(generated[0].read_text())
    second = yaml.safe_load(generated[1].read_text())
    assert first["sequences"][0]["protein"]["sequence"] == "BBBBBB"
    assert second["sequences"][0]["protein"]["sequence"] == "CCCCCC"


@pytest.mark.xfail(
    reason="Known bug: create_boltz_yaml currently ignores args.b_designs_from_pm.",
    strict=False,
)
def test_create_boltz_yaml_respects_b_designs_from_pm_limit(tmp_path):
    pm_output = tmp_path / "proteinmpnn"
    seq_dir = pm_output / "length_4" / "seqs"
    seq_dir.mkdir(parents=True)
    (seq_dir / "rf_output_l4_1.fa").write_text(
        ">original\nAAAAAA\n"
        ">design_1\nBBBBBB\n"
        ">design_2\nCCCCCC\n"
    )

    yaml_root = tmp_path / "boltz_yaml"
    args = SimpleNamespace(
        b_yaml=str(yaml_root),
        pm_output=str(pm_output),
        run_boltz_for_length=None,
        boltz_chain_break_contig="A1-2/LINK",
        b_use_fixed_residues_template=False,
        b_use_msa_server=False,
        b_designs_from_pm=1,
    )

    boltz.create_boltz_yaml(args)

    generated = sorted((yaml_root / "length_4").glob("*.yml"))
    assert len(generated) == 1
