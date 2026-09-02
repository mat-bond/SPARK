# Protein Design Pipeline (RFdiffusion → ProteinMPNN → Boltz → AF + Analysis)

End‑to‑end, scriptable workflow for protein linker/backbone design and evaluation.
It runs **RFdiffusion** for backbone sampling, **ProteinMPNN** for sequence design,
**Boltz** for sequence/structure scoring, optional **AlphaFold (via colabdesign)**
for structure prediction, and then aggregates metrics/plots (RMSD, pLDDT, PAE/PDE, cavity stats).

> Quick mental model: you define contigs and a linker length range; the pipeline
> generates candidates per length (`length_<N>/...`), designs sequences, evaluates,
> predicts structures, and emits summary tables/plots.

---

## Repository structure

```text
SPARK/
├── pipeline.py              # backwards-compatible 
├── src/
│   ├── main.py              # pipeline orchestration and CLI
│   ├── utils.py             # shared parsing, validation, environment and SLURM helpers
│   ├── constants.py         # shared constants and analysis indices
│   ├── plots_and_graphs.py  # plotting helpers
│   ├── alphafold_script.py  # backwards-compatible worker wrapper
│   ├── cavity_analysis.py   # backwards-compatible metric wrapper
│   ├── stages/
│   │   ├── rfdiffusion.py
│   │   ├── proteinmpnn.py
│   │   ├── boltz.py
│   │   ├── alphafold.py
│   │   └── analysis.py
│   ├── metrics/
│   │   └── cavity.py
│   └── workers/
│       └── alphafold_script.py
└── tests/                   # lightweight regression tests


```
---

## Requirements

- **Python ≥ 3.10**, **CUDA GPU** recommended.
- A base environment for orchestrating these scripts:
  ```bash
  pip install numpy pandas pyyaml matplotlib gemmi pyKVFinder
  # For metrics: PyMOL must be available (e.g., conda install -c schrodinger pymol-bundle)
  ```
- External tools (typically separate conda envs):
  - RFdiffusion
  - ProteinMPNN
  - Boltz
  - colabdesign / AlphaFold (for `mk_af_model`)
- Optional: **SLURM** for job arrays (the master `pipeline.py` generates sbatch scripts and sets dependencies). 

Each stage runner accepts environment names/paths and calls the tool via `conda run` (see `run_script_in_env` in `utils.py`). 

---

## Typical directory & file conventions

- Stage folders per linker length: `length_<N>/`
- RFdiffusion PDBs: `rf_output_l<N>_<ID>.pdb` (renumbered/chain‑fixed downstream). 
- Input YAML for Boltz:  
  ```yaml
  sequences:
    - protein:
        id: A             # one or many ids (list allowed)
        sequence: SEQ...
  ```
  The runner keeps multiple ids mapping to the same sequence and preserves order.

---

## Contig grammar & related flags

Several flags accept a **contig string**—a compact way to specify residue ranges, linkers,
and (where applicable) chain boundaries. The pipeline uses these strings to build
**lookup tables (LUTs)** that map original residue numbering → stage-specific numbering
and to decide which residues are **fixed** vs **designed** in downstream tools.

### 1) Basic residue ranges
- **Token form**: `<Chain><Start>-<End>` (inclusive, 1-based indices).  
  Examples: `A1-168`, `B20-60`, `a30-48`, `b10-20`
- **Multiple ranges** can be separated by **/** ("slashes", meaning “concatenate these blocks”).
- Chains are single letters. Uppercase/lowercase letters are allowed and are treated as chain IDs. (Upper and lower case are interpreted as different chains).
- Final "/0" followed by a space indicates a chain break on the designed structure. 
(e.g. `A1-10/B3-4/0`&nbsp;`C1-10` will create a design with 2 chains, the first concatenating the residues `A1-10` and `B3-4` of the initial structure, the second with the residues `C1-10`.
- Chains are renamed in alphabetic order and residues renumbered.

### 2) Special placeholders
- **`LINK`**: a variable-length linker region that the design tools will generate.  
  You place it **between** structured segments to indicate a gap that should be filled by design.
- **`HALFLINK`, `DES`, `BREAK`**: additional markers used by some stages to mark linkers,
  designed segments, or explicit chain breaks. When present, they are parsed just like `LINK`,
  i.e., as *non-template, designed* regions or separators.
- **`HALFLINK`** acts exactly like `LINK`, but with half the length. 
- **`DES`** marks places to insert point mutations and one-residue linkers. 
- **`BREAK`** adds an additional chain break in the sequence output by ProteinMPNN (e.g. to simulate downstream protein processing such as cleavage sites recognized after folding.)
- In RFdiffusion contigs, tokens are often **slash-separated** to form a segment with subfields.
  For example:  
  `a30-48/LINK/C2-5/0` &nbsp;or&nbsp; `b10-20/LINK/0`  
  Here the first field is the residue range, then a placeholder like `LINK`, then optional
  method-specific fields.

### 3) Flag-by-flag guide

#### `--rf_contig "<segments...>"`  (RFdiffusion)
- Purpose: defines the **input arrangement** used by RFdiffusion for backbone generation.  
- Format: new chain indicators on the to-be-designed structure are separated by **spaces**; each chain may itself contain **slash**-separated
  subfields. The can indicate the residues of the starting structure to concatenate with the rest of the subfields, or be `LINK`, `HALFLINK`, `DES`, `BREAK` indicators.
- Example:
  ```text
  a30-48/LINK/C2-5/0 b10-20/LINK/0
  ```
  Meaning (conceptually): take `a30-48` and `C2-5` from the starting, insert a linker between them to make them into one chain. Then add a new chain starting from residues `b10-20` followed by de novo residues. The pipeline will build LUTs and write chain‑fixed PDBs
  under `--rf_modified_files_path` for downstream stages.

#### `--boltz_chain_break_contig "<Astart-Aend/Bstart-Bend>"`  (Boltz)
- Purpose: defines **where chains are broken** for Boltz evaluation (used to split/join components
  consistently with design intent).
- Format: use a **slash** (`/`) to separate per‑chain ranges; use `Start-End` inside each.
- Example: `A1-10/B1-80`.

#### `--designed_residues_contig "<ranges>"` (Analysis & selection)
- Purpose: marks which residues are considered **designed** (vs fixed) for averaging PDE/PAE/pLDDT,
  plotting PAE heatmaps, and selection metrics.
- Format: same `<Chain><Start>-<End>` tokens; multiple ranges can be separated by spaces or slashes.
- Example: `A10-50/LINK/B20-60`.
- Note: **designed** residues are all new residues: on a de novo structure or sequence mutations.

#### `--alignment_contig "<ranges>"` and `--rmsd_alignment_residues "<ranges>"` (Analysis)
- Purpose: specifies residues used for **superposition/alignment** in RMSD calculations.
- Format: one or more ranges; spaces are allowed.  
- Example: `A1-18/LINK/B1-16`.

#### `--metric_residues "<ranges>"`
- Purpose: restricts downstream metrics (e.g., cavity or interface statistics) to specific spans.
- Format: same as above.  
- Example: `A10-60 B20-70` (commas or spaces both work; they are normalized internally).
- Example: `LINK`

#### `--same_chain_on_design "A-B"`
- Indicates which chains from the input structure get concatenated into one chain in the designed structure.
- Purpose: treat two chains as a **single designed unit** at the sequence‑design stage (useful for
  enforcing same‑chain behavior across tools). The value lists chain IDs separated by `-`.
- Example: `A-B` merges A and B for design coupling purposes.

### 4) How numbering & LUTs work
- Each stage may **renumber** residues (e.g., after inserting `LINK` regions or trimming templates).
- The pipeline constructs per‑length **lookup tables (LUTs)** that map **original PDB numbering**
  → **stage‑specific numbering** for each chain. These LUTs ensure that downstream scripts (ProteinMPNN,
  Boltz, AF, and analysis) all refer to the **same physical residues** even after edits.
- Practical effect: you can (and should) keep using the same contig strings across stages; the pipeline translates
  them as needed so your ranges always hit the intended residues.

### 5) Quick sanity checks
- Make sure chain letters exist in your input structures (`A`, `B`, …).
- Ranges are **inclusive**. `A1-10` contains 10 residues.
- Avoid overlapping segments unless the stage explicitly expects them.
- When in doubt, test a tiny run and inspect the stage’s “fixed/renumbered” PDBs written under
  your chosen output folders.


### Ordering requirement across contigs (important)

Several flags accept contigs that are later **cross‑referenced** using the LUT built from your primary contig (RF stage).
The helper `utils._get_residue_strings(...)` walks the segments of your secondary contigs
(e.g., `--designed_residues_contig`, `--alignment_contig`) and **matches each segment** against the LUT entries
produced by `utils.build_lut_from_contig(...)`. During this walk it increments counters for `LINK`, `HALFLINK`, and `DES`
(`LINK1`, `LINK2`, …; `DES1`, `DES2`, …) based strictly on **encounter order**.

**Therefore, all contigs must keep the same order and structure:**
- Same **chain order** (left→right in the string corresponds to output chains A→B→…; `BREAK` must appear in the same places).
- Same sequence of **segment kinds** per chain (fixed ranges vs `LINK`/`HALFLINK`/`DES`), so that `DES1` in one contig refers to the
  same physical region as `DES1` in the LUT; ditto for `LINK1`, `LINK2`, etc.
- Fixed segments must reference the **same original chains and ranges** as those used to build the LUT (e.g., `A10-60` then `B5-40`),
  otherwise mapping falls back to edge matching and will raise `ValueError("Problem with contig LUT or design contig")`
  if it can’t align the boundaries.
- Ranges must be **ascending** (`start ≤ end`), otherwise you’ll see errors like `Start > end in segment` or `Incorrect contig format`.
- You cannot exceed **26 output chains** (A–Z); extra `BREAK`s will error out (`More than 26 chains not supported`).

**Concrete example**

```text
# Primary RF contig (drives the LUT; linker_length = 8)
--rf_contig "A1-50/LINK B10-60/HALFLINK/DES"

# ✅ Matching secondary contigs (same order, same breaks)
--alignment_contig         "A1-50/LINK B10-60/HALFLINK/DES"
--designed_residues_contig "LINK HALFLINK DES"   # uses the same implicit numbering: LINK1, HALFLINK1, DES1

# ❌ Mismatched order (DES moved before HALFLINK) → counters don’t line up → mapping fails
--designed_residues_contig "LINK DES HALFLINK"

# ❌ Different chain order or missing BREAK → output chain letters shift → mapping fails
--alignment_contig "B10-60/HALFLINK/DES A1-50/LINK"
```

**Symptoms when order doesn’t match**
- Errors like `Problem with design contig`, `Problem with contig LUT or design contig`, or `Incorrect contig format`.
- Negative/zero computed ranges when matching edges.
- PyMOL selections that refer to **wrong residues** (if the mismatch slips through earlier checks).

**Practical advice**
- Start from your `--rf_contig` and derive every other contig by **removing segments**, not reordering them.
- If you need to split across chains, use `BREAK` in all contigs at the **same place**.
- Keep a small test case and run the analysis step on it to confirm selections before large jobs.

---

## SLURM / SBATCH remarks

If using the pipeline in array mode (batch jobs), modify the SLURM submission scripts in `pipeline.py` to match your local slurm configurations. There is one SBATCH script per step (RFdiffusion, PMPNN, ...).

---

## Quick start (per stage)

Below is a single end‑to‑end invocation of **`pipeline.py`** that matches the actual CLI flags.
You can still select/skip stages via switches shown after this block.


```bash
python pipeline.py \
  -l pipeline.log \
  --rf_script_path /path/to/RFdiffusion/inference.py \
  --rf_output_prefix ./rf_outputs/rf_output_ \
  --rf_input_pdb ./inputs/template.pdb \
  --rf_contig "a30-48/LINK/C2-5/0 b10-20/LINK/0" \
  --rf_min_linker_length 6 \
  --rf_max_linker_length 12 \
  --rf_num_designs_per_linker 25 \
  --rf_modified_files_path ./rf_outputs_fixed \
  --pm_jsonl_path ./pm_jsonl \
  --pm_proteinmpnn_path /path/to/ProteinMPNN \
  --pm_output_path ./pmpnn_outputs \
  --pm_seq_per_target 50 \
  --pm_sampling_temp 0.1 \
  --pm_seed 0 \
  --pm_batch_size 8 \
  --pm_env proteinmpnn \
  --b_out_dir ./boltz_outputs \
  --b_yaml_dir ./boltz_yaml \
  --b_designs_from_pm 5 \
  --b_recycling_steps 3 \
  --b_sampling_steps 50 \
  --b_diffusion_samples 4 \
  --b_step_scale 1.5 \
  --b_output_format pdb \
  --b_env boltz \
  --b_devices 1 \
  --max_parallel_samples 1 \
  --same_chain_on_design "A-B" \
  --fixed_residues "A1-90 B1-70" \
  --boltz_chain_break_contig "A1-10/B1-80" \
  --template_structure ./inputs/template.pdb \
  --af_output_path ./af_outputs \
  --af_env af_env \
  --af_recycles_amount 3 \
  --af_initial_guess \
  --af_use_initial_atom_pos \
  --af_template_contig "A1-18 B1-18" \
  --af_template_designed "A10-50 B20-60" \
  --af_template_path ./boltz_outputs/length_10/template_for_af.pdb \
  --filtered_designs_folder ./filtered_designs \
  --rmsd_alignment_residues "A1-18 B1-16" \
  --py_env pymol \
  --pymol_alignment_cycles 5 \
  --final_selection_amount 10 \
  --designed_residues_contig "A10-50/B20-60" \
  --align_structure ./inputs/template.pdb \
  --alignment_contig "A1-18 B1-16" \
  --rmsd_cutoff 3.0 \
  --plddt_cutoff 0.7 \
  --pae_cutoff 10 \
  --pde_cutoff 5 \
  --pykvfinder_env pykv \
  --af_params_dir /abs/path/to/af/params \
  --metric_residues "A10-60 B20-70"
````

### Stage-selection switches
- Run only RFdiffusion: `--only_rfdiff`
- Skip a stage: `--skip_rfdiff`, `--skip_pmpnn`, `--skip_boltz`, `--skip_af`
- Run only a later stage: `--only_pmpnn`, `--only_boltz`, `--only_af`
- SLURM arrays: `--run_array` (array range derived from RF min/max linker length)

### RFdiffusion flag name notes
Use `--rf_min_linker_length` / `--rf_max_linker_length` and `--rf_num_designs_per_linker`.
Also note `--rf_modified_files_path` is required.

### Required ProteinMPNN flags
`--pm_jsonl_path`, `--pm_proteinmpnn_path`, `--pm_output_path`,
`--pm_seq_per_target`, `--pm_sampling_temp`, `--pm_seed`,
`--pm_batch_size`, `--pm_env`.

### Key Boltz flags
`--b_out_dir`, `--b_yaml_dir`, `--b_designs_from_pm`, `--b_recycling_steps`,
`--b_sampling_steps`, `--b_diffusion_samples`, `--b_step_scale`, `--b_output_format`,
`--b_env`, `--b_devices`, `--max_parallel_samples`, `--same_chain_on_design`,
`--fixed_residues`, `--boltz_chain_break_contig`, `--template_structure`.

### AlphaFold note
The pipeline’s AF stage uses template-driven flags
(`--af_template_contig`, `--af_template_designed`, `--af_template_path`, …).

## Master launcher
 (SLURM arrays)

`pipeline.py` can generate sbatch scripts for each stage and submit them with dependencies (`afterok`). Flags like `--only_rfdiff`, `--only_pmpnn`, `--only_boltz`, `--only_af` let you run subsets; `--run_array/--no_run_array` controls array behavior. Logs are written and all scripts for downstream stages reuse the same flag block automatically. 

> Tip: set `--logfile pipeline.log` and ensure your output root has a `logs/` subfolder.

---

## Notes & conventions

- **File naming** is important for lookups (e.g., deducing linker length from `b_yaml_l{L}_{ID}` and finding the matching RFdiffusion model). 
- **Plots**: histograms/scatter include mean±SD; PAE heatmaps overlay designed indices as minor red ticks. 

---

## Troubleshooting

- **PyMOL not found**: install in a separate conda env and point `--py_env`. 
- **Bad contig segment** errors: check the contig string syntax (see `utils.parse_contig`). 
- **Multiple chains** in AF templates: `alphafold_pipeline.py` builds template structures chain‑wise and can append poly‑Ala stubs when needed. (Colabdesign AF requires the template sequence length to match exactly the input sequence length) 

---
