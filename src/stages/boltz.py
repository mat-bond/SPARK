#!/usr/bin/env python3
import bisect
from collections import defaultdict
import logging
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Tuple
import string 
import yaml 
import gemmi  # type: ignore
from .utils import run_command_in_env, parse_contig

#-------------------------------------- Boltz utility --------------------------------------------#
def _length_dirs(root: str,run_boltz_for_length: int | None):
    """Yield (name, abs_path) for every “length_X” directory under *root*."""
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.startswith("length_") and os.path.isdir(path):
            if run_boltz_for_length is not None:
                m = re.match(r'length_(\d+)',name)
                if m:
                    if int(m.group(1)) != run_boltz_for_length:
                        continue
                else:
                    logging.warning(f"Length folder missing length: {name}")
                    continue 
            yield name, path

def _fasta_files(folder: str):
    """Yield absolute paths to all .fa/.fasta files in *folder* (any order)."""
    for fname in os.listdir(folder):
        if fname.lower().endswith((".fa", ".fasta")):
            yield os.path.join(folder, fname)

def _yaml_files(folder: str):
    """Yield absolute paths to all .yml/.yaml files in *folder* (any order)."""
    for fname in os.listdir(folder):
        if fname.lower().endswith((".yml", ".yaml")):
            yield os.path.join(folder, fname)

def _subsequent_sequences(fasta_path: str) -> List[str]:
    """Return **every** sequence *after* the first one in *fasta_path*.

    ProteinMPNN FASTA files start with the original query sequence, followed by
    one or more designed sequences (each preceded by a header line beginning
    with ">").  For Boltz we must *ignore* the first sequence and create one
    output YAML per **subsequent** sequence.
    """
    sequences: List[str] = []
    current: List[str] = []
    header_count = 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                header_count += 1
                if header_count > 1:
                    # starting a *new* designed sequence; flush the previous one
                    if current:
                        sequences.append("".join(current))
                        current.clear()
                continue  # skip header content itself
            if header_count > 1:  # we are inside a designed sequence block
                current.append(line.strip())
    # append final sequence if file didn't end with another header
    if current:
        sequences.append("".join(current))
    return sequences

def _build_sequences_block(b_use_msa_server,chains,msa_paths,cyclic_flags):
    sequences_block =[]
    if msa_paths is None:
        msa_paths = [None] * len(chains)  # count chains

    if cyclic_flags is None:
        cyclic_flags = [False] * len(chains)

    if not(len(msa_paths) == len(chains) == len(cyclic_flags)):
        raise ValueError("Count of msa paths, chains and cyclic flags do not correspond")
    
    for idx, (seq, msa_pth, is_cyclic) in enumerate(zip(chains, msa_paths, cyclic_flags, strict=True)):
        chain_id = string.ascii_uppercase[idx]  # A, B, C, …
        entry = {
            "protein": {
                "id": chain_id,
                "sequence": seq,
                "cyclic": bool(is_cyclic),
            }
        }
        if not b_use_msa_server:
            entry["protein"]["msa"] = msa_pth if msa_pth is not None else "empty"

        sequences_block.append(entry)
    return sequences_block

def _create_fixed_residue_dict(fixed_residues: str,chain_group: Optional[list[str]] = None) -> Dict[str, List[Tuple[int, int]]]:
    fixed_residue_list = defaultdict(list)
    new_chains = fixed_residues.split()
    for new_chain in new_chains:
        for segment in new_chain.split('/'):
            if segment in ("DESIGN","LINK","HALFLINK","DES","0"):
                continue
            m = re.fullmatch(r'([A-Za-z])(\d+)-(\d+)',segment)
            if not m:
                raise ValueError("Couldn't match fixed residue contig")
            chain_id = m.group(1)
            first_res = int(m.group(2))
            last_res = int(m.group(3))
            if first_res > last_res:
                raise ValueError(f"Start > end in segment: {segment!r}")
            if chain_group is not None and not (chain_id in chain_group): continue
            fixed_residue_list[chain_id].append((first_res,last_res))

    return dict(fixed_residue_list) # Return a dict so as to not allow accidental insertions while reading

class UnionFind: # Create a disjoint‑set structure so that every chain ID ends up in an equivalence set with its partners. This allows for a flat hierarchical organisation with ~O(1) searches.
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x]) # Recursively searches for the root (chain ID whose parent is itself).
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: # Chains are in the same set (== will be merged) if and only if they share the same root.
            self.parent[rb] = ra 

def _index_above_closest_highest__value(stats, value_index, cutoff):
    """
    Return the index of the first entry whose RMSD (stats[i][1]) > cutoff.
    Equivalently, one past the last entry with RMSD <= cutoff.
    If all entries are <= cutoff, returns len(stats).
    If the very first entry is > cutoff, returns 0.
    """
    values = [row[value_index] for row in stats]
    pos = bisect.bisect_right(values, cutoff)
    return pos 

def _index_of_closest_lowest_value(stats, value_index, cutoff):
    values = [row[value_index] for row in stats]
    pos = bisect.bisect_left(values, cutoff)
    return pos 

def _build_output_structure(same_chain_on_design:str, fixed_residues: Dict[str, List[Tuple[int, int]]], input_structure: gemmi.Structure,b_percent_of_template: Optional[float], seed: float,invert:Optional[bool] = None)-> gemmi.Structure:

    # Prepare random range generator
    if b_percent_of_template is not None:
        rng = random.Random(seed)
    # Prepare output structure
    output_structure = gemmi.Structure()

    # Build union-find of chains to merge
    uf = UnionFind()
    for pair in same_chain_on_design.split('/'):
        m = re.match(r'([A-Za-z])-([A-Za-z])', pair)
        if not m:
            raise ValueError(f"Bad same-chain pair: {pair!r}")
        a, b = m.group(1), m.group(2)
        if invert:
            uf.union(b,a)
        else: 
            uf.union(a, b)
        logging.debug(f"Built union between {m.group(1)} and {m.group(2)}")

    # For each model, group chains by their root, then merge residues
    for model in input_structure:
        new_model = gemmi.Model(model.num)
        # bucket chains by root
        chains_by_root = defaultdict(list)
        for chain in model:
            root = uf.find(chain.name)
            chains_by_root[root].append(chain) # Chains with the same root (chain ID) get appended to the same list
            logging.debug(f"Added chain {chain.name} to root {root}")

        for root, chains in chains_by_root.items():
            eligible = gemmi.Chain(root)
            merged_chain = gemmi.Chain(root)
            for chain in chains:
                for res in chain:
                    num = res.seqid.num
                    # keep only the fixed residues
                    if num is not None and any(start <= num <= end for (start,end) in fixed_residues.get(chain.name, [])):
                        res.subchain = root
                        res.entity_id = root 
                        eligible.add_residue(res.clone())
            if b_percent_of_template is not None:
                all_residues = list(eligible)
                percent = float(b_percent_of_template)
                total_amount = len(all_residues)
                amount_of_residues_kept = int(round(percent*total_amount)) # Select the amount of residues we want to keep for the chain
                amount_of_residues_kept = max(0,min(amount_of_residues_kept,total_amount)) # guard so that 0 <= amount_of_residues_kept <= total_amount 
                kept_indices = set(rng.sample(range(total_amount), amount_of_residues_kept)) # Randomly sample "amount_of_residues_kept" residues in the total range, use a set for O(1) lookup
                for i, res in enumerate(all_residues):
                    if i in kept_indices:
                        r = res.clone()
                        r.subchain = root
                        r.entity_id = root
                        merged_chain.add_residue(r)
            else:
                merged_chain = eligible

            for i, res in enumerate(merged_chain, start=1):
                res.seqid.num = i        # Renumber the residues in the newly merged chain so Gemmi considers it as one polymer

            if len(merged_chain):
                new_model.add_chain(merged_chain)

        if len(new_model):
            output_structure.add_model(new_model)

    # Since we truncate a PDB, we risk creating chains/polymers that do not have correct entity id's
    # We therefore need to enforce Gemmi to overwrite any pre-existing ID's since they might no longer be coherent

    # The rest mimicks the workflow of setup_entities(), which is the Gemmi method that ensures loading usable entites 
    # however it doesn't allow for overwriting chain entity ID's
    # output_structure.assign_subchains()
    output_structure.ensure_entities()
    output_structure.deduplicate_entities()

    # Since we truncate a structure, the old residue numbering no longer corresponds to chain lengths or SEQRES/_entity_poly_seq
    output_structure.clear_sequences()  # drop any unknown/partial SEQRES/_entity_poly_seq

    # We can manually build SEQRES/_entity_poly_seq information
    # Build a mapping between subchains to the corresponding entity
    subchain_to_entity_map = {}
    for entity in output_structure.entities:
        for subchain in entity.subchains:
            logging.debug(f"Assigning subchain {str(subchain)} to entity {str(entity)}")
            subchain_to_entity_map[str(subchain)] = entity

    # Select residues and chains of interest
    subchain_to_res = defaultdict(list) 
    for model in output_structure:
        for chain in model:
            for res in chain:
                if res.entity_type == gemmi.EntityType.Polymer:
                    subchain_id = res.subchain
                    if subchain_id is None:
                        raise ValueError("Subchain id missing for residue")
                    if subchain_id in subchain_to_entity_map:
                        subchain_to_res[subchain_id].append(res.name)
                    else:
                        logging.warning("No entity for subchain %r; skipping", subchain_id)

    # Clear all entity sequences
    for entity in output_structure.entities:
        entity.full_sequence = []

    for entity in output_structure.entities:
        # By iterating entity.subchains we preserve Gemmi's ordering
        for sub in entity.subchains:
            residues = subchain_to_res.get(str(sub), [])
            entity.full_sequence += residues
            logging.debug(f"Assigned to entity {str(entity)} sequence {str(residues)}")

    output_structure.assign_label_seq_id()
    # Once we have the SEQRES/_entity_poly_seq, we can use Gemmi utility to re-number residues 
   # output_structure.setup_entities()

    return output_structure

def _make_fixed_residue_file(same_chain_on_design,rf_input_pdb,fixed_residues,output_folder_path,b_percent_of_template,seed,tag,chain_group:Optional[list[str]] = None,invert:Optional[bool] = None,file_type='cif',no_rewrite=False) -> str:
    """
    Generate a template file (mmCIF or PDB) containing only the fixed residues for Boltz.
    Returns the path to the written file.
    """
    # Normalize and guard
    file_type = file_type.strip().lower()
    if file_type not in {'cif','pdb'}: raise ValueError(f"Template file type can only be 'cif' or 'pdb', given: {file_type}")
    os.makedirs(output_folder_path, exist_ok=True)

    output_file_path = str(os.path.join(output_folder_path,f"fixed_residue_template_{tag}.{file_type}"))

    if no_rewrite: 
        # Check if the file exists already
        p = Path(output_file_path)
        if p.exists() and p.is_file():
            logging.debug(f"File already exists and no_rewrite is True, skipping {output_file_path}")
            return output_file_path
    
    # Prepare a dictionnary of fixed residues and chains
    fixed_residues = _create_fixed_residue_dict(fixed_residues,chain_group)

    # Read the input pdb into a Gemmi Structure
    input_structure = gemmi.read_structure(rf_input_pdb)
    input_structure.setup_entities()

    # Prepare output structure preserving only the fixed residues 
    output_structure = _build_output_structure(same_chain_on_design,fixed_residues,input_structure,b_percent_of_template,seed,invert)
    
    if file_type == 'cif':
        # Make a new cif file:
        output_structure.make_mmcif_document().write_file(output_file_path)
    elif file_type == 'pdb':
        output_structure.write_pdb(output_file_path)
        
    logging.info(f"Wrote {file_type} to {output_file_path}")

    return output_file_path

def _build_fixed_residues_templates(same_chain_on_design,rf_input_pdb,fixed_residues,cif_folder_path,b_force_template,b_percent_of_template,seed,one_temp_per_class):
    templates_block = []
    if one_temp_per_class:
        new_chains = [[],[]]
        chains = same_chain_on_design.split('/')
        for chain in chains:
            m = re.match(r'([A-Za-z])-([A-Za-z])',chain)
            if not m: raise ValueError(f"Could not match chains in same chain segment: {chain}")
            new_chains[0].append(m.group(1))
            new_chains[1].append(m.group(2))

        tag = 0
        for chain_group in new_chains:
            invert = (tag==1)
            entry = {
                    "cif": _make_fixed_residue_file(same_chain_on_design,rf_input_pdb,fixed_residues,cif_folder_path,b_percent_of_template,seed,str(tag),chain_group,invert,file_type='cif'),
                    "force": b_force_template  
                }
            templates_block.append(entry)
            tag+=1
    else:
        entry = {
                "cif": _make_fixed_residue_file(same_chain_on_design,rf_input_pdb,fixed_residues,cif_folder_path,b_percent_of_template,seed,"all",None,False,file_type='cif'),
                "force": b_force_template  
            }
        templates_block.append(entry)
    return templates_block

def _write_boltz_yaml(
    b_use_msa_server: bool,
    chains: List[str],
    out_path: str,
    template_cif_path_block,
    msa_paths: Optional[List[str]]= None,
    cyclic_flags: Optional[List[bool]] = None,
):
    """
    Write a Boltz YAML input file.

    Parameters
    ----------
    b_use_msa_server : bool
        If True, you plan to use Boltz's MSA server flag externally, so we omit
        per-chain `msa:` paths. If False, you can supply msa_paths (or leave them None).
    chains : iterable of str
        Each element is a plain amino-acid sequence for one chain.
    out_path : str
        Destination path for the YAML file.
    msa_paths : iterable of str|None, optional
        One entry per chain. If provided and not using the server, we'll set `msa: <path>`.
    cyclic_flags : iterable of bool, optional
        One entry per chain, default False if not given.

    Notes
    -----
    - This produces only the 'sequences' block.  'constraints', 'templates', and
      'properties' are left empty for you to fill later if needed.
    """

    # Build sequences block
    sequences_block = _build_sequences_block(b_use_msa_server,chains,msa_paths,cyclic_flags)

    # Build templates block
    templates_block = template_cif_path_block

    # Implement if needed
    constraints_block = [] 
    properties_block = []

    # Prepare document format
    yaml_doc = {
        "sequences": sequences_block,
        "constraints": constraints_block,  
        "templates": templates_block,
        "properties": properties_block,
    }

    # Write formatted, readable YAML
    with open(out_path, "w") as fh:
        yaml.safe_dump(yaml_doc, fh, sort_keys=False)

def _derive_output_name(src_fname: str, length_dir: str, seq_idx: int | None = None) -> str:
    """
    From a ProteinMPNN fasta filename like "…output_l29_3.fa" derive a Boltz
    filename:

    * For the *second* sequence in the file (our first to keep), result is
      "b_yaml_l29_3_1.yml".
    * For the third sequence → "b_yaml_l29_3_2.yml", and so on.

    If *seq_idx* is ``None`` (legacy behaviour) the function reproduces the
    previous "b_yaml_l29_3.yml" naming scheme.
    """
    m = re.search(r"l(\d+)_(\d+)", src_fname)
    if m:
        length, idx = m.groups()
        base = f"b_yaml_l{length}_{idx}"
    else:
        # fallback: sequential id based on existing files
        existing = [f for f in os.listdir(length_dir) if f.startswith("b_yaml_")]
        base = f"b_yaml_{len(existing)}"

    if seq_idx is not None:
        base = f"{base}_{seq_idx}"
    return f"{base}.yml"

def _get_new_chain_break_positions(boltz_chain_break_contig,linker_length):
    positions = []
    #parse_contig
    rf_chains = boltz_chain_break_contig.split()
    for chain in rf_chains:
        if 'BREAK' not in chain:
            positions.append(None)
            continue
        new_chains = chain.split('/BREAK/')
        if len(new_chains) > 2: 
            raise ValueError(f"Only one additional chain break supported for final chains")
        positions.append(parse_contig(new_chains[0],linker_length))

    return positions

def create_boltz_yaml(args):
    """
    Convert *every* ProteinMPNN FASTA inside each length_X folder to one (or
    many) Boltz‑formatted YAML(s):  one output file per *designed* sequence in
    the input, skipping the original query sequence.  The filenames follow the
    pattern produced by ``_derive_output_name`` ("…_Z.yml").
    """
    os.makedirs(args.b_yaml, exist_ok=True)

    template_cif_path_block = _build_fixed_residues_templates(args.same_chain_on_design,args.template_structure,args.template_residues,args.b_fixed_residue_cif_folder,args.b_force_template,args.b_percent_of_template,args.pm_seed,args.one_temp_per_class) if args.b_use_fixed_residues_template else []

    if args.run_boltz_for_length is not None:
        run_boltz_for_length = args.run_boltz_for_length 
    else:
        run_boltz_for_length = None

    for length_name, pm_dir in _length_dirs(args.pm_output,run_boltz_for_length):
        m = re.match(r'length_(\d+)',length_name)
        if not m: raise ValueError(f"Could not match length value in length folder: {length_name}")
        linker_length = int(m.group(1))
        breaks = _get_new_chain_break_positions(args.boltz_chain_break_contig,linker_length)
        logging.debug(f"Break positions for {length_name}: {breaks}")
        seqs_dir = os.path.join(pm_dir, "seqs")

        logging.debug(f"Searching for FASTA in {seqs_dir}")
        for fasta_path in _fasta_files(seqs_dir):
            logging.debug(f"Found fasta {fasta_path}")
            sequences = _subsequent_sequences(fasta_path)
            if not sequences:
                logging.warning(f"No designed sequences in {fasta_path}; skipped")
                continue
            boltz_dir = os.path.join(args.b_yaml, length_name)
            os.makedirs(boltz_dir, exist_ok=True)
            for idx, seq in enumerate(sequences, start=1):
                new_chains = []
                chains = seq.split("/")
                if len(chains) != len(breaks):
                    raise ValueError(f"#chains in FASTA ({len(chains)}) != #break specs ({len(breaks)})")
                for old_id, chain in enumerate(chains):
                    break_position = breaks[old_id]
                    if break_position is None: new_chains.append(chain)
                    else: 
                        if not (0 <= break_position <= len(chain)):
                            raise ValueError(f"Break position {break_position} outside sequence length {len(chain)}")
                        logging.debug(f"Chain {old_id}: len={len(chain)} break_at={break_position}")
                        left, right = chain[:break_position], chain[break_position:]
                        # Guard against empty strings (if a chain break is at the begining/end of a string, we don't want to include the first/last string as it will be empty)
                        if left:  new_chains.append(left)
                        if right: new_chains.append(right)
                out_fname = _derive_output_name(os.path.basename(fasta_path),
                                                boltz_dir,
                                                seq_idx=idx)
                out_path = os.path.join(boltz_dir, out_fname)
                _write_boltz_yaml(
                b_use_msa_server=args.b_use_msa_server,
                chains=new_chains,
                out_path=out_path,
                template_cif_path_block=template_cif_path_block
            )
                logging.info(f"Wrote Boltz YAML → {out_path}")

def _boltz_flags(yml: str, out_dir: str, args) -> list[str]:
    """
    Build the CLI argument list for a single Boltz prediction run.
    Flag names follow the public `boltz predict` interface
    (`--recycling_steps`, `--sampling_steps`, `--diffusion_samples`,
    `--step_scale`, `--output_format`, `--use_msa_server`). 
    """
    flags = [
        yml,                                # positional input
        f"--out_dir={out_dir}",
        f"--recycling_steps={args.b_recycling_steps}",
        f"--sampling_steps={args.b_sampling_steps}",
        f"--diffusion_samples={args.b_diffusion_samples}",
        f"--step_scale={args.b_step_scale}",
        f"--output_format={args.b_output_format}",
        f"--devices={args.b_devices}",
        f"--max_parallel_samples={args.max_parallel_samples}",
    ]
    if getattr(args, "b_use_msa_server", False):
        flags.append("--use_msa_server")
    return flags

#-------------------------------------- Run script functions --------------------------------------------# 

def run_boltz(args):
    """
    1) Convert every ProteinMPNN FASTA into Boltz format (one per design).
    2) Launch Boltz once for each length_X directory created in args.b_yaml.
       Boltz itself will iterate over all *.yml files in that directory.
    """
    logging.info("Creating Boltz YAML files …")
    create_boltz_yaml(args)

    # Walk through every ‘length_X’ directory inside args.b_yaml
    for length_name, boltz_len_dir in _length_dirs(args.b_yaml,args.run_boltz_for_length):

        # Skip if the directory is empty
        if not any(_yaml_files(boltz_len_dir)):
            logging.warning(f"No YAML files in {boltz_len_dir}; skipping")
            continue

        logging.info(f"Scheduling Boltz run for {length_name}")

        # One output directory per length
        out_dir = os.path.join(args.b_out, length_name)
        os.makedirs(out_dir, exist_ok=True)

        # Build CLI flags 
        b_args = _boltz_flags(boltz_len_dir, out_dir, args)
        logging.debug(f"Boltz CLI for {length_name}: {b_args}")

        # Execute Boltz inside its conda environment
        boltz_cmd = ["boltz", "predict"] + b_args        #  ← b_args already begins with input_path
        run_command_in_env(args.b_env, boltz_cmd)

    logging.info("All Boltz predictions completed.")