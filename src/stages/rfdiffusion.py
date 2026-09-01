#!/usr/bin/env python3
import logging
import json
import os
import re
from typing import List, Tuple
from utils import run_script_in_env, parse_contig

#-------------------------------------- RFdiffusion utility --------------------------------------------

def compute_chain_thresholds(contig_input: str, linker_length: int) -> List[int]:
    """
    Given a space-separated contig_input, return cumulative thresholds
    for each chain (end residue index global).
    """
    logging.info(f"Computing chain thresholds for contigs '{contig_input}'")
    cumul = 0
    thresholds: List[int] = []
    for cd in contig_input.split():
        length = parse_contig(cd, linker_length)
        cumul += length
        thresholds.append(cumul)
    logging.info(f"Computed chain thresholds: {thresholds}")
    return thresholds

def is_seg_fixed(segment,designed_residues_contig):
    chains = designed_residues_contig.split()
    for chain in chains:
        for designable_segment in chain.split('/'):
            if segment == designable_segment:
                return False
    return True 

def compute_segment_info(contig_input: str, linker_length: int,pm_redesign_designed_res_seq : bool,designed_residues_contig: str) -> Tuple[List[int], List[bool]]:
    """
    Build flat segment thresholds and is_fixed flags across all chains.
    """
    logging.info(f"Computing segment info for contigs '{contig_input}'")
    seg_thresholds: List[int] = []
    seg_is_fixed: List[bool] = []
    cumul = 0
    for cd in contig_input.split():
        for part in cd.split('/'):
            if part == '0':
                break
            if part == 'DES':
                length = 1
                is_fixed = False
            elif part == 'LINK':
                length = linker_length
                is_fixed = False
            elif part == 'HALFLINK':
                is_fixed = False
                length = linker_length // 2
            else:
                length = parse_contig(part, linker_length)
                if not pm_redesign_designed_res_seq:
                    is_fixed = True 
                else: 
                    is_fixed = is_seg_fixed(part,designed_residues_contig)
            cumul += length
            seg_thresholds.append(cumul)
            seg_is_fixed.append(is_fixed)
    return seg_thresholds, seg_is_fixed


def compute_chain_and_local(residue_index: int,
                            chain_thresholds: List[int]) -> Tuple[int, int]:
    """
    Given a global residue_index and chain_thresholds, return (chain_idx, local_res).
    """
    logging.debug(f"Mapping residue index {residue_index} to chain/local using thresholds {chain_thresholds}")
    prev = 0
    for idx, thr in enumerate(chain_thresholds):
        if residue_index <= thr:
            local = residue_index - prev
            return idx, local
        prev = thr
    idx = len(chain_thresholds) - 1
    local = residue_index - prev
    logging.warning(f"Residue index {residue_index} beyond thresholds, assigning to last chain {idx}")
    return idx, local


def reassign_chain_and_renumber(line: str,
                                chain_idx: int,
                                local_res: int) -> str:
    """
    Inject new chain ID and restart residue numbering per chain.
    """
    chain_id = chr(ord('A') + chain_idx)
    new_res_str = f"{local_res:>4}"
    line = line[:22] + new_res_str + line[26:]
    return line[:21] + chain_id + line[22:]



def fix_rfdiffusion_output(input_prefix: str,
                            output_folder: str,
                            contig_input: str,
                            linker_length: int,pm_redesign_designed_res_seq: bool, designed_residues_contig: str):
    """
    Reassigns chain IDs in RFdiffusion PDBs, restarts residue numbering per chain,
    writes out fixed_chain.json into output_folder (with both fixed_chain and chain_list).
    """
    logging.info(f"Fixing RFdiffusion outputs for prefix {input_prefix}, linker_length {linker_length}")
    # Deduce input_folder from input_prefix (everything before last '/')
    input_folder = os.path.dirname(input_prefix)
    logging.debug(f"Input folder for PDBs: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    logging.debug(f"Created/validated output folder: {output_folder}")
    
    # 1) thresholds for chain IDs
    chain_thresholds = compute_chain_thresholds(contig_input, linker_length)
    chain_ids = [chr(ord('A') + i) for i in range(len(chain_thresholds))]
    logging.debug(f"Chain IDs will be: {chain_ids}")

    # 2) thresholds and flags for LINK detection
    seg_thresholds, seg_is_fixed = compute_segment_info(contig_input, linker_length,pm_redesign_designed_res_seq,designed_residues_contig)

    # —— EARLY SANITY CHECKS ——
    # Every segment has a corresponding fixed-flag
    assert len(seg_thresholds) == len(seg_is_fixed), (
        f"[{input_prefix}] {len(seg_thresholds)} seg_thresholds != "
        f"{len(seg_is_fixed)} seg_is_fixed"
    )
    # Segment total matches chain total
    total_segs   = seg_thresholds[-1]
    total_chains = chain_thresholds[-1]
    assert total_segs == total_chains, (
        f"[{input_prefix}] total residues mismatch: "
        f"{total_segs} (segments) vs {total_chains} (chains)"
    )
    # Strictly increasing thresholds
    assert all(a < b for a, b in zip(seg_thresholds, seg_thresholds[1:])), \
        f"[{input_prefix}] seg_thresholds not strictly increasing"
    assert all(a < b for a, b in zip(chain_thresholds, chain_thresholds[1:])), \
        f"[{input_prefix}] chain_thresholds not strictly increasing"
    
    # prepare storage for per‑chain positions
    fixed_per_chain = {i: set() for i in range(len(chain_thresholds))}
    logging.debug("Initialized non-link position storage per chain")

    # process all PDBs
    for fname in os.listdir(input_folder):
        if not re.fullmatch(rf'.*l{linker_length}_\d+\.pdb$', fname.lower()):
            continue
        logging.info(f"Processing PDB file: {fname}")
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        with open(in_path) as infile, open(out_path, 'w') as outfile:
            prev_resnum = None
            residue_index = 0
            curr_seg = 0
            for line in infile:
                if line.startswith(('ATOM  ', 'HETATM')):
                    resnum = int(line[22:26])
                    if resnum != prev_resnum:
                        residue_index += 1
                        prev_resnum = resnum

                        # advance segment pointer (handles skipping multiple thresholds)
                        while curr_seg < len(seg_thresholds) and residue_index > seg_thresholds[curr_seg]:
                            curr_seg += 1

                        chain_idx, local_res = compute_chain_and_local(residue_index, chain_thresholds)

                        if curr_seg < len(seg_is_fixed) and seg_is_fixed[curr_seg]:
                                fixed_per_chain[chain_idx].add(local_res)

                    line = reassign_chain_and_renumber(line, chain_idx, local_res)
                outfile.write(line)
        logging.debug(f"Wrote fixed PDB to {out_path}")

    # build the comma‑separated string per chain
    chain_strs = []
    for i in range(len(chain_thresholds)):
        sorted_locals = sorted(fixed_per_chain[i])
        chain_strs.append(' '.join(str(x) for x in sorted_locals))
    fixed_chain_str = ','.join(chain_strs)
    chain_list_str = ' '.join(chain_ids)
    logging.debug(f"Fixed chain string: {fixed_chain_str}")
    logging.debug(f"Chain list string: {chain_list_str}")

     #write to JSON file with both entries
    json_path = os.path.join(output_folder, 'fixed_chain.json')
    with open(json_path, 'w') as jfh:
        json.dump({
            'fixed_chain': fixed_chain_str,
            'chain_list' : chain_list_str
        }, jfh, indent=2)
    logging.info(f"Wrote fixed_chain.json to {json_path}")

#-------------------------------------- Run script functions --------------------------------------------# 
def run_rfdiffusion(args):
    logging.info("Starting RFdiffusion run")
    # We want to run RFdiffusion for every linker size provided
    for length in range(args.min_length, args.max_length + 1):
        logging.info(f"Running RFdiffusion for linker length {length}")

        # Adapt contigs string
        contigInput = args.rf_contig.replace("HALFLINK", str(length//2))
        contigInput = contigInput.replace("LINK", str(length))
        contigInput = contigInput.replace("DES", "1")
        logging.debug(f"Contig input for length {length}: {contigInput}")

        # Build RFdiffusion argument list
        rf_args = [
            f"inference.output_prefix={args.rf_output_prefix}" + "l" + str(length),
            f"inference.input_pdb={args.rf_input_pdb}",
            f"contigmap.contigs=[{contigInput}]",
            f"inference.num_designs={args.rf_num_designs}"
        ]
        if args.rf_inpaint_seq is not None:
            rf_args.append(f"contigmap.inpaint_seq=[{args.rf_inpaint_seq}]")
        if args.rf_symmetry:
            rf_args.append(f"inference.symmetry={args.rf_symmetry}")
        logging.debug(f"RFdiffusion args: {rf_args}")

        run_script_in_env(args.rf_env, args.rf_script_path, rf_args)

        # Create subfolders in the fixed pdb file path, one folder per linker length
        length_subfolder = os.path.join(args.rf_fixpath, f"length_{length}")
        os.makedirs(length_subfolder, exist_ok=True)
        logging.debug(f"Created folder for fixed PDBs: {length_subfolder}")

        # Create files with an adapted format to account for RFdiffusion generating the entire structure as one single chain
        # Also generates a JSON file with the fixed chains, one for each length
        fix_rfdiffusion_output(args.rf_output_prefix, length_subfolder, args.rf_contig, length,args.pm_redesign_designed_res_seq,args.designed_residues_contig)
    logging.info("Completed RFdiffusion run")
