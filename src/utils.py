#!/usr/bin/env python3
import bisect
from collections import defaultdict
from enum import IntEnum
import math
import os
import pathlib
import re
import string
import subprocess
import logging
from pathlib import Path
import shlex
import json
import textwrap
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np

#-------------------------------------- General utility --------------------------------------------#
def geometric_mean(container):
    return np.exp(np.mean(np.log(container)))

def _get_file_from_same_dir(path,startswith):
    pdb_file = pathlib.Path(path).resolve()
    pred_dir = pdb_file.parent
    pae_file = ''

    for entry in os.listdir(pred_dir):
        if entry.startswith(startswith): pae_file = os.path.join(pred_dir,entry)

    if pae_file: 
        return pae_file
    else:
        logging.warning(f"Could not find pae file in directory {pred_dir}")
        return

def _truthy(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "true":  return True
            if s == "false": return False
        return v

def _to_int_na(x):
    if pd.isna(x) or x == "":
        return pd.NA
    try:
        return int(x)
    except (TypeError, ValueError):
        return pd.NA
    
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

def chain_offsets_from_lut(lut: List[Tuple[str, str]]) -> Dict[int, int]:
    """
    Build a mapping  {chain_numeric_id → global_start_index}
    where the numeric id is 0 for 'A', 1 for 'B', …

    Parameters
    ----------
    lut : list of (old_seg, new_range) tuples produced by build_lut_from_contig()
          Each *new_range* looks like "A1-168" or "B169-335".

    Notes
    -----
    We *sum* the true segment lengths so that gaps do **not** inflate the offset.
    
    """
    # Accumulate exact residue counts per chain
    lengths = defaultdict(int)                           # {letter → total_len}
    for _old_seg, new_range in lut:
        m = re.match(r"([A-Z])(\d+)-(\d+)", new_range)
        if not m:
            raise ValueError(f"Bad range '{new_range}' in LUT")
        chain_letter, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        seg_len = end - start + 1                        # inclusive range
        lengths[chain_letter] += seg_len

    # Convert to cumulative offsets in alphabetical order
    offsets: Dict[int, int] = {}
    running_total = 0
    for ch in sorted(lengths):                           # A, B, C, …
        numeric_id = ord(ch) - ord('A')
        offsets[numeric_id] = running_total
        running_total += lengths[ch]

    return offsets

def global_index(offsets,chain_id_letter: str, resseq: int) -> int:
    """
    chain_id_letter : 'A', 'B', …  (matching your FASTA order)
    resseq          : PDB residue number *within that chain*
    returns         : 0‑based global index into matrix/arrays
    """
    # convert 'A'→0, 'B'→1, …
    chain_num = ord(chain_id_letter.upper()) - ord('A')
    
    # local numbering in Boltz starts at 1 (PDB style).
    local_0_based = resseq - 1
    
    return offsets[chain_num] + local_0_based 

def _get_designed_residue_indices_n_offsets(residue_contig,contig,linker_length,global_res=True):
        # Get designed residue regions (chain, start, end) in new numbering
        chains_full = contig.split()
        lut = build_lut_from_contig(linker_length,chains_full)
        design_list,not_of_use = _get_residue_strings(False,residue_contig,contig,linker_length,lut,basic_format_residues,cmd=False,use_link_and_des=True)

        offsets = chain_offsets_from_lut(lut)
        designed_indices = set()

        for ch, start, end in design_list:
            for resnum in range(start,end+1):
                if global_res:
                    global_resnum = global_index(offsets, ch, resnum)
                    designed_indices.add(global_resnum)
                else:
                    designed_indices.add(resnum)
        return designed_indices, offsets

def _get_rf_diffusion_model(rf_folder,pdb_name,linker_length=None):
    model_path = ""
    pdb_m = re.match(r'^b_yaml_l(\d+)_(\d+)',pdb_name) # TODO: fix so as to not make the prefix hard-coded
    if not pdb_m: raise ValueError(f"Could not match length and ID in PDB name: {pdb_name}, linker_length = {linker_length}")
    deduced_length = int(pdb_m.group(1))
    if linker_length is not None:
        if (deduced_length != linker_length): raise ValueError(f"Deduce linker length {deduced_length} doesn't correspond to passed linker length = {linker_length} in PDB {pdb_name}")
    else: 
        linker_length = deduced_length
    model_id = int(pdb_m.group(2))

    for entry in os.listdir(rf_folder):
        m = re.match(r'^length_(\d+)$',entry)
        if not m or int(m.group(1)) != linker_length: continue 
        length_folder_path = os.path.join(rf_folder,entry)

        for file in os.listdir(length_folder_path):
            file_m = re.match(r'^rf_output_l(\d+)_(\d+)',file) 
            if not file_m or ((int(file_m.group(1)),int(file_m.group(2))) != (linker_length,model_id)) or not file.endswith('.pdb'): continue
            model_path = os.path.join(length_folder_path,file)
            break

        if model_path: break

    if not model_path:
        logging.warning(f"Returning empty model path for pdb {pdb_name}")

    return model_path

def parse_contig(contig_str: str, linker_length: int) -> int:
    """
    Parse one contig string like "a260-448/LINK/C2-35/.../0"
    into the total number of residues in that chain.
    """
    parts = contig_str.split('/')
    total = 0
    for p in parts:
        p.strip()
        if not p: continue 
        if p.upper() == 'LINK':
            total += linker_length
        elif p.upper() == 'HALFLINK':
            total += linker_length // 2
        elif p.upper() == 'DES':
            total += 1
        elif p.upper() == 'BREAK':
            continue
        elif p == '0':
            break
        else:
            m = re.match(r'[A-Za-z](\d+)-(\d+)', p)
            if m:
                start, end = map(int, m.groups())
                seg_len = (end - start + 1)
                total += seg_len
            else:
                raise ValueError(f"Bad contig segment: {p}")
    return total

def build_lut_from_contig(linker_length, chains_full):
    # Construct a look up table for old and new chain+residue ID's 
    lut = []
    link_count = 0
    halflink_count = 0
    des_count = 0
    offset = 0
    for i, chain in enumerate(chains_full):
        break_count = 0
        if i+offset >= 26:
            raise ValueError("More than 26 chains not supported")
        chain_id = string.ascii_uppercase[i+offset]
        cumul = 0
        for seg in chain.split('/'):
            if seg == 'DESIGN':
                raise ValueError(f"Fixed residue contig not accepted in LUT building: {chains_full}")
            elif seg == 'BREAK':
                offset += 1
                if i + offset >= 26:
                    raise ValueError("More than 26 chains not supported")
                break_count += 1
                if break_count > 1: 
                    raise ValueError(
                        f"Only one BREAK supported per input chain: {chain}"
                    )
                chain_id = string.ascii_uppercase[i+offset]
                cumul = 0
                continue
            elif seg.upper() == 'LINK':
                link_count += 1
                first_res = cumul+1
                cumul += linker_length
                last_res = cumul
                count_seg = seg + str(link_count)
                lut.append((count_seg, f"{chain_id}{first_res}-{last_res}"))
            elif seg.upper() == 'HALFLINK':
                halflink_count += 1
                first_res = cumul+1
                cumul += linker_length // 2
                last_res = cumul
                count_seg = seg + str(halflink_count)
                lut.append((count_seg, f"{chain_id}{first_res}-{last_res}"))
            elif seg.upper() == 'DES':
                des_count += 1
                first_res = cumul + 1
                last_res = first_res
                count_seg = seg + str(des_count)
                lut.append((count_seg, f"{chain_id}{first_res}-{last_res}"))
                cumul += 1
            elif seg == '0':
                break
            else:
                m = re.match(r'[A-Za-z](\d+)-(\d+)', seg)
                if not m:
                    raise ValueError("Incorrect contig format")
                start, end = map(int, m.groups())
                seg_len = (end - start + 1)
                lut.append((seg, f"{chain_id}{cumul+1}-{cumul+seg_len}"))
                cumul += seg_len
    return lut

def basic_format_residues(new_id,first_res,last_res):
    return (new_id,first_res,last_res)

def _get_residue_strings(align_only_one_chain,residue_contig,contig,linker_length,lut,function,cmd=True,use_link_and_des=False):
    # Calculate RMSD with the selected residues between original input file and Boltz prediction
    chains_fixed = residue_contig.split()
    parts_design = []
    parts_input = []
    des_count = 0
    link_count = 0
    halflink_count = 0
    for chain in chains_fixed:
        for seg in chain.split('/'):
            if seg.upper() == "BREAK":
                continue
            if seg.upper() in {"DES","DESIGN"}: 
                if use_link_and_des:
                    des_count += 1
                    count_seg = seg + str(des_count)
                    for old,new in lut:
                        if count_seg == old:
                            m = re.match(r'([A-Za-z])(\d+)-(\d+)',new)
                            if not m: raise ValueError("Problem with design contig")
                            el_chain_new = m.group(1)
                            el_start_new = int(m.group(2))
                            el_end_new = int(m.group(3))
                            parts_design.append(function(el_chain_new,el_start_new,el_end_new))
                continue
            if seg.upper() == "LINK": 
                if use_link_and_des:
                    link_count += 1
                    count_seg = seg + str(link_count)
                    for old,new in lut:
                        if count_seg == old:
                            m = re.match(r'([A-Za-z])(\d+)-(\d+)',new)
                            if not m: raise ValueError("Problem with design contig")
                            el_chain_new = m.group(1)
                            el_start_new = int(m.group(2))
                            el_end_new = int(m.group(3))
                            parts_design.append(function(el_chain_new,el_start_new,el_end_new))
                continue
            if seg.upper() == "HALFLINK": 
                if use_link_and_des:
                    halflink_count += 1
                    count_seg = seg + str(halflink_count)
                    for old,new in lut:
                        if count_seg == old:
                            m = re.match(r'([A-Za-z])(\d+)-(\d+)',new)
                            if not m: raise ValueError("Problem with design contig")
                            el_chain_new = m.group(1)
                            el_start_new = int(m.group(2))
                            el_end_new = int(m.group(3))
                            parts_design.append(function(el_chain_new,el_start_new,el_end_new))
                continue
            if seg == "0":
                break
            match_seg = re.match(r"([A-Za-z])(\d+)-(\d+)",seg)
            if not match_seg:
                 raise ValueError(f"Problem with design contig, segment : {seg}")
            seg_chain = match_seg.group(1)
            seg_start = int(match_seg.group(2))
            seg_end = int(match_seg.group(3))
            if not (seg_chain and seg_start and seg_end):
                raise ValueError("Problem with design contig")
            parts_input.append(function(seg_chain,seg_start,seg_end))
            matched = False
            for element in lut:
                if element[0].startswith("DES") or element[0].startswith("LINK") or element[0].startswith("HALFLINK"):
                    continue
                match_element = re.match(r"([A-Za-z])(\d+)-(\d+)",element[1])
                match_element_old = re.match(r"([A-Za-z])(\d+)-(\d+)",element[0])
                if match_element and match_seg and match_element_old: 
                    el_chain_new = match_element.group(1)
                    el_start_new = int(match_element.group(2))
                    el_end_new = int(match_element.group(3))
                    el_chain_old = match_element_old.group(1)
                    el_start_old = int(match_element_old.group(2))
                    el_end_old = int(match_element_old.group(3))
                    if not (el_chain_new and el_start_new and el_end_new and el_chain_old and el_start_old and el_end_old):
                        raise ValueError("Problem with design contig")
                    if element[0] == seg:
                        matched = True
                        parts_design.append(function(el_chain_new,el_start_new,el_end_new))
                        break
                    elif el_chain_old == seg_chain:
                        if seg_start == el_start_old:
                            matched = True
                            end_res = el_start_new+(seg_end-seg_start)
                            parts_design.append(function(el_chain_new,el_start_new,end_res))
                            break
                        elif seg_end == el_end_old:
                            matched = True
                            start_res = el_end_new-(seg_end-seg_start)
                            if start_res <= 0:
                                raise ValueError("Problem with contig LUT or design contig")
                            parts_design.append(function(el_chain_new,start_res,el_end_new))
                            break
                else:
                    raise ValueError("Problem with contig LUT or design contig")
            if not matched:
                raise ValueError("Design contig element not found in LUT")  
        if align_only_one_chain:
            break # If we aligned the chain we wanted to align, no need to loop through the rest
    if cmd:           
        design_cmd = " or ".join(parts_design)
        input_cmd = " or ".join(parts_input)
        return design_cmd, input_cmd
    return parts_design,parts_input

def expect(cond: bool, path: str, msg: str) -> None:
    if not cond:
        raise ValueError(f"{path}: {msg}")
    
def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan
    
def validate_file_path(path: Union[str, Path]) -> Path:
    """Ensure path exists and is a regular file (or symlink to one)."""
    p = Path(path)
    if not p.exists():
        logging.error("File path not found: %s", p)
        raise FileNotFoundError(p)
    if p.is_dir():
        logging.error("Path exists but is a directory (expected file): %s", p)
        raise IsADirectoryError(p)
    if not p.is_file():  # covers sockets, fifos, etc.
        logging.error("Path exists but is not a regular file: %s", p)
        raise ValueError(f"Not a regular file: {p}")
    logging.debug("File path validation passed: %s", p)
    return p

def validate_dir(path: Union[str, Path],create_if_absent=True,parent=False) -> Path:
    """Ensure `path` (or its parent if `parent=True`) exists as a directory; create it if missing."""
    p = Path(path)
    if parent: p = p.parent
    if p.exists() and not p.is_dir():
        logging.error("Path exists but is not a directory: %s", p)
        raise NotADirectoryError(p)
    if not p.exists() and not create_if_absent:
        logging.error("File path not found: %s", p)
        raise FileNotFoundError(p)
    if not p.exists(): logging.warning(f"Making directory {p}")
    p.mkdir(parents=True, exist_ok=True)
    logging.debug("Directory validation passed: %s", p)
    return p

def validate_conda_env(env_name: str):
    """Check if a conda environment exists."""
    logging.debug(f"Validating conda environment: {env_name}")
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True
        )
        logging.debug("Successfully ran `conda env list --json`")
    except FileNotFoundError:
        logging.error("`conda` command not found")
        raise RuntimeError("`conda` command not found. Is Conda installed and on your PATH?")
    except subprocess.CalledProcessError as e:
        logging.error(f"Conda command failed: {e.stderr.strip()}")
        raise RuntimeError(f"Conda command failed (exit {e.returncode}): {e.stderr.strip()}") from e

    try:
        env_paths = json.loads(result.stdout)["envs"]
        env_names = [Path(p).name for p in env_paths]
        logging.debug(f"Available conda envs: {env_names}")
    except (json.JSONDecodeError, KeyError) as e:
        logging.error("Failed to parse conda env list JSON")
        raise RuntimeError(f"Unexpected JSON format from `conda env list`: {e}") from e

    if env_name not in env_names:
        logging.error(f"Conda env '{env_name}' not found")
        raise ValueError(
            f"Conda environment '{env_name}' not found. "
            f"Available envs: {', '.join(env_names)}"
        )
    logging.debug(f"Conda environment validated: {env_name}")
    
# Check that positive inputs are positive
def validate_positive(value, name):
    logging.debug(f"Validating positive value for {name}: {value}")
    if value <= 0:
        logging.error(f"Validation failed: {name} must be positive (got {value})")
        raise ValueError(f"{name} must be positive (got {value})")
    logging.debug(f"Validation passed: {name} is positive")

def validate_input(args):
    logging.info("Validating input arguments")
    
    if args.run_array and args.no_run_array:
        raise ValueError("run_array and no_run_array both present")
    validate_positive(int(args.b_devices), "b_devices")
    validate_positive(int(args.max_parallel_samples), "max_parallel_samples")

    # RFdiffusion
    if args.only_rfdiff and args.skip_rfdiff:
        raise ValueError("Cannot have only_rfdiff and skip_rfdiff simultaneously")
    if len(args.rf_contig.split()) > len(string.ascii_uppercase):
        raise ValueError(f"Only supports up to {len(string.ascii_uppercase)} chains (A–Z)")
    if args.max_length < args.min_length:
        raise ValueError("max_length must be ≥ min_length")
    validate_file_path(args.rf_script_path)
    validate_file_path(args.rf_input_pdb)
    validate_dir(args.rf_fixpath)
    validate_conda_env(args.rf_env)  # Ensure the conda environment exists
    validate_positive(int(args.rf_num_designs), "rf_num_designs")
    validate_positive(int(args.min_length), "min_length")
    validate_positive(int(args.max_length), "max_length")

    if args.b_percent_of_template is not None:
        p = args.b_percent_of_template
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(
                "b_percent_of_template must be a finite number in [0, 1]."
            )

    # ProteinMPNN
    validate_dir(args.pm_output)
    validate_dir(args.pm_fold,create_if_absent=False)
    validate_dir(args.pm_jsonl)
    validate_positive(args.pm_seq_per_target,'pm_seq_per_target')
    validate_positive(args.pm_batch_size,'pm_batch_size')
    validate_conda_env(args.pm_env)

    # Boltz
    if args.b_use_fixed_residues_template:
        if not args.b_fixed_residue_cif_folder:
            raise ValueError(f"--b_fixed_residue_cif_folder is required when --b_use_fixed_residues_template is set.")

        if not args.template_residues:
            raise ValueError(
                "--template_residues is required when "
                "--b_use_fixed_residues_template is set."
            )
    
        validate_dir(args.b_fixed_residue_cif_folder)
    validate_positive(int(args.b_recycling_steps), "b_recycling_steps")
    validate_positive(int(args.b_sampling_steps), "b_sampling_steps")
    validate_positive(int(args.b_diffusion_samples), "b_diffusion_samples")
    validate_positive(float(args.b_step_scale), "b_step_scale")
    ALLOWED_FORMATS = {"pdb", "mmcif"}
    if args.b_output_format not in ALLOWED_FORMATS:
        raise ValueError(f"b_output_format must be one of {ALLOWED_FORMATS}")
    validate_dir(args.b_out)
    validate_dir(args.b_yaml)

    # AlphaFold
    validate_conda_env(args.af_env)
    validate_dir(args.af_output_path)
    validate_dir(args.af_params_dir,create_if_absent=False)

    # Filtering args
    validate_dir(args.filtered_designs_folder)
    validate_file_path(args.align_structure)
    validate_file_path(args.template_structure)

    # Path to *this* file
    this_file = pathlib.Path(__file__).resolve()

    # Directory that contains it
    this_dir = this_file.parent

    validate_file_path(os.path.join(this_dir,"cavity_analysis.py"))

    validate_positive(args.b_designs_from_pm, "b_designs_from_pm")

    logging.info("All input arguments validated successfully")
    
def run_script_in_env(scriptEnv: str, scriptPath: str, scriptArgs: list,return_result = False):
    logging.info(f"Running script {scriptPath} in env {scriptEnv} with args: {scriptArgs}")    
    try:
        # Run the script passed by the user with its arguments
        # Each script requires its own environment, so we use conda run with scriptEnv
        cmd = ["conda", "run", "-n", scriptEnv, "python", scriptPath] + scriptArgs
        logging.debug(f"Executing command: {shlex.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.debug(f"Command stdout: {result.stdout.strip()}")
        logging.debug(f"Command stderr: {result.stderr.strip()}")
        if result.returncode:
            logging.error(f"Script {scriptPath} failed with exit code {result.returncode}")
            raise RuntimeError(f"Script {scriptPath} failed (exit {result.returncode}): {result.stderr.strip()}")
        logging.info(f"Script {scriptPath} executed successfully")
    except subprocess.CalledProcessError as e:
        logging.exception(f"Error executing {scriptPath} script")
        raise RuntimeError(f"Error executing {scriptPath} script: {e}")
    except FileNotFoundError:
        logging.error(f"Script not found: {scriptPath}")
        raise FileNotFoundError(f"The {scriptPath} script was not found. Please check the path.") 
    if return_result: return result
    
def run_command_in_env(env_name: str, cmd: list[str], return_result = False) -> None:
    """
    Execute *cmd* inside the conda environment *env_name*.
    `cmd` must be a list of tokens, e.g. ["boltz", "predict", ...].
    Raises RuntimeError if the command exits with a non‑zero code.
    """
    full_cmd = ["conda", "run", "-n", env_name] + cmd
    logging.debug("Executing: %s", shlex.join(full_cmd))

    result = subprocess.run(full_cmd, capture_output=True, text=True)
    logging.debug("stdout:\n%s", result.stdout)
    logging.debug("stderr:\n%s", result.stderr)

    if result.returncode:
        raise RuntimeError(
            f"Command {cmd[0]} failed (exit {result.returncode}). "
            "See log for details."
        )
    
    if return_result: return result

def write_sbatch(path: str, content: str):
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))

def submit_sbatch(path: str, parsable: bool=False) -> str:
    cmd = ["sbatch"]
    if parsable:
        cmd.append("--parsable")
    cmd.append(path)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"sbatch failed: {res.stderr.strip()}")
    return res.stdout.strip()

def _parse_chain_order(s: str) -> List[str]:
    """
    Accepts 'A,B,c', 'A B c', 'A,B c', '["A","B","c"]', etc. Returns ['A','B','c'].
    """
    return re.findall(r"[A-Za-z]", s)
