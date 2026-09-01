#!/usr/bin/env python3
from argparse import Namespace
import math
import pathlib
import shutil
import subprocess
import logging
import shlex
import os
import re
from typing import List, Tuple
import tempfile
import textwrap
import numpy as np # type: ignore
import matplotlib # type: ignore
matplotlib.use("Agg")   # non-interactive backend
import json as js
import pandas as pd # type: ignore
from utils import run_script_in_env,_as_float,_get_residue_strings,basic_format_residues,build_lut_from_contig,_get_file_from_same_dir,_to_int_na,_truthy,chain_offsets_from_lut,global_index,_index_of_closest_lowest_value,_index_above_closest_highest__value,_get_rf_diffusion_model
from constants import Stats
from plots_and_graphs import create_stats_graphs,_make_pae_heat_map

#-------------------------------------- Filtering  --------------------------------------------#
def rmsd_pymol_string(new_id,first_res,last_res):
    return f"(chain {new_id} and resi {str(first_res)}-{str(last_res)})"

def compute_rmsd_with_pymol(env_name: str,
                            ref_pdb: str,
                            pred_pdb: str,
                            sel_ref: str,
                            sel_pred: str,
                            cycles: int = 5,full_align_struct=False) -> float:
    """
    Return the RMSD (Å) between *sel_ref* in *ref_pdb* and *sel_pred* in
    *pred_pdb*, using PyMOL’s `align` (which superposes + refines).

    Parameters
    ----------
    env_name   : conda environment that has PyMOL installed
    ref_pdb    : path to the original/input structure
    pred_pdb   : path to the designed/predicted structure
    sel_ref    : PyMOL selection string (old numbering)
    sel_pred   : PyMOL selection string (new numbering)
    cycles     : refinement cycles for `align` (default 5; set 0 to skip)
    """

    if not full_align_struct:
        pml_text = textwrap.dedent(f"""
            load {ref_pdb}, ref
            load {pred_pdb}, pred
            select sel_ref,  (ref and ({sel_ref}))
            select sel_pred, (pred and ({sel_pred}))
            # run align; PyMOL prints a summary we can parse
            align sel_pred, sel_ref, cycles={cycles}
            quit
        """)
    else:
        pml_text = textwrap.dedent(f"""
            load {ref_pdb}, ref
            load {pred_pdb}, pred
            # run align; PyMOL prints a summary we can parse
            align pred, ref, cycles={cycles}
            quit
        """)

    logging.debug("Creating tempfile")
    logging.debug(f"Tempfile input : {pml_text}")
    with tempfile.NamedTemporaryFile("w", suffix=".pml", delete=False) as fh:
        fh.write(pml_text)
        pml_path = fh.name

    cmd = ["pymol", "-cq", pml_path]
    full_cmd = ["conda", "run", "-n", env_name] + cmd
    logging.debug("Running PyMOL: %s", shlex.join(full_cmd))

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True)  
        logging.debug("stdout:\n%s", result.stdout)
        logging.debug("stderr:\n%s", result.stderr)

        m = re.search(r"Executive:\s*RMSD\s*=\s*([\d.]+)", result.stdout)
        if result.returncode != 0 or not m:
            logging.error("PyMOL align failed:\nstdout:\n%s\nstderr:\n%s",
                        result.stdout, result.stderr)
            raise RuntimeError("PyMOL align failed (see log for details)")
        
    finally:
        # Remove temp file
        os.unlink(pml_path)
    
    return float(m.group(1))

def calculate_RMSD(rmsd_alignment_residues,alignment_contig,predicted_file_path,linker_length,args,align_structure,full_align=False):

    # Build commands for residue selection
    if not full_align:
        lut = build_lut_from_contig(linker_length,alignment_contig.split())
        design_sel_cmd, input_sel_cmd = _get_residue_strings(args.align_one_chain_only,rmsd_alignment_residues,alignment_contig,linker_length,lut,rmsd_pymol_string)
    else: 
        design_sel_cmd = ""
        input_sel_cmd = ""
    
    return compute_rmsd_with_pymol(
        env_name=args.py_env,
        ref_pdb=align_structure,
        pred_pdb=predicted_file_path,
        sel_ref=input_sel_cmd,
        sel_pred=design_sel_cmd,
        cycles=args.pymol_alignment_cycles,
        full_align_struct=full_align          
    )

def _get_pae_matrix(desired_quantity,second_name,pae_file):
    # Load numpy file 
    with np.load(pae_file) as data:
        if desired_quantity in data:
            matrix = data[desired_quantity]
        elif second_name in data:  # Different models/versions have different naming conventions
            matrix = data[second_name]
        else:
            logging.warning(f"No {desired_quantity} matrix found in NPZ {pae_file}; expected {desired_quantity} or {second_name}")  
            return
        
        return matrix

def calculate_pae(residue_contig,contig,linker_length,predicted_file_path,lut,desired_quantity,second_name):
    logging.debug(f"Starting PAE calculation for {predicted_file_path}")
    # Build residue selection command for designed residues 
    design_list,not_of_use = _get_residue_strings(False,residue_contig,contig,linker_length,lut,basic_format_residues,cmd=False,use_link_and_des=True)

    # Load numpy file 
    matrix = _get_pae_matrix(desired_quantity,second_name,predicted_file_path)
    if matrix is None:
        raise KeyError(f"No {desired_quantity}/{second_name} in {predicted_file_path}")

    offsets = chain_offsets_from_lut(lut)

    # Calculate average for selection
    value_sum = 0.0
    value_count = 0.0

    for first_entry in design_list:
        for second_entry in design_list:
            ch1,r1_start,r1_end = first_entry
            ch2,r2_start,r2_end = second_entry
            # Guard against unknown chains 
            # convert letter → numeric id (A→0, B→1, …)
            id1 = ord(ch1.upper()) - ord('A')
            id2 = ord(ch2.upper()) - ord('A')
            if id1 not in offsets or id2 not in offsets:
                raise ValueError(f"Chain(s) {ch1}/{ch2} not present in {desired_quantity} file")
            s1_start = global_index(offsets,ch1,r1_start)
            s1_end = global_index(offsets,ch1,r1_end)
            s2_start = global_index(offsets,ch2,r2_start)
            s2_end = global_index(offsets,ch2,r2_end)
            logging.debug(f"Building PAE matrix slice between designed residues {ch1}{r1_start}-{r1_end} (Global {s1_start}-{s1_end}) and {ch2}{r2_start}-{r2_end} (Global {s2_start}-{s2_end})")
            slice_1 = slice(s1_start,s1_end+1)
            slice_2 = slice(s2_start,s2_end+1)
            block = matrix[slice_1,slice_2]
            value_sum += block.sum()
            value_count += block.size
    if value_count == 0: 
        raise ValueError(f"No entries found for {desired_quantity} in design_list")  
      
    des_avg_pae = value_sum/value_count
    full_avg_pae = np.nanmean(matrix)
    return des_avg_pae,full_avg_pae 

def calculate_average_pde(
    residue_contig: str,
    contig: str,
    linker_length: int,
    npz_path: str,
    lut: List[Tuple[str, str]],
) -> float:
    logging.debug(f"Starting PDE calculation for {npz_path}")
    """
    Return avg_des_pde: mean PDE over the selected designed residues,
    counting each unordered pair once (upper triangle incl. diagonal, i ≤ j).
    """
    #---- Designed residue average ----
    # Which residues?
    design_list, _ = _get_residue_strings(
        False,
        residue_contig,
        contig,
        linker_length,
        lut,
        basic_format_residues,
        cmd=False,
        use_link_and_des=True
    )

    # 2) Load the symmetric PDE matrix
    with np.load(npz_path) as data :
        key = "predicted_distance_error" if "predicted_distance_error" in data else "pde"
        matrix = data[key]

    # 3) Offsets table → global indices
    offsets = chain_offsets_from_lut(lut)

    value_sum = 0.0
    value_count = 0

    for idx_a, (ch1, r1_start, r1_end) in enumerate(design_list):
        slice_a = slice(
            global_index(offsets, ch1, r1_start),
            global_index(offsets, ch1, r1_end) + 1,
        )
        logging.debug(f"Building PDE matrix slice between designed residues {ch1}{r1_start}-{r1_end} and themselves")

        # ─ diagonal block (A,A): upper triangle only (i ≤ j)
        block = matrix[slice_a, slice_a]
        r, c = np.triu_indices_from(block, k=0)
        value_sum += block[r, c].sum()
        value_count += len(r)  # n*(n+1)/2

        # ─ off‑diagonal blocks (A,B) with B > A
        for ch2, r2_start, r2_end in design_list[idx_a + 1 :]:
            slice_b = slice(
                global_index(offsets, ch2, r2_start),
                global_index(offsets, ch2, r2_end) + 1,
            )
            block = matrix[slice_a, slice_b]
            logging.debug(f"Building PDE matrix slice between designed residues {ch1}{r1_start}-{r1_end} and {ch2}{r2_start}-{r2_end}")
            value_sum += block.sum()
            value_count += block.size  # counted once, no mirror

    if value_count == 0:
        raise ValueError("No residues matched for PDE averaging")

    avg_des_pde = value_sum / value_count

    return  avg_des_pde

def calculate_designed_avg_plddt(residue_contig,contig,linker_length,predicted_file_path,lut) -> float:
    """
    Average pLDDT over the *designed* residues defined by `residue_contig`.
    Works exactly like `calculate_average` but for the 1‑dimensional
    pLDDT array.
    """
    logging.debug(f"Starting pLDDT calculation for {predicted_file_path}")
    # Extract residues of the designed part in the new design's chain and residue numbering
    design_list,not_of_use = _get_residue_strings(False,residue_contig,contig,linker_length,lut,basic_format_residues,cmd=False,use_link_and_des=True)

    # Load numpy file and find the pLDDT entry
    with np.load(predicted_file_path) as data:
        found = False
        for key in ("plddt", "predicted_lddt", "predicted_LDDT"):
            if key in data:
                plddt = data[key]          # shape (L,)
                found = True
                logging.debug("pLDDT found, array shape: %s", plddt.shape)  
                break
        if not found :
            raise KeyError(
                f"pLDDT not found in {predicted_file_path}; "
                "looked for keys: 'plddt', 'predicted_lddt', 'predicted_LDDT'"
            )

    # Build dictionary for residue positions
    offsets = chain_offsets_from_lut(lut)

    # Accumulate total pLDDT for the selected residues
    total = 0.0
    res_count = 0.0

    for ch,r_start,r_end in design_list:
        ch_id = ord(ch.upper()) - ord('A') # Convert letter to corresponding ASCII number
        if ch_id not in offsets:
            raise ValueError(f"Couldn't find chain {ch} in dictionary")
        
        # Extract pLDDT block associated with current entry in design list
        logging.debug(f"Adding to pLDDT sum block of residues between {ch}{r_start} and {ch}{r_end}, file: {predicted_file_path}")
        block = plddt[global_index(offsets,ch,r_start):global_index(offsets,ch,r_end)+1] # Inclusive slice
        total += block.sum()
        res_count += block.size

    logging.debug(f"Final pLDDT residue count: {res_count}, file: {predicted_file_path}")

    if res_count == 0:
        raise ValueError("No residues matched for pLDDT averaging")

    return total/res_count

def _get_cavity_script_args(args: Namespace, pdb_path: str, linker_length: int) -> list:

    # Convert the default arguments to a dict
    arg_dict = vars(args).copy() 

    # Add the cavity-specific arguments
    arg_dict["pdb_path"] = str(pdb_path)
    arg_dict["linker_length"] = linker_length

    # Build the list
    script_args = []

    for key,val in arg_dict.items():
        if val is None: continue 
        
        val = _truthy(val)

        # Depending on the type of argument, we need to store it differently

        # --------- booleans ----------
        if isinstance(val,bool): 
            if val:
                script_args.append(f"--{key}")
            continue

        # --------- sequences (e.g. repeatable CLI options) ----------
        if isinstance(val, (list, tuple)):
            for item in val:
                item = _truthy(item)
                script_args.append(f"--{key}={item}")
            continue

        # --------- scalars ----------

        script_args.append(f"--{key}={val}")

    return script_args

def _parse_result_cavity(cmd_result):

    m = re.search(r"fixed_total_volume=([\d.]+),fixed_avg_volume=([\d.]+),fixed_count=([\d.]+),designed_total_volume=([\d.]+),designed_avg_volume=([\d.]+),designed_count=([\d.]+)", cmd_result.stdout)

    if m is None: raise RuntimeError(f"Could not find values in cavity result: {cmd_result.stdout}")

    return [float(m.group(i)) for i in range (1,7)]

def calculate_cavity_metrics(args,pdb_path, linker_length)-> Tuple[
                                                                    float,  # fixed_total_volume
                                                                    float,  # fixed_avg_volume
                                                                    float,    # fixed_count
                                                                    float,  # designed_total_volume
                                                                    float,  # designed_avg_volume
                                                                    float,    # designed_count
                                                                ]:
    
    # Find the directory to the cavity script
    this_file = pathlib.Path(__file__).resolve()

    # src/stages/analysis.py -> src/
    src_dir = this_file.parent.parent

    cavity_analysis_script_path = str(
        src_dir / "cavity_analysis.py"
    )


    # Build arguments for cavity script
    script_args = _get_cavity_script_args(args,pdb_path, linker_length)

    # Run and capture result
    cmd_result = run_script_in_env(args.pykvfinder_env,cavity_analysis_script_path,script_args,return_result=True)

    # Parse result
    fixed_total_volume, fixed_cavity_averageVolume, fixed_cavity_amount, designed_total_volume, designed_cavity_averageVolume, designed_cavity_amount = _parse_result_cavity(cmd_result)
    return fixed_total_volume, fixed_cavity_averageVolume, int(fixed_cavity_amount), designed_total_volume, designed_cavity_averageVolume, int(designed_cavity_amount)
        
def _get_full_boltz_reported_metrics(predicted_file_path):
    full_pde = float('nan')
    full_plddt = float('nan')

    try: 
        # json.load expects a file object
        p = pathlib.Path(predicted_file_path) 
        with p.open("r") as file:
            data = js.load(file)
            # Extract with safety
            full_pde = _as_float(data.get("complex_pde"))
            full_plddt = _as_float(data.get("complex_plddt"))
    except:
        return 0,0

    if math.isnan(full_pde):
        full_pde = 0
        logging.warning(f"Could not extract full PDE average from prediction file {predicted_file_path}")
    if math.isnan(full_plddt):
        full_plddt = 0
        logging.warning(f"Could not extract full pLDDT average from prediction file {predicted_file_path}")

    return full_pde,full_plddt 

def _get_af_metrics(args,file_path):
    af_pdb_path = None
    af_boltz_rmsd = float('nan')
    af_full_pae = float('nan')
    af_full_plddt = float('nan')
    af_des_pae = float('nan')
    af_des_plddt = float('nan')
    return af_pdb_path, af_boltz_rmsd, af_full_pae, af_full_plddt, af_des_pae, af_des_plddt

def _analyse_boltz_prediction_folder(args,prediction_folder,folder_path,linker_length):
    logging.debug(f"Analysing with {prediction_folder}")
    pdb_path = ""
    rmsd = 0
    des_avg_pae = 0
    des_avg_pde = 0
    avg_designed_plddt = 0
    rmsd_alignment_residues = args.rmsd_alignment_residues
    alignment_contig = args.alignment_contig
    designed_residues_contig = args.metric_residues
    contig = args.boltz_chain_break_contig
    rf_rmsd = float('nan')
    full_pae = 0 
    full_pde = 0 
    full_plddt = 0  
    fixed_total_volume = float('nan')
    fixed_cavity_averageVolume = float('nan')
    fixed_cavity_amount = float('nan')
    designed_total_volume = float('nan')
    designed_cavity_averageVolume = float('nan')
    designed_cavity_amount = float('nan')
    for predicted_file in os.listdir(folder_path):
            
            predicted_file_path = os.path.join(folder_path,predicted_file)

            logging.debug(f"Looking at predicted_file: {predicted_file_path}")

            chains_full = args.boltz_chain_break_contig.split()
            lut = build_lut_from_contig(linker_length,chains_full)

            if predicted_file.startswith('confidence_'):
                if predicted_file.endswith('.json'):
                    full_pde,full_plddt = _get_full_boltz_reported_metrics(predicted_file_path)
                else:
                    logging.warning(f"Confidence file not in the required JSON format: {predicted_file}")

            if predicted_file.endswith('.pdb'):
                pdb_path = predicted_file_path
                # Calculate RMSD after alignment to non-designed input structure
                rmsd = calculate_RMSD(rmsd_alignment_residues, alignment_contig, pdb_path, linker_length, args, args.align_structure)

                # Calculate RMSD after aligning to RFdiffusion generated model (after chain+residue renaming & renumbering)
                rf_align_structure = _get_rf_diffusion_model(args.rf_fixpath,predicted_file,linker_length)
                if rf_align_structure:
                    rf_rmsd = calculate_RMSD(args.rmsd_alignment_residues, args.alignment_contig, pdb_path, linker_length, args, rf_align_structure,full_align=True)
                else:
                    logging.warning(f"No RF model for {predicted_file}; skipping RF_RMSD")

                # Calculate cavity metrics
                fixed_total_volume, fixed_cavity_averageVolume, fixed_cavity_amount, designed_total_volume, designed_cavity_averageVolume, designed_cavity_amount = calculate_cavity_metrics(args,pdb_path, linker_length)
                #af_pdb_path, af_boltz_rmsd, af_full_pae, af_full_plddt, af_des_pae, af_des_plddt = _get_af_metrics(args,predicted_file_path) 

            if  predicted_file.startswith('pae_'):
                des_avg_pae,full_pae = calculate_pae(designed_residues_contig,contig,linker_length,predicted_file_path,lut,"predicted_aligned_error","pae")

            if predicted_file.startswith('pde_'):
                des_avg_pde = calculate_average_pde(designed_residues_contig,contig,linker_length,predicted_file_path,lut)

            if predicted_file.startswith('plddt_'):
                avg_designed_plddt = calculate_designed_avg_plddt(designed_residues_contig,contig,linker_length,predicted_file_path,lut)

    if pdb_path == "":
        raise ValueError('Boltz output pdb not found')
    return (pdb_path, rmsd, rf_rmsd, des_avg_pae, des_avg_pde, avg_designed_plddt, full_pae, full_pde, full_plddt, fixed_total_volume, fixed_cavity_averageVolume, 
            fixed_cavity_amount, designed_total_volume, designed_cavity_averageVolume, designed_cavity_amount, linker_length)

def create_boltz_output_log(args):
    stats = []
    for entry in os.listdir(args.b_out):
        if not entry.startswith('length_'):
            continue 
        entry_path = os.path.join(args.b_out, entry)
        for length_folder in os.listdir(entry_path):
            if not length_folder.startswith('boltz_results_length_'):
                continue 
            match = re.match(r'^boltz_results_length_(\d+)$', length_folder)  
            if match:
                linker_length = int(match.group(1)) 
                length_path = os.path.join(entry_path, length_folder)
                logging.info(f"Analysing Boltz output for {length_folder}")
                logging.debug(f"Length path: {length_path}")
                predictions_folder = os.path.join(length_path,"predictions")
                for pred_entry in os.listdir(predictions_folder):
                    if not pred_entry.startswith('b_yaml_'):
                        continue
                    pred_entry_path = os.path.join(predictions_folder,pred_entry)
                    logging.debug(f"Calling analyse_prediction_folder with {pred_entry_path}")
                    row = _analyse_boltz_prediction_folder(args,pred_entry,pred_entry_path,linker_length)
                    if not (len(row) == len(Stats)): raise RuntimeError(f"Tuple length {len(row)} / Stats enum length {len(Stats)} mismatch")
                    stats.append(row)   
            else:
                raise ValueError(f'"{length_folder}" does not match the expected pattern')
    return stats        

def create_stats_file(stats,filtered_designs_folder,name="output_stats"):
    out_path = os.path.join(filtered_designs_folder, name)
    if len(stats) == 0: return
    with open(out_path, 'a') as outfile:
        # Write header row
        logging.info("Writing to stats file")
        char_shift_size = len(stats[0][Stats.PATH]) 
        char_shift = char_shift_size*"#"
        header = " | ".join(stat.name for stat in Stats)
        outfile.write(f"{char_shift} {header}\n")

        # Write stats
        for entry in stats:    
            line = " | ".join(str(x) for x in entry)
            outfile.write(f">{line}\n")

def create_stats_xlsx(stats, filtered_designs_folder, name="output_stats"):
    if not stats:
        return

    name = name + ".xlsx"
    
    out_path = os.path.join(filtered_designs_folder, name)

    try:
        headers = [s.name for s in Stats]  
    except NameError:
        headers = [f"col{i}" for i in range(len(stats[0]))]

    df = pd.DataFrame(stats, columns=headers)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="stats", index=False)
        ws = writer.sheets["stats"]
        wb = writer.book

        header_fmt = wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
        cell_fmt   = wb.add_format({"align": "left", "valign": "top"})

        # Resize columns based on content length, and apply alignment
        for col_idx, col in enumerate(df.columns):
            maxlen = max(len(str(col)), *(len(str(v)) for v in df[col].astype(str)))
            ws.set_column(col_idx, col_idx, min(maxlen + 2, 60), cell_fmt)

        # Re-write headers with the header format
        for col_idx, col in enumerate(df.columns):
            ws.write(0, col_idx, col, header_fmt)

        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

def process_best_designs(args,best_designs_list):
    for i, design in enumerate(best_designs_list):
        # Make design mutable (from tuple to list)
        design = list(design)

        # Deduce file path and pdb file name
        in_path = design[Stats.PATH]
        pdb_name = os.path.splitext(os.path.basename(in_path))[0]
        out_path =  os.path.join(args.filtered_designs_folder,pdb_name)

        # Prepare output directory 
        os.makedirs(out_path, exist_ok=True)
        shutil.copy(in_path,out_path) 

        # Make PAE heat map and calculate RMSD after aligning to RFdiffusion generated model (after chain+residue renaming & renumbering)
        linker_length = design[Stats.LENGTH]
        # Find the pae file in the same folder as the designed pdb
        pae_file = _get_file_from_same_dir(in_path,'pae_')
        if pae_file:
            # Extract pae matrix
            matrix = _get_pae_matrix("predicted_aligned_error","pae",pae_file) 
            _make_pae_heat_map(out_path,args.designed_residues_contig,args.boltz_chain_break_contig,linker_length,matrix)

        # Write back to tuple
        best_designs_list[i] = tuple(design) 


def get_stats_from_file(args: Namespace, stats_file_name: str = "global_stats.xlsx"):
    stats = []
    stats_file_path = os.path.join(args.filtered_designs_folder, stats_file_name)

    if not os.path.isfile(stats_file_path):
        logging.warning("Stats file not found: %s", stats_file_path)
        return stats

    conversion_functions = {
                            Stats.PATH.name: str,
                            Stats.FIX_CAVITYCOUNT.name: _to_int_na,
                            Stats.DES_CAVITYCOUNT.name: _to_int_na,
                            Stats.LENGTH.name:          _to_int_na,
                        }
    data = pd.read_excel(stats_file_path,converters=conversion_functions,sheet_name="stats")

    # Enforce the enum column order (so tuple indices match Stats values)
    data = data[[s.name for s in Stats]]

    stats = list(data.itertuples(index=False, name=None))

    return stats

def run_analysis(args):
    """
    Analyse Boltz predictions by:
    - Calculating RMSD vs input using PyMOL
    - Computing average PAE, PDE, and pLDDT over designed residues
    - Measuring cavity volumes (fixed vs designed regions)
    - Filtering designs based on thresholds
    - Copying best structures and generating annotated PAE heatmaps
    - Writing statistics file and summary graphs
    """
    if not args.stats_read_file_only:
        # Create statistics for design outputs
        stats = create_boltz_output_log(args)
    else:
        # Read stats already from file
        stats = get_stats_from_file(args)

    # Sort statistics and RMSD<=cutoff
    stats.sort(key=lambda t: t[Stats.RMSD])
    best_designs = stats[:_index_above_closest_highest__value(stats,Stats.RMSD,args.rmsd_cutoff)]

    # Within the designs with the best rmsd, only keep those with pLDDT >= cutoff
    best_designs.sort(key=lambda t: t[Stats.DES_PLDDT])
    best_designs = best_designs[_index_of_closest_lowest_value(best_designs,Stats.DES_PLDDT,args.plddt_cutoff):]
    # Keep pAE <= cutoff
    best_designs.sort(key=lambda t: t[Stats.DES_PAE])
    best_designs = best_designs[:_index_above_closest_highest__value(best_designs,Stats.DES_PAE,args.pae_cutoff)] 

    # Keep pDE <= cutoff
    best_designs.sort(key=lambda t: t[Stats.DES_PDE]) 
    best_designs = best_designs[:_index_above_closest_highest__value(best_designs,Stats.DES_PDE,args.pde_cutoff)] 

    # Select final_selection_amount amount of designs with lowest RMSD in the designed regions
    best_designs.sort(key=lambda t: t[Stats.RMSD],reverse=False)
    best_designs = best_designs[:(min(args.final_selection_amount,len(best_designs)))]

    # Copy best selections to the filtered designs folder, make PAE heatmap and calculate RMSD after alignment to RFdiffusion model
    process_best_designs(args,best_designs)

    # Create and write to stats file with best outputs and all statistics:
    create_stats_xlsx(best_designs,args.filtered_designs_folder,"best_stats")     
    create_stats_xlsx(stats,args.filtered_designs_folder,"global_stats")

    # Create statistics plots/graphs
    create_stats_graphs(stats,args.filtered_designs_folder)
