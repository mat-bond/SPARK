#!/usr/bin/env python3
import logging
import json
import os
import re
import shutil
from utils import validate_dir, validate_file_path, run_script_in_env,_get_rf_diffusion_model

#-------------------------------------- ProteinMPNN utility --------------------------------------------#
def run_parse_multiple_chains_py(args,jsonl_input, jsonl_output_file):

    # Deduce and validate path to script 
    script_path = os.path.join(args.pm_fold, "helper_scripts", "parse_multiple_chains.py")
    logging.debug(f"Preparing to run parse_multiple_chains.py: {script_path}")
    validate_file_path(script_path)

    # Construct arguments for the parser script
    scriptArgs = [
        f"--input_path={jsonl_input}",
        f"--output_path={jsonl_output_file}"
    ]
    logging.debug(f"Parse multiple chains args: {scriptArgs}")

    # Run helper script in ProteinMPNN conda environment
    run_script_in_env(args.pm_env, script_path, scriptArgs)

def create_fixed_jsonl(args, input_jsonl, output_folder, length):
    """
    Reads the fixed_chain.json produced by fix_rfdiffusion_output for the given length,
    then calls the ProteinMPNN helper script with those chain and position lists.
    """
    logging.info(f"Creating fixed JSONL for length {length}")
    # Path to the RFdiffusion JSON file
    fix_json_path = os.path.join(args.rf_fixpath, f"length_{length}", "fixed_chain.json")
    validate_file_path(fix_json_path)
    with open(fix_json_path, 'r') as jfh:
        data = json.load(jfh)
    chain_list = data["chain_list"]
    position_list = data["fixed_chain"]
    logging.debug(f"Read chain_list: {chain_list}, position_list: {position_list}")

    # Path to the helper script
    script_path = os.path.join(args.pm_fold, "helper_scripts", "make_fixed_positions_dict.py")
    validate_file_path(script_path)

    # Prepare output folder for this length
    os.makedirs(output_folder, exist_ok=True)
    output_jsonl = os.path.join(output_folder, "fixed_chains_JSONL.jsonl")
    logging.debug(f"Output fixed JSONL path: {output_jsonl}")

    # Build and run the command
    script_args = [
        f"--input_path={input_jsonl}",
        f"--chain_list={chain_list}",
        f"--position_list={position_list}",
        f"--output_path={output_jsonl}",
    ]
    run_script_in_env(args.pm_env, script_path, script_args)

def create_tied_jsonl(args, input_jsonl, output_folder, length):
    logging.info(f"Creating tied JSONL for length {length}")
    # Path to the helper script
    script_path = os.path.join(args.pm_fold, "helper_scripts", "make_tied_positions_dict.py")
    validate_file_path(script_path)

    # Prepare output folder for this length
    os.makedirs(output_folder, exist_ok=True)
    output_jsonl = os.path.join(output_folder, "tied_chains_JSONL.jsonl")
    logging.debug(f"Output tied JSONL path: {output_jsonl}")

     # Build and run the command
    script_args = [
        f"--input_path={input_jsonl}",
        f"--homooligomer=1",
        f"--output_path={output_jsonl}",
    ]
    run_script_in_env(args.pm_env, script_path, script_args)

def _make_jsonl_files(args,jsonl_output_folder,full_path,length):
    os.makedirs(jsonl_output_folder, exist_ok=True)

    # Build the output JSONL filepath
    output_jsonl = os.path.join(jsonl_output_folder, 'parsedJSONL.jsonl')
    logging.debug(f"Parsed JSONL output path: {output_jsonl}")
    
    # Run helper script provided by ProteinMPNN
    run_parse_multiple_chains_py(args,full_path,output_jsonl)
    validate_file_path(output_jsonl)

    # Generate fixed-sequence JSONL
    create_fixed_jsonl(args,output_jsonl,jsonl_output_folder,length)

    # Generate tied-sequence JSONL
    create_tied_jsonl(args,output_jsonl,jsonl_output_folder,length)

def generate_jsonl(args,pdb_path=None,length=None,output_path=None):
    """
    ProteinMPNN requires a jsonl file instead of pdb files and provides by default a helper script for it
    """
    logging.info("Generating JSONL files for ProteinMPNN")

    if output_path is None: 
        output_path = args.pm_jsonl

    if pdb_path is None:
        # Loop through all items in the fixed output directory
        for item in os.listdir(args.rf_fixpath):
            full_path = os.path.join(args.rf_fixpath, item)
            
            # Only process directories that match our length pattern
            if os.path.isdir(full_path) and item.startswith("length_"):
                logging.debug(f"Found length directory: {item}")
                try:
                    # Extract the length number from folder name
                    length = int(item.split("_")[1])
                    if args.run_pmpnn_for_length is not None and length != args.run_pmpnn_for_length: continue
                    jsonl_output = os.path.join(output_path, f"length_{length}")
                    _make_jsonl_files(args,jsonl_output,full_path,length)
                    if args.run_pmpnn_for_length is not None: break
                except (IndexError, ValueError):
                    logging.warning(f"Skipping malformed folder: {item}")
    else:
        # Generate JSON files only for the passed PDB into the passed file path
        if length is None or output_path is None: 
            raise ValueError(f"Length not passed to generate_jsonl with pdb path {pdb_path}")
        jsonl_output = os.path.join(output_path, f"length_{length}") 
        os.makedirs(jsonl_output, exist_ok=True)
        full_path = os.path.join(pdb_path, f"length_{length}")
        validate_dir(full_path,create_if_absent=False) 
        _make_jsonl_files(args,jsonl_output,full_path,length)
        
def build_proteinMPNN_CLI(args,parsed_jsonl,fixed_jsonl,tied_jsonl,output_folder,glycosylate_best_designs=False):
    pm_args = [ 
    f"--jsonl_path={parsed_jsonl}",
    f"--fixed_positions_jsonl={fixed_jsonl}",
    f"--tied_positions_jsonl={tied_jsonl}",
    f"--num_seq_per_target={args.pm_seq_per_target}",
    f"--batch_size={args.pm_batch_size}",
    f"--sampling_temp={args.pm_sampling_temp}",
    f"--seed={args.pm_seed}",
    f"--out_folder={output_folder}",
    # Add these if using solubilization mode
    *(["--use_soluble_model"] if args.pm_sol else []),

    # Reserved for a future glycosylation workflow; not currently exposed by the SPARK CLI.
    *(["--unconditional_probs_only=1"] if glycosylate_best_designs else []),
    
    ]
    return pm_args

#-------------------------------------- Run script functions --------------------------------------------# 
def run_proteinMPNN(args,length=None,pdb_name=None,output_path=None):
    """
    1) Generates parsed, fixed, and tied JSONL files for each linker length.
    2) Invokes protein_mpnn_run.py once per length, passing the actual JSONL file paths.
    """
    logging.info("Starting ProteinMPNN pipeline")

    # Check coherence
    batch_mode  = (pdb_name is None and length is None and output_path in (None, args.pm_jsonl))
    single_mode = (pdb_name is not None and length is not None and output_path is not None)
    if not (batch_mode or single_mode):
        raise ValueError("Provide (pdb_name & length & output_path) for single-design, or none for batch mode.")

    if not output_path: 
        input_path = args.pm_jsonl
    else:
        input_path = output_path
    if length is None:
        if args.run_pmpnn_for_length is not None:
            length = args.run_pmpnn_for_length

    if pdb_name is not None:
        rf_pdb_path = _get_rf_diffusion_model(args.rf_fixpath,pdb_name) # Deduces length from pdb_name
        validate_file_path(rf_pdb_path)
        new_location = os.path.join(output_path,f"length_{length}")
        os.makedirs(new_location, exist_ok=True)
        shutil.copy(rf_pdb_path,new_location) 
    pdb_folder_path = output_path

    # Prepare all JSONLs
    generate_jsonl(args,pdb_folder_path,length,output_path)

    # Locate the ProteinMPNN runner
    script_path = os.path.join(args.pm_fold, "protein_mpnn_run.py")
    validate_file_path(script_path)
    logging.debug(f"ProteinMPNN runner script: {script_path}")

    # For each length folder under pm_jsonl, run ProteinMPNN
    for entry in os.listdir(input_path):
        if not entry.startswith("length_"):
            continue
        length_folder = entry  # e.g. "length_12"
        if length is not None:
            m = re.match(r'length_(\d+)',length_folder)
            if not m or (int(m.group(1))!=length): continue
        logging.info(f"Running ProteinMPNN for folder: {length_folder}")

        # actual JSONL file paths
        parsed_jsonl = os.path.join(input_path,    length_folder, "parsedJSONL.jsonl")
        fixed_jsonl  = os.path.join(input_path, length_folder, "fixed_chains_JSONL.jsonl")
        tied_jsonl   = os.path.join(input_path,  length_folder, "tied_chains_JSONL.jsonl")
        logging.debug(f"JSONL paths: {parsed_jsonl}, {fixed_jsonl}, {tied_jsonl}")

        # sanity-check
        validate_file_path(parsed_jsonl)
        validate_file_path(fixed_jsonl)
        validate_file_path(tied_jsonl)

        # output subfolder for this length
        output_subfolder = os.path.join(args.pm_output, length_folder) if output_path is None else os.path.join(output_path, length_folder)
        os.makedirs(output_subfolder, exist_ok=True)
        logging.debug(f"ProteinMPNN output folder: {output_subfolder}")

        # build the actual CLI args, pointing at the files
        pm_args = build_proteinMPNN_CLI(args,parsed_jsonl,fixed_jsonl,tied_jsonl,output_subfolder)
        logging.debug(f"ProteinMPNN args: {pm_args}")

        # run ProteinMPNN for this length
        run_script_in_env(args.pm_env, script_path, pm_args)
        if args.run_pmpnn_for_length is not None: break
    if args.run_pmpnn_for_length is None:
        logging.info("Completed ProteinMPNN pipeline")
    else:
        logging.info(f"Completed ProteinMPNN pipeline for length {args.run_pmpnn_for_length}")