#!/usr/bin/env python3
import argparse as arg
import logging
from pathlib import Path
import re
from typing import List, Tuple
from utils import expect
from colabdesign import mk_af_model # type: ignore
import yaml # type: ignore

def _get_sequence_from_yaml(yaml_path: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (chain_id, sequence) pairs in the order they appear.
    Duplicates are kept if the YAML maps multiple ids to the same sequence.
    """
    logging.debug(f"Extracting sequences from {yaml_path}")

    sequences: List[Tuple[str, str]] = []

    logging.debug(f"Opening YAML file: {yaml_path}")
    with open(yaml_path, "r") as yaml_file:
        doc = yaml.safe_load(yaml_file)
        logging.debug(f"Loaded YAML. Type: {type(doc).__name__}")

        # Guard against incorrect file formatting
        expect(isinstance(doc, dict), yaml_path, "top-level must be a mapping/dict")
        logging.debug(f"Top-level keys: {list(doc.keys())}")

        # Extract sequences block
        seq_block = doc.get("sequences")
        if isinstance(seq_block, dict):
            logging.debug("sequen<ces block is a dict; normalizing to a single-item list")
            seq_block = [seq_block] # Normalize a dict to a list
        expect(isinstance(seq_block, list), f"{yaml_path}.sequences", "must be a list (or a single mapping)")
        logging.debug(f"'sequences' has {len(seq_block)} entries")

        for i, entry in enumerate(seq_block): # Enumeration allows for precise errors
            logging.debug(f"Processing sequences[{i}] with type {type(entry).__name__}")
            expect(isinstance(entry, dict),f"{yaml_path}.sequences.entry",f"entry[{i}] {entry} must be a dict")

            # Extract protein entry
            protein_block = entry.get("protein")
            if protein_block is None:
                # not a protein entity (could be ligand/dna/rna); skip
                logging.debug(f"Entry[{i}] has no 'protein' block; skipping.")
                continue
            logging.debug(f"Entry[{i}] protein block keys: {list(protein_block.keys())}")
            expect(isinstance(protein_block, dict),f"{yaml_path}.sequences.entry[{i}].protein",f"protein entry {protein_block} must be a dict")

            # Each protein block has a sequence entry
            sequence = protein_block.get("sequence")
            logging.debug(f"Entry[{i}] raw sequence present: {sequence is not None}")
            expect(sequence is not None,f"{yaml_path}.sequences.entry[{i}].protein.sequence",f"protein sequence is None")
            expect(isinstance(sequence, str) and sequence != "",f"{yaml_path}.sequences.entry[{i}].protein.sequence",f"Sequence {sequence} must be a string")
            sequence = sequence.strip() # Remove white space
            logging.debug(f"Entry[{i}] sequence length after strip: {len(sequence)}")
            expect(sequence != "",f"{yaml_path}.sequences.entry[{i}].protein.sequence",f"Sequence string must be a non-empty")

            # Each protein block has one chain id or a list of chain ids : homomers can have several chains with the same sequence
            ids = protein_block.get("id") 
            logging.debug(f"Entry[{i}] raw ids: {ids!r}")
            if isinstance(ids, str):
                logging.debug(f"Entry[{i}] id was a string; normalizing to list")
                ids = [ids] # wrap single string in a list for normalization
            expect(isinstance(ids, list),f"{yaml_path}.sequences.entry[{i}].protein.id",f"Could not read/convert IDs {ids} to a list of strings")

            for j, chain_id in enumerate(ids):
                logging.debug(f"Entry[{i}] id[{j}] raw: {chain_id!r}")
                expect(isinstance(chain_id,str),f"{yaml_path}.sequences.entry[{i}].protein.idlist.id",f"ID in id block needs to be a string, id : {chain_id}")
                stripped = chain_id.strip()
                logging.debug(f"Entry[{i}] id[{j}] stripped: {stripped!r}")
                expect(stripped != "",f"{yaml_path}.sequences.entry[{i}].protein.idlist.id",f"ID string in id block needs to be non-empty, id : {chain_id}")
                sequences.append((stripped,sequence))
                logging.debug(f"Appended (chain_id={stripped}, seq_len={len(sequence)})")

    logging.debug(f"Total sequences extracted from {yaml_path}: {len(sequences)}")
    return sequences

def run_inference(yaml_file: str,template_pdb_path: str, af_recycles_amount: int, template_chains: str, colab_data_dir: str, af_initial_guess: bool, af_use_initial_atom_pos: bool, output_path: str) -> None:
    logging.debug(f"run_inference called with yaml_file={yaml_file}, template_pdb_path={template_pdb_path}, "
                 f"af_recycles_amount={af_recycles_amount}, template_chains={template_chains}, "
                 f"colab_data_dir={colab_data_dir}, af_initial_guess={af_initial_guess}, "
                 f"af_use_initial_atom_pos={af_use_initial_atom_pos}, output_path={output_path}")
    sequence_list: List[Tuple[str, str]] = _get_sequence_from_yaml(yaml_file)
    logging.debug(f"Sequence list count: {len(sequence_list)}")

    # Just the sequences, in chain order
    seqs = [seq for _, seq in sequence_list]
    logging.debug(f"Per-chain sequence lengths: {[len(s) for s in seqs]}")

    # Join them for colabdesign input
    joined_seq = "".join(seqs)
    logging.debug(f"Joined sequence: {joined_seq}")
    logging.debug(f"Joined sequence length: {len(joined_seq)}")

    logging.debug(f"Setting up AF run for {yaml_file}")
    af_model = mk_af_model(protocol='fixbb', use_templates=bool(template_pdb_path), 
                            initial_guess=af_initial_guess, 
                            use_initial_atom_pos=af_use_initial_atom_pos, 
                            use_multimer=True,
                            data_dir=colab_data_dir)
    logging.debug(f"mk_af_model created (use_templates={bool(template_pdb_path)}, use_multimer=True)")

    chains = ",".join([c.strip() for c in str(template_chains).split(",") if c.strip()]) or None
    logging.debug(f"template_chains raw={template_chains!r} parsed={chains!r}")

    logging.debug(f"Calling prep_inputs with pdb_filename={template_pdb_path}, chain={chains}, "
                 f"rm_template_seq=True, rm_template_sc=True")
    af_model.prep_inputs(pdb_filename=template_pdb_path, 
                            chain=chains,
                        rm_template_seq=True,  # don't force-copy template sequence
                        rm_template_sc=True)    # don't keep template sidechains
    logging.debug("prep_inputs completed")

    logging.debug("Setting sequence on model")
    af_model.set_seq(joined_seq)
    logging.debug("Sequence set on model")

    logging.debug(f"Running AF prediction for {yaml_file} with num_recycles={af_recycles_amount}")
    af_model.predict(num_recycles=af_recycles_amount)
    logging.debug("Prediction finished; saving PDB")

    af_model.save_pdb(str(output_path))
    logging.debug(f"PDB saved to {output_path}")

def run_af_yaml(args: arg.Namespace, path: str):
    logging.debug(f"run_af_yaml invoked with path={path}")
    # Extract (chain_id, sequence) pairs from YAML
    p = Path(path)
    out_folder = Path(args.af_output_path)
    logging.debug(f"Ensuring output folder exists: {out_folder}")
    out_folder.mkdir(parents=True, exist_ok=True) 
    if p.is_dir():
        logging.debug(f"Path is a directory: {p}")
        for entry in p.iterdir():
            logging.debug(f"Inspecting entry: {entry}")
            if entry.is_dir():
                logging.debug(f"Entry is a directory: {entry}")
                if args.run_for_length is not None:
                    m = re.match(r'length_(\d+)',entry.name)
                    logging.debug(f"Length filter={args.run_for_length}, directory match={(m.group(1) if m else None)}")
                    if not m or int(m.group(1)) != args.run_for_length: 
                        logging.debug(f"Skipping directory {entry} due to length filter")
                        continue
                logging.debug(f"Descending into {entry}")
                run_af_yaml(args,str(entry))
            else: 
                logging.debug(f"Entry is a file: {entry}")
                m = re.match(r'b_yaml_l(\d+)_',entry.stem)
                logging.debug(f"File stem regex result: {(m.group(1) if m else None)}; suffix={entry.suffix.lower()}")
                if m and entry.suffix.lower() in {".yml", ".yaml"}:
                    if args.run_for_length is None or args.run_for_length == int(m.group(1)):
                        output_path = (out_folder / entry.stem).with_suffix(".pdb")
                        logging.debug(f"Launching inference for file {entry} -> {output_path}")
                        run_inference(str(entry),args.template_pdb_path,args.af_recycles_amount,args.template_chains,args.colab_data_dir,args.af_initial_guess,args.af_use_initial_atom_pos,output_path)
                    else:
                        logging.debug(f"Skipping file {entry} due to length filter; wanted {args.run_for_length} got {int(m.group(1))}")
                else:
                    logging.debug(f"Skipping file {entry} (name pattern/suffix did not match)")
    elif p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
        logging.debug(f"Single YAML file detected: {p}")
        m = re.match(r'b_yaml_l(\d+)_',p.stem)
        logging.debug(f"File stem regex result: {(m.group(1) if m else None)}")
        if m:
            if args.run_for_length is None or args.run_for_length == int(m.group(1)):
                output_path = (out_folder / p.stem).with_suffix(".pdb")
                logging.debug(f"Launching inference for single file {p} -> {output_path}")
                run_inference(str(p),args.template_pdb_path,args.af_recycles_amount,args.template_chains,args.colab_data_dir,args.af_initial_guess,args.af_use_initial_atom_pos,output_path)
            else:
                logging.debug(f"Skipping single file {p} due to length filter; wanted {args.run_for_length} got {int(m.group(1))}")
        return

def main():
    # Parse command line arguments
    parser = arg.ArgumentParser(
        prog = "Protein Design Pipeline",
        description = "Protein design pathline for backbone design through RFdiffusion, " \
        "sequence generation through ProteinMPNN, and testing through Boltz.",
    )

    parser.add_argument('-l', '--logfile', type=str, help='Path to the log file', default='pipeline.log')
    parser.add_argument('--yaml_path',dest='yaml_path',type=str,required=True)
    parser.add_argument('--colab_data_dir',dest='colab_data_dir',type=str,required=True) 
    parser.add_argument('--template_pdb_path',dest='template_pdb_path',type=str)
    parser.add_argument('--template_chains',dest='template_chains',type=str)
    parser.add_argument('--af_initial_guess',dest='af_initial_guess',action="store_true")
    parser.add_argument('--af_use_initial_atom_pos',dest='af_use_initial_atom_pos',action="store_true")
    parser.add_argument('--af_recycles_amount',dest='af_recycles_amount',type=int,required=True)
    parser.add_argument('--af_output_path', dest='af_output_path', type=str, required=True,
                    help='Folder where predicted PDBs will be written')
    parser.add_argument('--run_for_length',dest='run_for_length',type=int)
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        filename=args.logfile,
        filemode='a',
        level=logging.DEBUG, # ADD CONSISTENCY WHITH MASTER SCRIPT 
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.debug("=== Protein Design Pipeline: start ===")
    logging.debug(f"CLI args parsed: {args}")

    # TODO: IMPLEMENT NON-LENGTH RUNS !
    run_af_yaml(args,args.yaml_path)
    logging.debug("=== Protein Design Pipeline: finished ===")

if __name__ == "__main__":
    main()
