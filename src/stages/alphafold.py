#!/usr/bin/env python3
import argparse as arg
import logging
import os
import pathlib
import re
import gemmi
from utils import parse_contig, run_script_in_env

def last_two_ca_positions(chain: gemmi.Chain):
    logging.debug(f"[last_two_ca_positions] Enter chain='{getattr(chain,'name',None)}' with {len(chain)} residues")
    pos = []
    for res in chain:
        for atom in res:
            if atom.name == "CA":
                pos.append(atom.pos)
                logging.debug(f"[last_two_ca_positions]   found CA at resnum={res.seqid.num} pos=({atom.pos.x:.3f},{atom.pos.y:.3f},{atom.pos.z:.3f})")
                break
    if len(pos) >= 2:
        logging.debug(f"[last_two_ca_positions] Returning last/prev CA positions (nCA={len(pos)})")
    elif pos:
        logging.debug(f"[last_two_ca_positions] Only one CA found; prev=None")
    else:
        logging.debug(f"[last_two_ca_positions] No CA found; returning (None,None)")
    return (pos[-1], pos[-2]) if len(pos) >= 2 else (pos[-1] if pos else None, None)

def unit_vec(v: gemmi.Vec3) -> gemmi.Vec3:
    logging.debug(f"[unit_vec] Input v=({v.x:.3f},{v.y:.3f},{v.z:.3f}), |v|={v.length():.6f}")
    n = v.length()
    out = gemmi.Vec3(1.0, 0.0, 0.0) if n < 1e-6 else v / n
    logging.debug(f"[unit_vec] Output =({out.x:.3f},{out.y:.3f},{out.z:.3f})")
    return out

def translated(pos: gemmi.Position, direction: gemmi.Vec3, dist: float) -> gemmi.Position:
    logging.debug(f"[translated] pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) dir=({direction.x:.3f},{direction.y:.3f},{direction.z:.3f}) dist={dist}")
    out = gemmi.Position(pos.x + direction.x * dist,
                            pos.y + direction.y * dist,
                            pos.z + direction.z * dist)
    logging.debug(f"[translated] -> ({out.x:.3f},{out.y:.3f},{out.z:.3f})")
    return out

def append_poly_ala_stub(chain: gemmi.Chain, n_res: int, spacing: float = 3.8):
    logging.debug(f"[append_poly_ala_stub] chain='{getattr(chain,'name',None)}' n_res={n_res} spacing={spacing}")
    if n_res <= 0:
        logging.debug("[append_poly_ala_stub] n_res <= 0, nothing to append")
        return
    last, prev = last_two_ca_positions(chain)
    if last is None:
        logging.debug("[append_poly_ala_stub] No anchor CA found; starting at origin along +X")
        last = gemmi.Position(0.0, 0.0, 0.0)
        direction = gemmi.Vec3(1.0, 0.0, 0.0)
        start_num = 0
    else:
        direction = unit_vec(last - prev) if prev is not None else gemmi.Vec3(1.0, 0.0, 0.0)
        start_num = chain[-1].seqid.num if len(chain) else 0
        logging.debug(f"[append_poly_ala_stub] Anchor at ({last.x:.3f},{last.y:.3f},{last.z:.3f}); start_num={start_num}; dir=({direction.x:.3f},{direction.y:.3f},{direction.z:.3f})")
    for i in range(1, n_res + 1):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid.num = start_num + i
        for name in {"CA","N","C","O"}:
            ca = gemmi.Atom()
            ca.name = name
            element = "C" if name == "CA" else name
            ca.element = gemmi.Element(element)
            ca.occ = 1.0
            ca.b_iso = 20.0
            ca.pos = translated(last, direction, spacing * i)
            res.add_atom(ca)
        chain.add_residue(res)
        logging.debug(f"[append_poly_ala_stub]   added ALA CA stub resnum={res.seqid.num} at ({ca.pos.x:.3f},{ca.pos.y:.3f},{ca.pos.z:.3f})")
    logging.debug(f"[append_poly_ala_stub] Done; chain now has {len(chain)} residues")

def _create_AF_template_structure(input_structure: gemmi.Structure,template_residues: str,designed_residues: str,linker_length: int) -> gemmi.Structure:
    logging.debug("[_create_AF_template_structure] START")
    logging.debug(f"  template_residues='{template_residues}'")
    logging.debug(f"  designed_residues='{designed_residues}'")
    logging.debug(f"  linker_length={linker_length}")
    
    # Prepare output structure
    output_structure = gemmi.Structure()
    chains = []

    designed_residues = designed_residues.split()
    logging.debug(f"  designed_residues tokens={designed_residues}")

    for model in input_structure:
        logging.debug(f"  Processing input model num={getattr(model,'num',None)} with {len(model)} chains")
        new_model = gemmi.Model(model.num)
        chain_idx = 0
        rf_chains = template_residues.split()
        logging.debug(f"  rf_chains tokens={rf_chains}")
        if len(rf_chains) != len(designed_residues): logging.debug(f"  ERROR: rf_chains({len(rf_chains)}) != designed_residues({len(designed_residues)})"); raise ValueError(f"Length of AF chain contig not equal to designed residue contig")
        for rf_idx, rf_chain in enumerate(rf_chains):
            logging.debug(f"    rf_idx={rf_idx} rf_chain='{rf_chain}'")
            new_chains = rf_chain.split('/BREAK/')
            logging.debug(f"      split into {len(new_chains)} output chain(s): {new_chains}")
            for boltz_chain in new_chains:
                chain_id = chr(ord('A') + chain_idx)
                logging.debug(f"      -> starting output chain_id='{chain_id}' from spec='{boltz_chain}'")
                new_chain = gemmi.Chain(chain_id)
                has_template_residues = False
                segments = boltz_chain.split('/')
                logging.debug(f"         segments={segments}")
                for segment in segments:
                    logging.debug(f"         segment='{segment}'")
                    if segment == 'DESIGN':
                        length = parse_contig(designed_residues[rf_idx],linker_length)
                        logging.debug(f"           DESIGN segment -> length={length}")
                        append_poly_ala_stub(new_chain,length)
                        has_template_residues = True
                        continue
                    m = re.match(r'([A-Za-z])(\d+)-(\d+)',segment)
                    if not m:
                        logging.debug("           segment did not match template regex; skipping")
                        continue
                    old_id = m.group(1)
                    first_res = int(m.group(2))
                    last_res = int(m.group(3))
                    logging.debug(f"           TEMPLATE segment -> old_id='{old_id}' range={first_res}-{last_res}")
                    for chain in model:
                        logging.debug(f"             scanning model chain name='{chain.name}' (len={len(chain)})")
                        if old_id != chain.name:
                            continue
                        logging.debug(f"             MATCH chain '{chain.name}' -> collecting residues")
                        added_count_before = len(new_chain)
                        for res in chain:
                            num = res.seqid.num
                            if num is not None and first_res <= num <= last_res:
                                new_res = res.clone()
                                new_res.subchain = chain_id
                                new_res.entity_id = chain_id 
                                new_chain.add_residue(new_res)
                                has_template_residues = True
                        added_count_after = len(new_chain)
                        logging.debug(f"             added {added_count_after - added_count_before} residues from chain '{chain.name}'")
                logging.debug(f"         has_template_residues={has_template_residues}; new_chain_len={len(new_chain)}")
                if has_template_residues:
                    chains.append(chain_id)
                    chain_idx += 1
                    new_model.add_chain(new_chain)
                    logging.debug(f"      ++ added output chain '{chain_id}' (len={len(new_chain)}). Next chain_idx={chain_idx}")
                else:
                    logging.debug(f"      xx skipping empty output chain '{chain_id}'")
        output_structure.add_model(new_model)
        logging.debug(f"  Added new_model with {len(new_model)} chains to output_structure")

    # Renumber residues 
    for model in output_structure:
        logging.debug(f"  Renumbering model (has {len(model)} chains)")
        for chain in model:
            logging.debug(f"    Renumbering chain '{chain.name}' with {len(chain)} residues")
            resnum = 1 
            for res in chain:
                res.seqid.num = resnum
                resnum+=1
            logging.debug(f"    -> chain '{chain.name}' renumbered 1..{len(chain)}")

    # Finalize structure setup
    logging.debug("  Finalizing entities/subchains")
    output_structure.add_entity_types(overwrite=True)
    output_structure.assign_subchains()
    output_structure.ensure_entities()
    output_structure.deduplicate_entities()
    logging.debug("[_create_AF_template_structure] DONE")
    return output_structure, chains


def _make_AF_fixed_residue_file(template_reference_pdb,template_residues,designed_residues,output_folder_path,tag,linker_length):
    # Make file path
    folder_p = pathlib.Path(output_folder_path)
    folder_p.mkdir(parents=True, exist_ok=True)
    file_p = pathlib.Path(os.path.join(output_folder_path,tag))
    if file_p.exists() and file_p.is_file():
        # Deduce polymer (no DNA,RNA,Ligans,...) chains from file
        chains = []
        structure = gemmi.read_structure(str(file_p))
        structure.setup_entities()
        for model in structure:
            for ch in model:
                if any(res.entity_type == gemmi.EntityType.Polymer for res in ch):
                    chains.append(ch.name)
        return file_p,chains
    
    # Find input file for reference to template
    template_p = pathlib.Path(template_reference_pdb)
    if not (template_p.exists() and template_p.is_file()):
        raise ValueError(f"Could not find reference PDB for template in path: {template_reference_pdb}")
    
    # Read the input pdb into a Gemmi Structure
    input_structure = gemmi.read_structure(template_reference_pdb)
    input_structure.setup_entities()

    # Create template structure
    output_structure,chains = _create_AF_template_structure(input_structure,template_residues,designed_residues,linker_length)

    # Save template structure to PDB
    output_structure.write_pdb(str(file_p))

    return file_p,chains

def _build_AF_template(args):
    output_folder_path = args.af_template_path
    template_reference_pdb = args.rf_input_pdb
    designed_residues = args.af_template_designed
    template_residues = args.af_template_contig
    if args.run_boltz_for_length is not None:
        # Create a template PDB file for only one length
        tag = "length_"+str(args.run_boltz_for_length)+"_template.pdb"
        path,chains = _make_AF_fixed_residue_file(template_reference_pdb,template_residues,designed_residues,output_folder_path,tag,args.run_boltz_for_length)
        logging.debug(f"Built template file with chains {chains} to path {path}")
    else:
         # Create a template PDB file for all lengths
        for length in range(args.min_length,args.max_length+1):
            tag = "length_"+str(length)+"_template.pdb"
            path,chains = _make_AF_fixed_residue_file(template_reference_pdb,template_residues,designed_residues,output_folder_path,tag,length)
            logging.debug(f"Built template file with chains {chains} to path {path}")

    return path,chains

def _make_AF_args(args):
    template_path,template_chains = _build_AF_template(args)
    yaml_path = args.b_yaml
    scriptArgs = [f'--logfile={args.logfile}',
    f'--af_output_path={args.af_output_path}',
    f'--yaml_path={yaml_path}',
    f'--colab_data_dir={args.af_params_dir}',
    f'--template_pdb_path={template_path}',
    f'--template_chains={",".join(template_chains)}',
    f'--af_recycles_amount={args.af_recycles_amount}',
    ]
    if args.af_initial_guess:
        scriptArgs.append('--af_initial_guess')
    if args.af_use_initial_atom_pos:
        scriptArgs.append('--af_use_initial_atom_pos')
    if args.run_boltz_for_length is not None:
         scriptArgs.append(f'--run_for_length={args.run_boltz_for_length}')
    return scriptArgs

def run_af(args: arg.Namespace):
    # Run AF in env args.af_env for length args.run_boltz_for_length
    scriptEnv = args.af_env
    
    # Find the directory to the alphafold script
    this_file = pathlib.Path(__file__).resolve()
    src_dir = this_file.parent.parent
    scriptPath = str(src_dir / "alphafold_script.py")

    scriptArgs = _make_AF_args(args)

    result = run_script_in_env(scriptEnv,scriptPath,scriptArgs,return_result=True)
    logging.debug(f"AlphaFold script result: {result}")