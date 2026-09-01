#!/usr/bin/env python3
from argparse import Namespace
import argparse as arg
import bisect
import logging
import math
import sys
from typing import Dict, List, Tuple
import pyKVFinder as kv # type: ignore
from pipeline.spike_design.utils import _get_residue_strings,basic_format_residues,build_lut_from_contig

def _get_fixed_residue_set(args: Namespace, linker_length: int) -> set[Tuple[str,int]]:
    residue_set = set()
    reference_contig = args.boltz_chain_break_contig
    chains_full = reference_contig.split()
    lut = build_lut_from_contig(linker_length,chains_full)
    residues_segments, _ = _get_residue_strings(False,args.fixed_residues,reference_contig,linker_length,lut,basic_format_residues,cmd=False,use_link_and_des=False)
    logging.debug("Cavity analysis found fixed residue segments: %s", residues_segments)

    for residue_segment in residues_segments:
        chain_id = residue_segment[0]
        first_residue = residue_segment[1]
        last_residue = residue_segment[2]
        for resnum in range(first_residue,last_residue+1):
            residue_set.add((chain_id,resnum))
    return residue_set 

def _get_cavities(args: Namespace ,residueDict: Dict[str, List[List[str]]],linker_length: int) -> Tuple[set[str], set[str]]:
    fixed_residues = _get_fixed_residue_set(args, linker_length)
    fixed_cavities = set()
    designed_cavities = set()

    for cavity, res_list in residueDict.items():
        for res_num, res_id, _ in res_list:
            res_num = int(res_num)
            if (res_id,res_num) in fixed_residues:   
                fixed_cavities.add(cavity)
            else:
                designed_cavities.add(cavity)

    return fixed_cavities, designed_cavities

def _get_cavity_metrics(cavities: set[str], volumesDict: Dict[str, float], bin_amount: int) -> Tuple[float,float,float]:

    if not len(cavities):
        return 0.0, 0.0, 0
    
    total_volume = 0
    number_of_cavities = 0
    floatmax = sys.float_info.max 
    max_cavity_volume = -floatmax
    min_cavity_volume = floatmax

    for cavity in cavities:
        volume = volumesDict[cavity]
        total_volume += volume
        number_of_cavities += 1
        if volume < min_cavity_volume: min_cavity_volume = volume
        if volume > max_cavity_volume: max_cavity_volume = volume
    
    avg_volume = total_volume/number_of_cavities 

    min_edge = min_cavity_volume
    max_edge = max_cavity_volume
    length_edges    = max_edge - min_edge
    step     = length_edges / bin_amount if bin_amount else 0 

    # build all 11 edges with an exact formula
    edges = [min_edge + i*step for i in range(bin_amount+1)] if step else []
    bins = [(edges[i], edges[i+1]) for i in range(bin_amount)] if step else []

    if step and not math.isclose(edges[-1], max_edge, rel_tol=1e-9, abs_tol=1e-12): 
        logging.warning(f"Last edge in cavity-amount-per-volume bins: {edges[-1]} not equal to maximum length {max_edge}")
    
    return total_volume,avg_volume,number_of_cavities

def calculate_cavities(args,pdb_path, linker_length)-> Tuple[
                                                                    float,  # fixed_total_volume
                                                                    float,  # fixed_avg_volume
                                                                    float,    # fixed_count
                                                                    float,  # designed_total_volume
                                                                    float,  # designed_avg_volume
                                                                    float,    # designed_count
                                                                ]:
    # Calculate cavity metrics
    results = kv.run_workflow(pdb_path)

    if results is None:
        logging.warning(f"No cavities found for {pdb_path}")
        return 0,0,0,0,0,0
    
    volumesDict = results.volume # Dict[str, float] - A dictionary with volume of each detected cavity.
    residueDict = results.residues # Dict[str, List[List[str]]] - A dictionary with a list of interface residues for each detected cavity.

    # Find the ID's of the cavities of interest. Note that 
    fixed_cavities, designed_cavities = _get_cavities(args,residueDict,linker_length)
    
    # Calculate metrics : 
    fixed_total_volume,  fixed_avg_volume,  fixed_count  = _get_cavity_metrics(fixed_cavities,volumesDict,10)
    designed_total_volume, designed_avg_volume, designed_count = _get_cavity_metrics(designed_cavities,volumesDict,10)
    return fixed_total_volume,  fixed_avg_volume,  fixed_count, designed_total_volume, designed_avg_volume, designed_count

def main():

    # Parse command line arguments
    parser = arg.ArgumentParser(
        prog = "Protein Design Pipeline",
        description = "Protein design pathline for backbone design through RFdiffusion, " \
        "sequence generation through ProteinMPNN, and testing through Boltz.",
    )

    # Logging arguments: 
    parser.add_argument('-l', '--logfile', type=str, help='Path to the log file', default='pipeline.log')

    # Job managment 
    parser.add_argument("--no_validate_inputs",dest="no_validate_inputs",action="store_true")
    parser.add_argument("--only_rfdiff",dest="only_rfdiff", action="store_true", help="just run RFdiffusion and exit")
    parser.add_argument("--skip_rfdiff",dest="skip_rfdiff", action="store_true",help="run everything _except_ RFdiffusion")
    parser.add_argument("--run_array",dest="run_array",action="store_true")
    parser.add_argument("--no_run_array",dest="no_run_array",action="store_true")
    parser.add_argument("--only_boltz",dest="only_boltz",action="store_true")
    parser.add_argument("--skip_boltz",dest="skip_boltz",action="store_true")
    parser.add_argument("--only_pmpnn",dest="only_pmpnn",action="store_true")
    parser.add_argument("--skip_pmpnn",dest="skip_pmpnn",action="store_true")
    parser.add_argument("--only_af",dest="only_af",action="store_true")
    parser.add_argument("--skip_af",dest="skip_af",action="store_true")
    
    # RFdiffusion arguments
    rf_group = parser.add_argument_group('RFdiffusion parameters')
    rf_group.add_argument('--rf_script_path', dest='rf_script_path', type=str, required=True, 
                         help='Path to the RFdiffusion script')
    rf_group.add_argument('--rf_output_prefix', dest='rf_output_prefix',
                         required=True, help='Output prefix for designs')
    rf_group.add_argument('--rf_input_pdb', dest='rf_input_pdb',
                         required=True, help='Input PDB file')
    rf_group.add_argument('--rf_contig', dest='rf_contig', type=str,
                         required=True, help='Contig mapping specifications')
    rf_group.add_argument('--min_length', dest='min_length', type=int, required=True,
                          help='Minimum (positive integer) length for the designed linker')
    rf_group.add_argument('--max_length', dest='max_length', type=int, required=True,
                          help='Maximum (positive integer) length for the designed linker') 
    rf_group.add_argument('--rf_num_designs', dest='rf_num_designs', type=int,
                         required=True, help='Number (positive integer) of designs to generate')
    rf_group.add_argument('--rf_env', dest='rf_env' , type=str, default='SE3nv',
                         help='Conda environment for RFdiffusion') 
    rf_group.add_argument('--rf_fixpath', dest='rf_fixpath', type=str, required=True,
                          help='Path where to create the modified pdb files generated from RFdiffusion (adapts chains)')     
    rf_group.add_argument('--rf_inpaint_seq',dest='rf_inpaint_seq',type=str,help='specify amino acids whose sequence should be hidden')               
    rf_group.add_argument('--rf_symmetry',dest='rf_symmetry',type=str)
    rf_group.add_argument('--rf_compact',dest='rf_compact',action="store_true")
    
    # ProteinMPNN arguments
    pm_group = parser.add_argument_group('ProteinMPNN parameters')
    pm_group.add_argument('--pm_jsonl_path', dest='pm_jsonl', type=str, required=True)
    pm_group.add_argument('--pm_fold',dest='pm_fold', type=str, required=True, 
                          help='Path to ProteinMPNN folder')
    pm_group.add_argument('--pm_output',dest='pm_output', type=str, required=True,
                          help='Path where to generate the ProteinMPNN outputs')
    pm_group.add_argument('--pm_seq_per_target', dest='pm_seq_per_target', type=int, required=True,
                            help='Number of sequences per target')
    pm_group.add_argument('--pm_sampling_temp', dest='pm_sampling_temp', type=float, required=True)
    pm_group.add_argument("--pm_sol", action="store_true", dest='pm_sol', default=False, 
                            help="Flag to load ProteinMPNN weights trained on soluble proteins only.")
    pm_group.add_argument('--pm_seed', dest='pm_seed', required=True)
    pm_group.add_argument('--pm_batch_size', dest='pm_batch_size', type=int, required=True)
    pm_group.add_argument('--pm_env', dest='pm_env', type=str, required=True,
                          help='Environment to run ProteinMPNN')
    pm_group.add_argument('--pm_redesign_designed_res_seq', action="store_true",dest='pm_redesign_designed_res_seq',default=False,)
    pm_group.add_argument('--run_pmpnn_for_length',dest='run_pmpnn_for_length',type=int,help='Only run ProteinMPNN for one length')
 

    # Boltz arguments 
    b_group = parser.add_argument_group('Boltz parameters')
    b_group.add_argument('--b_out', dest='b_out', type=str, required=True,
                         help='Output directory for Boltz predictions.')
    b_group.add_argument('--b_yaml', dest='b_yaml', type=str, required=True,
                          help='Directory to create the fasta files necessary for Boltz')
    b_group.add_argument('--b_designs_from_pm',dest='b_designs_from_pm', type=int, required=True,
                         help='For each RFdiffusion design, amount of sequences from ProteinMPNN to run Boltz on.')
    b_group.add_argument('--b_recycling_steps', dest='b_recycling_steps', type=int,required=True,
                         help='Number of recycling steps for Boltz.')
    b_group.add_argument('--b_sampling_steps', dest='b_sampling_steps', type=int, required=True,
                         help='The number of sampling steps to use for prediction for Boltz.')
    b_group.add_argument('--b_diffusion_samples',dest='b_diffusion_samples', type=int, required=True,
                         help='The number of diffusion samples to use for prediction for Boltz.')
    b_group.add_argument('--b_step_scale',dest='b_step_scale', type=float, required=True,
                         help='The step size is related to the temperature at which the diffusion process samples the distribution. ' \
                         'The lower the higher the diversity among samples (recommended between 1 and 2), for Boltz.')
    b_group.add_argument('--b_output_format', dest='b_output_format', required=True, 
                         help='	The output format to use for the predictions, for Boltz')
    b_group.add_argument("--b_use_msa_server", action="store_true", dest='b_use_msa_server', default=False, 
                            help="Whether to use the msa server to generate msa's for Boltz.")
    b_group.add_argument('--b_env',dest='b_env',type=str, required=True)
    b_group.add_argument('--b_devices',dest='b_devices',required=True,type=int)
    b_group.add_argument('--max_parallel_samples',dest='max_parallel_samples',required=True,type=int)
    b_group.add_argument('--b_use_fixed_residues_template',dest='b_use_fixed_residues_template',action="store_true")
    b_group.add_argument('--b_fixed_residue_cif_folder',dest='b_fixed_residue_cif_folder',type=str)
    b_group.add_argument('--run_boltz_for_length',dest='run_boltz_for_length',type=int,help='Only run Boltz for one length')
    b_group.add_argument('--same_chain_on_design',dest='same_chain_on_design',required=True,type=str)
    b_group.add_argument('--b_force_template',dest='b_force_template',action="store_true")
    b_group.add_argument('--fixed_residues',dest='fixed_residues',type=str,required=True)
    b_group.add_argument('--b_percent_of_template',dest='b_percent_of_template',type=float)
    b_group.add_argument('--template_residues',dest='template_residues',type=str,required=False)
    b_group.add_argument('--one_temp_per_class',dest='one_temp_per_class',action="store_true")
    b_group.add_argument('--boltz_chain_break_contig',dest='boltz_chain_break_contig',type=str,required=True)
    b_group.add_argument('--template_structure',dest='template_structure',required=True,type=str)

    # AlphaFold aruments
    af_group = parser.add_argument_group('AlphaFold parameters')
    b_group.add_argument('--af_output_path',dest='af_output_path',required=True)
    af_group.add_argument('--af_env',dest='af_env',required=True)
    af_group.add_argument('--af_initial_guess',dest='af_initial_guess',action="store_true")
    af_group.add_argument('--af_recycles_amount',dest='af_recycles_amount',type=int,required=True)
    af_group.add_argument('--af_use_initial_atom_pos',dest='af_use_initial_atom_pos',action="store_true")
    af_group.add_argument('--af_template_contig',dest='af_template_contig',required=True,type=str)
    af_group.add_argument('--af_template_designed',dest='af_template_designed',required=True,type=str)
    af_group.add_argument('--af_template_path',dest='af_template_path',required=True,type=str)

    # Filtering arguments 
    f_group = parser.add_argument_group('Filtering parameters')
    f_group.add_argument('--filtered_designs_folder',dest='filtered_designs_folder',required=True)
    f_group.add_argument('--rmsd_alignment_residues',dest='rmsd_alignment_residues',required=True)
    f_group.add_argument('--py_env',dest='py_env',required=True)
    f_group.add_argument('--pymol_alignment_cycles',dest='pymol_alignment_cycles',required=True,type=int)
    f_group.add_argument('--final_selection_amount',dest='final_selection_amount',required=True,type=int)
    f_group.add_argument('--designed_residues_contig',dest='designed_residues_contig',type=str,required=True)
    f_group.add_argument('--align_one_chain_only', dest='align_one_chain_only', action="store_true")
    f_group.add_argument('--align_structure',dest='align_structure',required=True,type=str,help='Structure against which to align in analysis')
    f_group.add_argument('--alignment_contig',dest='alignment_contig',required=True, type=str)
    f_group.add_argument('--rmsd_cutoff',dest='rmsd_cutoff',type=float,required=True)
    f_group.add_argument('--plddt_cutoff',dest='plddt_cutoff',type=float)
    f_group.add_argument('--pae_cutoff',dest='pae_cutoff',type=float)
    f_group.add_argument('--pde_cutoff',dest='pde_cutoff',type=float)
    f_group.add_argument('--pdb_path',dest='pdb_path',type=str,required=True)
    f_group.add_argument('--linker_length',dest='linker_length',type=int,required=True)
    f_group.add_argument('--pykvfinder_env',dest='pykvfinder_env', type=str, required=True)
    f_group.add_argument('--af_params_dir',dest='af_params_dir',type=str,required=True)
    f_group.add_argument('--metric_residues',dest='metric_residues',type=str,required=True)
    f_group.add_argument('--stats_read_file_only',dest='stats_read_file_only',action="store_true")
    f_group.add_argument('--glycosylate_best_designs',dest='glycosylate_best_designs',action="store_true")
    f_group.add_argument('--glycan_amount',dest='glycan_amount',default=1,type=int)
    f_group.add_argument('--glycosylated_design_amount',dest='glycosylated_design_amount',default=20,type=int)
    f_group.add_argument('--glycosylated_path',dest='glycosylated_path',type=str)
    # Parse
    args = parser.parse_args()

     # Set up logging
    logging.basicConfig(
        filename=args.logfile,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.debug(f"Starting the cavity calculation for file {args.pdb_path}.")

    result = calculate_cavities(args, args.pdb_path, args.linker_length)

    result_names = ["fixed_total_volume",  "fixed_avg_volume",  "fixed_count", "designed_total_volume", "designed_avg_volume", "designed_count"]
    output = ",".join(f"{name}={str(result_value)}" for name,result_value in zip(result_names,result)) 
    sys.stdout.write(output)

    logging.debug(f"Finished the cavity calculation for file {args.pdb_path}.")
    
if __name__ == "__main__":
    main()

