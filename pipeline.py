#!/usr/bin/env python3
import logging
import shlex
import sys
import argparse as arg
import os
from utils import validate_input,write_sbatch,submit_sbatch,validate_dir
from rfdiffusion_pipeline import run_rfdiffusion
from proteinmpnn_pipeline import run_proteinMPNN
from boltz_pipeline import run_boltz
from analysis import run_analysis
from alphafold_pipeline import run_af
#-------------------------------------------- Helpers  --------------------------------------------------#
def _internal_flags(raw_argv: list[str],strip=True) -> list[str]:
    """
    Return a copy of raw_argv with any “internal” flags removed if strip=True.
    Supports ONLY "--flag" for boolean store_true flags
    """
    # Arguments that we don't want to propagate into child processes
    to_drop = {"--run_array", "--only_rfdiff", "--skip_rfdiff","--only_boltz","--skip_boltz","--only_pmpnn","--skip_pmpnn","--only_af","--skip_af"}

    cleaned: list[str] = []
    it = iter(raw_argv)

    for tok in it:
        key = tok.split("=", 1)[0]      # '--foo=bar' → '--foo'
        if strip: 
            if key in to_drop:
                continue                     # skip this flag token
        cleaned.append(tok)

    return cleaned

def launch_master(args, script_path):

    # Calculate exclusive difference 
    length = args.max_length-args.min_length
    # re‑quote for bash line‐continuation
    shared_block = " \\\n    ".join(shlex.quote(x) for x in _internal_flags(sys.argv[1:],True))

    logging.debug(f"Constructing rfdiffusion command with flags and {shared_block}")

    rf_array = f"""#!/bin/bash
#SBATCH --job-name=rf_array
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBatch --qos:gpu
#SBATCH --mem=80g
#SBATCH --time=30:00:00
#SBATCH --output={args.filtered_designs_folder}/logs/pipeline_lasv.out 
#SBATCH --error={args.filtered_designs_folder}/logs/pipeline_lasv.err
#SBATCH --array=0-{length}

python {script_path} \\
    --no_validate_inputs \\
    --no_run_array \\
    --only_rfdiff \\
    {shared_block}
"""
    logging.warning("Submitting arrays for RFdiffusion")
    rf_s_path = os.path.join(args.filtered_designs_folder,"rf_array.sh")
    write_sbatch(rf_s_path, rf_array)
    rf_job_id = submit_sbatch(rf_s_path, parsable=True)
    logging.warning(f"RF array job submitted: {rf_job_id}")

    # Build ProteinMPNN job, dependent on the previous RFDiffusion array
    logging.warning(f"Constructing pmpnn command with flags and {shared_block}")
    pmpnn = f"""#!/bin/bash
#SBATCH --job-name=pmpnn
#SBATCH --dependency=afterok:{rf_job_id}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBatch --qos:gpu
#SBATCH --mem=80g
#SBATCH --time=16:00:00
#SBATCH --array=0-{length}
#SBATCH --output={args.filtered_designs_folder}/logs/pipeline_lasv.out 
#SBATCH --error={args.filtered_designs_folder}/logs/pipeline_lasv.err

python {script_path} \\
    --no_validate_inputs \\
    --no_run_array \\
    --skip_rfdiff \\
    --only_pmpnn \\
    {shared_block}
"""
    
    logging.warning("Continuing pipeline with ProteinMPNN after RFdiffusion")
    pmnn_s_path = os.path.join(args.filtered_designs_folder,"pmpnn.sh")
    write_sbatch(pmnn_s_path, pmpnn)
    pmpnn_job_id = submit_sbatch(pmnn_s_path,parsable=True)
    logging.warning(f"Downstream job submitted: {pmpnn_job_id}")

   # Build Boltz job, dependent on the previous ProteinMPNN array (will only run after)
    logging.warning(f"Constructing Boltz command with flags and {shared_block}")
    boltz_sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --dependency=afterok:{pmpnn_job_id}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBatch --qos:gpu
#SBATCH --mem=80g
#SBATCH --time=16:00:00
#SBATCH --output={args.filtered_designs_folder}/logs/pipeline_lasv.out 
#SBATCH --error={args.filtered_designs_folder}/logs/pipeline_lasv.err
#SBATCH --array=0-{length}

python {script_path} \\
    --no_validate_inputs \\
    --no_run_array \\
    --skip_rfdiff \\
    --skip_pmpnn \\
    --only_boltz \\
    {shared_block}
"""
    
    logging.warning("Continuing pipeline with Boltz after ProteinMPNN")
    boltz_s_path = os.path.join(args.filtered_designs_folder,"boltz.sh")
    write_sbatch(boltz_s_path, boltz_sbatch_script)
    boltz_job_id = submit_sbatch(boltz_s_path,parsable=True)
    logging.warning(f"Downstream job submitted: {boltz_job_id}")

  # Build AF job, dependent on the previous Boltz array (will only run after)
    logging.warning(f"Constructing AlphaFold (colab) command with flags and {shared_block}")
    af_sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=colabAF
#SBATCH --dependency=afterok:{boltz_job_id}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBatch --qos:gpu
#SBATCH --mem=80g
#SBATCH --time=16:00:00
#SBATCH --output={args.filtered_designs_folder}/logs/pipeline_lasv.out 
#SBATCH --error={args.filtered_designs_folder}/logs/pipeline_lasv.err
#SBATCH --array=0-{length}

python {script_path} \\
    --no_validate_inputs \\
    --no_run_array \\
    --skip_rfdiff \\
    --skip_pmpnn \\
    --skip_boltz \\
    --only_af \\
    {shared_block}
"""
    
    logging.warning("Continuing pipeline with AlphaFold after Boltz")
    af_s_path = os.path.join(args.filtered_designs_folder,"alphafold.sh")
    write_sbatch(af_s_path, af_sbatch_script)
    af_job_id = submit_sbatch(af_s_path,parsable=True)
    logging.warning(f"Downstream job submitted: {af_job_id}")

    #Build final result analysis job, dependent on the previous Boltz array (will only run after)
    logging.warning(f"Constructing Analysis command with flags and {shared_block}")
    # add : #SBATCH --dependency=afterok:{af_job_id}
    analysis_sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=post
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2
#SBatch --qos:gpu
#SBATCH --mem=80g
#SBATCH --time=10:00:00
#SBATCH --output={args.filtered_designs_folder}/logs/pipeline_lasv.out 
#SBATCH --error={args.filtered_designs_folder}/logs/pipeline_lasv.err

python {script_path} \\
    --no_validate_inputs \\
    --no_run_array \\
    --skip_rfdiff \\
    --skip_pmpnn \\
    --skip_boltz \\
    --skip_af \\
    {shared_block}
"""
    
    logging.warning("Continuing pipeline with result analysis after Boltz")
    analysis_s_path = os.path.join(args.filtered_designs_folder,"analysis.sh")
    write_sbatch(analysis_s_path, analysis_sbatch_script)
    result_job_id = submit_sbatch(analysis_s_path,parsable=True)
    logging.warning(f"Downstream job submitted: {result_job_id}")

#-------------------------------------------- Main --------------------------------------------------#

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
    rf_group.add_argument('--rf_min_linker_length', dest='min_length', type=int, required=True,
                          help='Minimum (positive integer) length for the designed linker')
    rf_group.add_argument('--rf_max_linker_length', dest='max_length', type=int, required=True,
                          help='Maximum (positive integer) length for the designed linker') 
    rf_group.add_argument('--rf_num_designs_per_linker', dest='rf_num_designs', type=int,
                         required=True, help='Number (positive integer) of designs to generate')
    rf_group.add_argument('--rf_env', dest='rf_env' , type=str, default='SE3nv',
                         help='Conda environment for RFdiffusion') 
    rf_group.add_argument('--rf_modified_files_path', dest='rf_fixpath', type=str, required=True,
                          help='Path where to create the modified pdb files generated from RFdiffusion (adapts chains)')     
    rf_group.add_argument('--rf_inpaint_seq',dest='rf_inpaint_seq',type=str,help='specify amino acids whose sequence should be hidden')               
    rf_group.add_argument('--rf_symmetry',dest='rf_symmetry',type=str)
    rf_group.add_argument('--rf_compact',dest='rf_compact',action="store_true")

    # ProteinMPNN arguments
    pm_group = parser.add_argument_group('ProteinMPNN parameters')
    pm_group.add_argument('--pm_jsonl_path', dest='pm_jsonl', type=str, required=True)
    pm_group.add_argument('--pm_proteinmpnn_path',dest='pm_fold', type=str, required=True, 
                          help='Path to ProteinMPNN folder')
    pm_group.add_argument('--pm_output_path',dest='pm_output', type=str, required=True,
                          help='Path where to generate the ProteinMPNN outputs')
    pm_group.add_argument('--pm_seq_per_target', dest='pm_seq_per_target', type=int, required=True,
                            help='Number of sequences per target')
    pm_group.add_argument('--pm_sampling_temp', dest='pm_sampling_temp', type=float, required=True)
    pm_group.add_argument("--use_soluble_model", action="store_true", dest='pm_sol', default=False, 
                            help="Flag to load ProteinMPNN weights trained on soluble proteins only.")
    pm_group.add_argument('--pm_seed', dest='pm_seed', required=True)
    pm_group.add_argument('--pm_batch_size', dest='pm_batch_size', type=int, required=True)
    pm_group.add_argument('--pm_env', dest='pm_env', type=str, required=True,
                          help='Environment to run ProteinMPNN')
    pm_group.add_argument('--pm_redesign_designed_res_seq', action="store_true",dest='pm_redesign_designed_res_seq',default=False,)
    pm_group.add_argument('--run_pmpnn_for_length',dest='run_pmpnn_for_length',type=int,help='Only run ProteinMPNN for one length')

    # Boltz arguments 
    b_group = parser.add_argument_group('Boltz parameters')
    b_group.add_argument('--b_out_dir', dest='b_out', type=str, required=True,
                         help='Output directory for Boltz predictions.')
    b_group.add_argument('--b_yaml_dir', dest='b_yaml', type=str, required=True,
                          help='Directory to create the yaml files necessary for Boltz')
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
    af_group.add_argument('--af_output_path',dest='af_output_path',required=True)
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
    f_group.add_argument('--pykvfinder_env',dest='pykvfinder_env', type=str, required=True)
    f_group.add_argument('--af_params_dir',dest='af_params_dir',type=str,required=True)
    f_group.add_argument('--metric_residues',dest='metric_residues',type=str,required=True)
    f_group.add_argument('--stats_read_file_only',dest='stats_read_file_only',action="store_true")

    # Parse
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        filename=args.logfile,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.warning("Starting the protein design pipeline.")

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logging.getLogger().addHandler(console)

    try:
        logging.warning(f"Called with {_internal_flags(sys.argv[1:],False)}")
        if not args.no_validate_inputs:
            #Validate inputs
            validate_input(args)

        if args.run_array:
             logging.warning("Preparing master job")
             launch_master(args,os.path.abspath(sys.argv[0]))
             logging.warning("Master job exiting after scheduling children")
             sys.exit(0)

        if args.only_rfdiff:
            # if this is an array task, override min/max to the single array index
            if "SLURM_ARRAY_TASK_ID" in os.environ:
                min_length = args.min_length
                max_length = args.max_length
                idx = min_length+int(os.environ["SLURM_ARRAY_TASK_ID"])
                args.min_length = idx
                args.max_length = idx

                if (not ("SLURM_ARRAY_TASK_MAX" in os.environ)) or (not ("SLURM_ARRAY_TASK_MIN" in os.environ)):
                    raise ValueError("TASK_MAX_MIN_ID'S not found in environ")

                max_task = int(os.environ["SLURM_ARRAY_TASK_MAX"])

                if (min_length+max_task) != max_length:
                    raise ValueError("Array task amount doesn't correspond to lengths")

        if not args.skip_rfdiff:
            # Run RFdiffusion 
            logging.warning("Running RFdiffusion script...")
            run_rfdiffusion(args)
            logging.warning("RFdiffusion script executed successfully.")
        else:
            logging.warning("Skipping RFdiffusion step")  

        if not args.only_rfdiff: 

            if not args.skip_pmpnn:
                # if this is an array task, override boltz residue length to the array index
                if "SLURM_ARRAY_TASK_ID" in os.environ:
                    # ProteinMPNN will only run on folder named length_{args.run_pmpnn_for_length}
                    args.run_pmpnn_for_length = args.min_length+int(os.environ["SLURM_ARRAY_TASK_ID"])

                # Run ProteinMPNN 
                logging.warning("Running ProteinMPNN script...")
                run_proteinMPNN(args)
                logging.warning("ProteinMPNN script executed successfully.")

            if not args.only_pmpnn:
                if not args.skip_boltz:
                    # if this is an array task, override boltz residue length to the array index
                    if "SLURM_ARRAY_TASK_ID" in os.environ:
                        # Boltz will only run on folder named length_{args.run_boltz_for_length}
                        args.run_boltz_for_length = args.min_length+int(os.environ["SLURM_ARRAY_TASK_ID"])

                    # Run Boltz 
                    logging.warning("Running Boltz script...")
                    run_boltz(args)
                    logging.warning("Boltz script executed successfully.")

                if not args.only_boltz:
                    if not args.skip_af:
                        # if this is an array task, override boltz residue length to the array index
                        if "SLURM_ARRAY_TASK_ID" in os.environ:
                            # AlphaFold will only run on named linker lengths args.run_boltz_for_length
                            # Note: The run_boltz_for_length is used both for Boltz and AlphaFold
                            args.run_boltz_for_length = args.min_length+int(os.environ["SLURM_ARRAY_TASK_ID"])

                        # Run AlphaFold
                        logging.warning("Running AlphaFold script...")
                        run_af(args)
                        logging.warning("AlphaFold script executed successfully.")

                    if not args.only_af:    
                        # Compare Boltz Predictions with original structure, calculate selection metrics, select best candidates and plot
                        logging.warning("Analyzing Boltz outputs...")
                        run_analysis(args)
                        logging.warning("Analysis executed successfully.")
                        logging.warning("Pipeline script executed successfully.")
                    else:
                        logging.warning("Only AlphaFold was executed; exiting.")
                else:
                    logging.warning("Only Boltz was executed; exiting.")
            else:
                logging.warning("Only ProteinMPNN was executed; exiting.")
        else:
            logging.warning("Only RFDiffusion was executed; exiting.")

    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Error executing pipeline script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

