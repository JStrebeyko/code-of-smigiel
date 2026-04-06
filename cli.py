from utils.slurm import get_slum_script_content
from utils.time import get_time_string
from utils.consts import models, domains, all_dataset_names
import os
import subprocess


# CLI
from argparse import ArgumentParser

parser = ArgumentParser(
  prog='ŚMIGIEL GEN CLI',
  description='LLM-aided text generation helper'
)
parser.add_argument('command', default='help')
parser.add_argument('--name', default='my_batch')
parser.add_argument('--walltime', default='2:00:00')
parser.add_argument('--input_file', default=None) # used for generation
parser.add_argument('--output_file', default='./generation_input.csv') # used for prepartion of human data
parser.add_argument('--models', nargs='+', default=models)
parser.add_argument('--domains', nargs='+', default=domains)
parser.add_argument('--datasets', nargs='+', default=all_dataset_names)
parser.add_argument('--cap', type=int, default = 0)
parser.add_argument('--batched', default=1000, type=int)
parser.add_argument('--include_batches', nargs="+", default=[])
parser.add_argument('--mode', default='generate', type=str)
parser.add_argument("--first_n_only", type=int, default=None)

# new arguments
parser.add_argument('--input_path', default=None) # not input_file
parser.add_argument('--output_path', default=None) # not output_path


parser.add_argument('--test', action='store_true', help='Test command to print all dataset names')

args = parser.parse_args()

if args.command == 'test':
  print(all_dataset_names)
   
elif args.command == 'extract':
  """Extract beginnings of the texts according to dataset-specific preset"""

  from utils.dataset import Dataset
   
  for ds_name in args.datasets:
      ds = Dataset(ds_name)
      ds.extract()
      ds.update_preprocessed()

elif args.command == 'prompt':
  """Generate prompts for the extracted prefixes, according to dataset-specific preset"""
  from utils.dataset import Dataset
   
  for dataset_name in args.datasets:
      dataset = Dataset(dataset_name)
      dataset.promptify()
      dataset.update_preprocessed()
      
elif args.command == 'prepare':
    from utils.domains import prepare_domain
    for domain in args.domains:
      prepare_domain(domain, args.cap)

elif args.command == 'output':
    from utils.fileio import output_domains
    output_domains(args.domains, args.cap, args.output_file)


elif args.command == 'generate':
  """Run generation jobs on the cluster.
  
  The process consists of following steps:
  0. Establish indexes for effective batchifying of input data for generation concurrence
  1. Create custom, per-job slurm config files, incl. necessasry resources, models and input subset definition (see point 0)
  2. Run the jobs.
  """
  if not args.input_file:
    print('no input file provided!')
  import pandas as pd
  import math
  # create general folder name
  datestring = get_time_string()
  batch_name =  f"{datestring}_{args.name}"

  # create general folder
  current_path = os.getcwd() + '/'
  output_folder = f'{current_path}data/{batch_name}/'
  os.makedirs(f"{output_folder}/scripts") # creating scripts directory

  for model in args.models:

    the_input_df = pd.read_csv(args.input_file)
    data_len = args.first_n_only if args.first_n_only else len(the_input_df)

    # 2 batchify
    num_of_batches = math.ceil(data_len / args.batched)

    for batch_index in range(0, num_of_batches):
      if (not args.include_batches or (str(batch_index) in args.include_batches)):
        starting_index = batch_index * args.batched
        ending_index = starting_index + min(args.batched, args.first_n_only if args.first_n_only else args.batched) -1

        input_path = args.input_file

        run_id = f"{args.name}.{model}.{batch_index}"


        logs_path = output_folder + 'logs/'
        template = get_slum_script_content(
          run_id,
          model=model,
          walltime=args.walltime,
          input_file=input_path,
          subset_start_index=starting_index,
          subset_end_index=ending_index,
          job_dir = output_folder,
          std_out=logs_path + run_id + '_out.txt',
          std_err=logs_path + run_id + '_err.txt',
          mode=args.mode
        )
        slurm_script_path=f'{output_folder}scripts/generate_{run_id}.sh'
        print(f"ssp: {slurm_script_path}")
        with open(slurm_script_path, 'w') as text_file:
          text_file.write(template)
          text_file.close()
        subprocess.run(f"sbatch {slurm_script_path}", shell=True)

elif args.command == 'train-bert-baseline':
  template = get_slum_script_content(
    name=args.name,
    mode='train-bert'
    )
  slurm_script_path=f'{output_folder}scripts/generate_{run_id}.sh'
  print(f"ssp: {slurm_script_path}")
  with open(slurm_script_path, 'w') as text_file:
    text_file.write(template)
    text_file.close()
  subprocess.run(f"sbatch {slurm_script_path}", shell=True)


elif args.command == 'preprocess':
  from utils.dataset import Dataset
  
  for dataset_name in args.datasets:
      dataset = Dataset(dataset_name)
      dataset.extract()
      dataset.promptify()
      dataset.update_preprocessed()


elif args.command == 'forge-tests':
  from postprocessing.forge_tests import (
      load_txt_key_meta, write_txt_key_meta, split_in, interleave_lists
  )
  import pathlib
  data_path = pathlib.Path(__file__).parent / 'data'
  alpha = load_txt_key_meta(data_path / 'postprocessed' / 'test')  # test split = alpha
  beta  = load_txt_key_meta(data_path / 'postprocessed_beta')
  gamma = load_txt_key_meta(data_path / 'postprocessed_gamma')

  alpha_a, alpha_b        = split_in(2, alpha)
  beta_a, *beta_b_parts   = split_in(3, beta)
  gamma_a, *gamma_b_parts = split_in(3, gamma)

  test_a = interleave_lists([alpha_a, beta_a,  gamma_a])
  test_b = interleave_lists([alpha_b, beta_b_parts[0] + beta_b_parts[1],
                                      gamma_b_parts[0] + gamma_b_parts[1]])

  write_txt_key_meta(test_a, data_path / 'test_A')
  write_txt_key_meta(test_b, data_path / 'test_B')
  print(f"TEST A: {len(test_a)} examples → data/test_A/")
  print(f"TEST B: {len(test_b)} examples → data/test_B/")

elif args.command == 'baseline-train' or args.command == 'baseline-test':

  baseline_mode = 'testing' if args.command == 'baseline-test' else 'training_autoclass'
   # it is batch generation, but adapted
  datestring = get_time_string()
  batch_name =  f"{datestring}_{args.name}"

  # create general folder
  current_path = os.getcwd() + '/'
  output_folder = f'{current_path}data/{batch_name}/'
  os.makedirs(f"{output_folder}/scripts") # creating scripts directory
  logs_path = output_folder + 'logs/'

  for model in args.models:
    run_id = f'{args.name}_{model}'

    template = get_slum_script_content(
      run_id,
      model=model,
      walltime=args.walltime,
      # input_file=input_path,
      # subset_start_index=starting_index,
      # subset_end_index=ending_index,
      job_dir = output_folder,
      std_out=logs_path + run_id + '_out.txt',
      std_err=logs_path + run_id + '_err.txt',
      mode=baseline_mode
    )
    slurm_script_path=f'{output_folder}scripts/generate_{run_id}.sh'
    with open(slurm_script_path, 'w') as text_file:
      text_file.write(template)
      text_file.close()
      subprocess.run(f"sbatch {slurm_script_path}", shell=True)

