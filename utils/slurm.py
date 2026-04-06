
def get_slum_script_content(
    name,
    model,
    walltime,
    job_dir,
    input_file=None,
    subset_start_index=None,
    subset_end_index=None,
    std_out=None,
    std_err=None,
    mode='generate'
    ):
  template = f'''#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J {name}
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time={walltime}
## Plik ze standardowym wyjściem
#SBATCH --output="{std_out}"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="{std_err}"

## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem=256GB
## Nazwa grantu do rozliczenia zużycia zasobów CPU
#SBATCH -A plgmgtpl-gpu-a100
## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:4

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

echo "Loading dependencies"
module load CUDA/12.4.0
module load Python/3.10.4
# module load libffi/3.4.2

# echo "Activating environment"
source .venv/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR
bash utils/install.sh

# run the generation
echo $(pwd)

TORCH_LOGS="recompiles" TRANSFORMERS_VERBOSITY='info' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 utils/{mode}.py\
 --model='{model}'\
 --input_file_path='{input_file}'\
 --job_dir='{job_dir}'\
{f' --subset_start_index={subset_start_index}' if isinstance(subset_start_index, int) else ''}\
{f' --subset_end_index={subset_end_index}' if isinstance(subset_end_index, int) else ''}\
'''
  return template
