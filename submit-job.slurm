#!/bin/bash
#SBATCH --output=./logs/rocl_pretrining_out.txt
#SBATCH --error=./logs/rocl_pretraining_error.txt
#SBATCH --job-name=ROCL
#SBATCH --time=08:00:00
#SBATCH --gpus=2
#parallel details
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=2048M
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user tusharsangam5@gmail.com

##srun="srun -n1 -N1 --exclusive"

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --gpus=2



# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load python/python-3.8.0-gcc-9.1.0
module load cuda/cuda-10.2
#module load anaconda/anaconda3

#conda env list >> env.txt
# List the modules that are loaded
#module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of Pytorch
source activate pytorch-1.8.0+cuda10_2
#source activate pytorch-1.6.0+py38+cuda10_2
#pip install torch torchvision torchaudio diffdist
#python -m pip install --upgrade pip
pip uninstall torchvisison torchlars diffdist
pip install torchvision torchlars diffdist

# Run Script
echo
python -m torch.distributed.launch --nproc_per_node=2 main.py --epoch=1
echo

# You're done!
echo "Ending script..."
date


