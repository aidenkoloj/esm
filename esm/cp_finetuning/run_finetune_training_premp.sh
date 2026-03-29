#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1        # Always 1 — torchrun handles process spawning
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 46:00:00
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH -o output_%j.txt 
#SBATCH --exclude=node9905

module load miniforge/24.3.0-0
source activate esm2-dev
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export TORCH_DISTRIBUTED_DEBUG=DETAIL


torchrun --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint="localhost:1234" \
finetune_esm_distributed_wip.py \
  --batch_size 64 \
  --max_length 256 \
  --run_name esm_cp_mlength256_batchsize64_run1
