#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gres=gpu:h200:4
#SBATCH -c 16
#SBATCH --mem=80GB
#SBATCH -t 60
#SBATCH --requeue
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt

module load miniforge/24.3.0-0
source activate esm2-dev

torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint="localhost:1234" \
finetune_esm_distributed.py \
  --batch_size 32 \
  --fasta_path /orcd/home/002/aidenkzj/uniref50_filtered512.fasta \
  --run_name esm_cp_4gpu