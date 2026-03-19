#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -t 240
#SBATCH -o filter_fasta.log

python reduce_uniref50.py \
    --input /home/aidenkzj/uniref50.fasta \
    --output /home/aidenkzj/uniref50_filtered512.fasta \
    --max_len 512