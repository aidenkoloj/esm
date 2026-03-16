#!/usr/bin/env bash
set -e

PROT=$1

FASTA="${PROT}.fasta"
CP_DIR="${PROT}_CP"

BASE=/home/ubuntu/esm/esm/cp_interp

echo "======================================"
echo "Running synCP pipeline for $PROT"
echo "======================================"

#######################################
# Step 1 — Generate synCP structures
#######################################

echo "Generating synCPs with ESMFold..."

conda run -n esmfold_env \
python esmfold_synCPs.py $FASTA


#######################################
# Step 2 — Convert PDB → FASTA
#######################################

echo "Generating FASTAs from PDBs..."

conda run -n esm2-dev \
bash make_fastas.sh $CP_DIR


#######################################
# Step 3 — Contact prediction
#######################################

echo "Running ESM2 contact prediction..."

conda run -n esm2-dev \
python contact_pred.py $CP_DIR


#######################################
# Step 4 — TM-score calculation
#######################################

echo "Computing TM-scores..."

conda run -n esm2-dev \
python tm_scores.py $CP_DIR


#######################################
# Step 5 — Mean pLDDT
#######################################

echo "Computing mean pLDDT..."

conda run -n esm2-dev \
python plddt.py $CP_DIR


#######################################
# Step 6 — Generate plots
#######################################

# echo "Generating plots..."

# conda run -n esm2-dev \
# python make_plots.py $CP_DIR


echo "======================================"
echo "Pipeline finished for $PROT"
echo "======================================"