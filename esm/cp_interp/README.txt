Directory containing scripts to do the following:
For a given protein of interest:
> Run esmfold_synCP.py in esmfold_dev conda environment to get all synCPs (input: fasta. i.e. 23TZ.fasta) 
> Run the rest in esm2-dev
> run make_fastas.sh {PDB} to get fastas of all synCPs (input: PDB_CP i.e. 23TZ_CP)
> Run contact_pred to get P@L .tsv (input: PDB_CP i.e. 23TZ_CP)
> Run tm_scores to get tm scores against unpermuted structures
and make a new colun in the previous .tsv with tm_score (input: PDB_CP i.e. 23TZ_CP)
> Run plddt to get mean plddt for each synCP (input: PDB_CP i.e. 23TZ_CP)






> ESM Fold all synCPs (requires being done in esmfold_env)
> Get P@L for all synCPs > ESM Fold all synCPs (requires being done in ESM2-dev)

> Calculate TMscore to the first synCP (the unpermuted structure predicted by ESM Fold)
> Make three plots
> P@L vs plddt
> P@L vs TM score
> TM score vs plddt