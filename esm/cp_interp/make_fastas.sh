#!/usr/bin/env bash

DIR="${1:-.}"

for pdb in "$DIR"/*.pdb; do
    base=$(basename "$pdb" .pdb)
    pdb_tofasta "$pdb" | sed "s/>PDB|A/>$base/" > "$DIR/${base}.fasta"
    echo "Written: $DIR/${base}.fasta"
done