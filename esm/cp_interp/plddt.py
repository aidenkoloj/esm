import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


def mean_plddt(plddt_txt: str) -> float:
    """Read a per-residue pLDDT file (one float per line) and return the mean."""
    with open(plddt_txt) as fh:
        values = [float(line.strip()) for line in fh if line.strip()]
    return float(np.mean(values))


if len(sys.argv) < 2:
    print("Usage: python add_plddt.py <cp_directory>")
    sys.exit(1)

cp_dir = sys.argv[1]

# Base directory where everything lives
base_dir = "/home/ubuntu/esm/esm/cp_interp"

data_dir = os.path.join(base_dir, cp_dir)

# Input TSV
in_path = os.path.join(base_dir, f"{cp_dir}_contact_precision_tm_score.tsv")

df = pd.read_csv(in_path, sep="\t")

plddts = []

for _, row in df.iterrows():
    plddt_file = os.path.join(data_dir, f"{row['id']}.plddt.txt")
    plddt = mean_plddt(plddt_file)
    plddts.append(plddt)

df["plddt"] = plddts

# Output TSV
out_path = os.path.join(base_dir, f"{cp_dir}_contact_precision_tm_score_plddt.tsv")

df.to_csv(out_path, sep="\t", index=False)

print(f"Saved to {out_path}")