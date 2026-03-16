import sys
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def tmscore(q, t, cp=True):
    ''' Run TM-align and get back TM-align score '''
    if cp:
        output = os.popen(f'/home/ubuntu/TM_tools/TMalign {q} {t} -cp')
    else:
        output = os.popen(f'/home/ubuntu/TM_tools/TMalign {q} {t}')
    tms = {"tms": []}
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    for line in output:
        line = line.rstrip()
        if line.startswith("TM-score"):
            tms["tms"].append(parse_float(line))
    min_tms = min(tms['tms'])
    return min_tms

data_dir = '/home/ubuntu/esm/esm/cp_interp/' + sys.argv[1]
contact_precision = pd.read_csv(f'{data_dir}_contact_precision.tsv', sep='\t')

# Copy the dataframe
contact_precision_tm_score = contact_precision.copy()

# Get unpermuted pdb (file beginning with 'cut0000')
ref_pdb_matches = glob.glob(os.path.join(data_dir, 'cut0000*.pdb'))
if not ref_pdb_matches:
    raise FileNotFoundError(f"No reference PDB starting with 'cut0000' found in {data_dir}")
ref_pdb = ref_pdb_matches[0]

# Iterate over rows, compute TM-score for each
tm_scores = []
for _, row in contact_precision_tm_score.iterrows():
    tpdb = os.path.join(data_dir, row['id'] + '.pdb')
    tm = tmscore(ref_pdb, tpdb)
    tm_scores.append(tm)

contact_precision_tm_score['tm_score'] = tm_scores

# Save as TSV
out_path = f'{data_dir}_contact_precision_tm_score.tsv'
contact_precision_tm_score.to_csv(out_path, sep='\t', index=False)
print(f"Saved to {out_path}")