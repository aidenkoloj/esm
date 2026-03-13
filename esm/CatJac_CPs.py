''' Script to get Cat Jac of all CPs of input list of pdbs '''
from transformers import AutoModel, AutoTokenizer, EsmTokenizer, EsmForMaskedLM
import torch
from pathlib import Path
import time
import numpy as np
import gc
import sys
import json
import argparse

### Argparser
parser = argparse.ArgumentParser()
parser.add_argument("pdb_list", help="Path to JSON file containing PDB IDs")
parser.add_argument("--cp", action="store_true", help="Load finetuned CP model")
args = parser.parse_args()

### Save print statements to log
log = open("CatJac_CP.log", "w")
sys.stdout = log

############################
# Load the model
############################
if args.cp:
    print("Loading finetuned CP model...", flush=True)
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
    checkpoint = torch.load('/home/ubuntu/esm/esm/checkpoints/latest_best_ESM_CP_finetune_uniref50_chkpt_resume.pt')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model_tag = "ESM2_CP_finetuned"
else:
    print("Loading base ESM2 650M model...", flush=True)
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model_tag = "ESM2_650M"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Read filtered PDB list for analysis
with open(args.pdb_list, "r") as f:
    PDB_IDs = json.load(f)

### Funcs
def cp_seq(seq, i):
    '''circularly permute the sequence at position i'''
    return seq[i:] + seq[0:i]

def parse_fasta(fasta):
    with open(fasta, "r") as f:
        lines = f.readlines()
    return "".join(line.strip() for line in lines[1:])

def get_categorical_jacobian(seq):
    x, ln = alphabet.get_batch_converter()([("seq", seq)])[-1], len(seq)
    with torch.no_grad():
        f = lambda x: model(x)["logits"][...,1:(ln+1),4:24].cpu().numpy()
        fx = f(x.to(device))[0]
        x = torch.tile(x, [20,1]).to(device)
        fx_h = np.zeros((ln,20,ln,20))
        for n in range(ln):
            x_h = torch.clone(x)
            x_h[:,n+1] = torch.arange(4,24)
            fx_h[n] = f(x_h)
        return fx_h - fx

# Compute categorical jacobian for proteins
for PDB_ID in PDB_IDs:
    SEQ = parse_fasta(f'/home/ubuntu/esm/esm/test_pdbs/{PDB_ID}.fasta')
    L = len(SEQ)
    for i in range(L):
        SEQ_CP = cp_seq(SEQ, i)
        print(f"Compute catjac for {PDB_ID} CP {i}", flush=True)
        start_time = time.time()
        catjac = get_categorical_jacobian(SEQ_CP)
        print(f"Shape of jac: {catjac.shape}", flush=True)
        Path("catjac_outputs").mkdir(parents=True, exist_ok=True)
        np.save(f"catjac_outputs/{PDB_ID}_CP_{i}_{model_tag}_CatJac.npy", catjac)
        print(f"{PDB_ID}_CP_{i} saved", flush=True)
        runtime = time.time() - start_time
        print(f"Total runtime: {runtime/60:.4f} mins", flush=True)

log.close()
model = model.to("cpu")
gc.collect()
torch.cuda.empty_cache()