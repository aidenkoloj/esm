import sys
import torch
import openfold
from openfold.utils.kernel import attention_core
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import os, time

# ── Hardcoded model path ───────────────────────────────────────────────────
MODEL_PATH = "/home/ubuntu/esm/esm/esmfold.model"

# ── Parse fasta input ──────────────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: python run_CP_esmfold.py <input.fasta>")

fasta_file = sys.argv[1]
with open(fasta_file) as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

jobname = lines[0].lstrip(">")[:4]
sequence = "".join(lines[1:])

def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]
    bins = np.append(0,np.linspace(2.3125,21.6875,63))
    sm_contacts = softmax(output["distogram_logits"],-1)[0]
    sm_contacts = sm_contacts[...,bins<8].sum(-1)
    xyz = output["positions"][-1,0,:,1]
    mask = output["atom37_atom_exists"][0,:,1] == 1
    o = {"pae":pae[mask,:][:,mask],
         "plddt":plddt[mask],
         "sm_contacts":sm_contacts[mask,:][:,mask],
         "xyz":xyz[mask]}
    return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()

alphabet_list = list(ascii_uppercase+ascii_lowercase)

jobname = re.sub(r'\W+', '', jobname)[:50]

def get_circular_permutations(seq):
    return [(i, seq[i:] + seq[:i]) for i in range(len(seq))]

sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
sequence = re.sub(":+",":",sequence)
sequence = re.sub("^[:]+","",sequence)
sequence = re.sub("[:]+$","",sequence)

copies = 1
sequence = ":".join([sequence] * copies)
num_recycles = 3
chain_linker = 25

base_sequence = sequence.split(":")[0]
circular_permutations = get_circular_permutations(base_sequence)
print(f"Generating {len(circular_permutations)} circular permutations of length {len(base_sequence)}")

ID = jobname+"_"+get_hash(sequence)[:5]
seqs = sequence.split(":")
lengths = [len(s) for s in seqs]
length = sum(lengths)
print("length", length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load(MODEL_PATH, weights_only=False)
model = model.eval().to(device)
model.requires_grad_(False)

os.system(f"mkdir -p {jobname}_CP")

if length > 700:
    model.set_chunk_size(64)
else:
    model.set_chunk_size(128)


batch_size = 8
perms = circular_permutations

for i in range(0, len(perms), batch_size):

    batch = perms[i:i+batch_size]
    seqs = [p[1] for p in batch]

    print(f"\nBatch {i//batch_size + 1}")

    with torch.no_grad():
        outputs = model.infer(
            seqs,
            num_recycles=num_recycles,
            chain_linker="X"*chain_linker,
            residue_index_offset=512
        )

    for j, (cut_pos, seq) in enumerate(batch):

        output = tree_map(lambda x: x[j].cpu().numpy(), outputs)

        ptm = output["ptm"][0]
        plddt = output["plddt"][...,1].mean()

        prefix = f"{jobname}_CP/cut{cut_pos:04d}_ptm{ptm:.3f}_r{num_recycles}"

        np.savetxt(f"{prefix}.plddt.txt", output["plddt"][:,1], "%.3f")

        with open(f"{prefix}.pdb","w") as out:
            out.write(model.output_to_pdb(outputs)[j])


print("\nDone! All circular permutations complete.")