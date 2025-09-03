# Import ESM classes from HF transformers
from transformers import EsmTokenizer, EsmForMaskedLM, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torch.nn.functional as F
import random
import wandb
import argparse


def setup_model(model_name="facebook/esm2_t6_8M_UR50D"):
    """
    Initialize model for fine-tuning
    """
    
    model = EsmForMaskedLM.from_pretrained(model_name)
    
    return model


class UniRef50Dataset(Dataset):
    """
    Dataset class for UniRef50 protein sequences.

    This class loads sequences from a UniRef50 FASTA file and tokenizes them 
    using an ESM tokenizer for downstream model training. Sequences 
    can be circularly permuted with (`cp`). 

    Parameters
    ----------
    fasta_path : str
        Path to the UniRef50 FASTA file containing protein sequences.
    tokenizer_name : str
        Name of the pretrained ESM tokenizer to use (e.g. `"facebook/esm2_t33_650M_UR50D"`).
    max_length : int, default=1024
        Maximum sequence length (longer sequences will be truncated).
    cp : bool, default=True
        Whether to apply circular permutation to sequences.

    Attributes
    ----------
    sequences : list of str
        Raw protein sequences read from the FASTA file.
    tokenizer : EsmTokenizer
        Tokenizer used to convert protein sequences into tokens.
    max_length : int
        Maximum length cutoff for sequences.
    cp : bool
        circular permutation applied
    """
    def __init__(self, fasta_path, tokenizer_name, max_length = 1024, cp = True):
        self.sequences = self.read_fasta(fasta_path)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.cp = cp

    
    def read_fasta(self, fasta_path):
        ''' Get list of sequences from the fasta file 
        Input: fasta file path
        Returns: sequences from fasta file'''
        sequences = []
        seq = []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq:
                        sequences.append("".join(seq))
                        seq = []
                else:
                    seq.append(line)
            if seq:
                sequences.append("".join(seq))
                
            return sequences
            
    def __len__(self):
        ''' Returns number of sequences in the dataset '''
        return len(self.sequences)

    def __getitem__(self, idx):
    """
    Retrieve and tokenize a protein sequence.

    Parameters
    ----------
    idx : int
        Index of the sequence to retrieve.

    Returns
    -------
    dict
        A dictionary containing:
        
        - **input_ids** (torch.Tensor): Tokenized sequence IDs of shape `(L,)`,
          where `L <= max_length`.
        - **attention_mask** (torch.Tensor): Attention mask of shape `(L,)`
          indicating which tokens are padding (0) vs. real sequence (1).
    """
        seq = self.sequences[idx]

        # Create circular permutation of the sequence if cp == True
        if self.cp:
            i = random.randint(0, len(seq)-1)
            seq_cp = seq[i:] + seq[:i]

            #Tokenize the circular permutation
            tokenized = self.tokenizer(
                seq_cp,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        else:
            #Tokenize the sequence
            tokenized = self.tokenizer(
                seq,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

        # Return a dictionary
        return {key: val.squeeze(0) for key, val in tokenized.items()}

        
def generate_dataset_splits(dataset, splits=[0.70, 0.15, 0.15], seed=None):
    """
    Split a dataset into train, validation, and test subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be split.
    splits : list of float, optional, default=[0.70, 0.15, 0.15]
        Proportions for train, validation, and test splits. Must sum to 1.0.
    seed : int or None, optional
        Random seed for reproducibility. If None (default), a generator 
        with no fixed seed is used.

    Returns
    -------
    tuple of torch.utils.data.Subset
        - **train_dataset** : Subset corresponding to the training split.
        - **val_dataset** : Subset corresponding to the validation split.
        - **test_dataset** : Subset corresponding to the test split.

    Notes
    -----
    - Internally uses `torch.utils.data.random_split`.
    - If `seed` is set, the split will always be the same for the same dataset.
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * splits[0])
    val_size = int(dataset_size * splits[1])
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset

from torch.utils.data import DataLoader

def generate_dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Create DataLoaders for training, validation, and test splits.

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        Training dataset (usually from `random_split`).
    val_dataset : torch.utils.data.Dataset
        Validation dataset.
    test_dataset : torch.utils.data.Dataset
        Test dataset.
    batch_size : int
        Number of samples per batch.

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        - **train_loader** : DataLoader with shuffling, prefetching, and pinned memory.
        - **val_loader** : DataLoader without shuffling.
        - **test_loader** : DataLoader without shuffling.

    Notes
    -----
    - `train_loader` is configured with:
        - `shuffle=True` to randomize batches each epoch.
        - `pin_memory=True`, which can speed up host-to-device transfer when using GPUs.
        - `prefetch_factor=2` and `num_workers=16` to load data in parallel.
    - `val_loader` and `test_loader` use `shuffle=False` for deterministic evaluation.
    - A common rule of thumb for `num_workers` is `(CPU cores // 2)`, 
      but the optimal value depends on your system and dataset.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=16
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16
    )

    return train_loader, val_loader, test_loader



def get_special_token_ids(tokenizer_name):
    """Extract special token IDs once and return them
      Helper function to extract special token IDs once
    """
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
    special_ids = [
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id
    ]
    return special_ids, tokenizer.mask_token_id

def generate_training_mask(input_ids, special_ids, mask_token_id, mask_prob=0.15):
    """
    Generate masked input IDs and corresponding ground-truth labels for MLM training.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor of token IDs with shape `(batch_size, seq_len)` or `(seq_len,)`.
    special_ids : torch.Tensor
        1D tensor containing token IDs that should never be masked 
        (e.g., [CLS], [SEP], [PAD]).
    mask_token_id : int
        Token ID to use for masking (e.g., tokenizer.mask_token_id).
    mask_prob : float, optional, default=0.15
        Probability of masking a given non-special token.

    Returns
    -------
    masked_input_ids : torch.Tensor
        Copy of `input_ids` with ~`mask_prob` fraction of non-special tokens replaced 
        by `mask_token_id`.
    ground_truth_labels : torch.Tensor
        Copy of `input_ids` where unmasked positions are set to -100 (ignored by 
        `nn.CrossEntropyLoss`), and masked positions retain their original token IDs.

    Notes
    -----
    - Roughly 15% of non-special tokens are replaced by the mask token.
    - Special tokens (e.g. [CLS], [SEP], [PAD]) are preserved and never masked.
    - The `ground_truth_labels` tensor is used as the target for loss computation.
    """
    # copy the input ids to a masked inputs ids tensor
    masked_input_ids = input_ids.clone()

    # Create a labels tensor used for ground truth labels during training
    ground_truth_labels = input_ids.clone()
    
    # assign random number 0 to 1 to each input id
    rand = torch.rand(input_ids.shape, device=input_ids.device)

    # Create a mask; all positions less than mask_prob and not special tokens
    mask_arr = (rand < mask_prob) & (~torch.isin(input_ids, special_ids))

    # Apply the mask to the input_ids
    masked_input_ids[mask_arr] = mask_token_id

    # For the labels, set unmasked tokens to -100 (ignored in loss calculation)
    ground_truth_labels[~mask_arr] = -100

    return masked_input_ids, ground_truth_labels



def training_loop(run_name, model, train_loader, val_loader):
    """
    Training loop for fine-tuning ESM on circularly-permuted sequences
    """
    # Integrate W&B
    run = wandb_run(run_name)
    
    # Get hyperparameters (number of epochs, learning rate, etc)
    config = run.config
    
    # Move model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer; give the optimizer the model parameters and learning rate as inputs. 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        # Set model to train and loss to zero
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        # Loop through the batches in the dataloader
        for step, batch in enumerate(train_loader):
            
            # Move the inputs from the batch to the GPU
            input_ids = batch['input_ids'].to(device)
            # Generate masked inputs and ground truth labels
            masked_input_ids, ground_truth_labels = generate_training_mask(input_ids, config.mask_prob)
            # Get the attention mask (masks out padding tokens during self-attention
            attention_mask = batch['attention_mask'].to(device)
            # Generate labels using mask function
            #labels = batch['labels'].to(device) # Loader doesn't have labels in this case

            # Get outputs from the model
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=ground_truth_labels
            )

            # Scale loss for gradient accumulation
            loss = outputs.loss / config.accumulation_steps
            loss.backward()
    
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            # Step optimizer every accumulation_steps
            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
    
            total_loss += loss.item() * config.accumulation_steps  # undo scaling for logging

        
        avg_loss = total_loss / len(train_loader)
        run.log({"train_loss": avg_loss, "epoch": epoch + 1})
        #print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        
        val_total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
        
                # Mask for MLM (same as training)
                masked_input_ids, ground_truth_labels = generate_training_mask(input_ids, mask_prob=config.mask_prob)
        
                # Forward pass
                outputs = model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    labels=ground_truth_labels
                )
        
                # Collect loss
                val_total_loss += outputs.loss.item()
        
        # Average validation loss for the epoch
        avg_val_loss = val_total_loss / len(val_loader)
        run.log({"val_loss": avg_val_loss, "epoch": epoch + 1})
        #print(f"Validation Loss: {avg_val_loss:.4f}")

def wandb_run(run_name):
    run = wandb.init(
        entity="aidenkzj",  # Change to your W&B entity
        project="esm-circular-permutation-finetune",  # Change to your project name
        name=run_name,  # Set run name dynamically
        config={
            "accumulation_steps": 8,
            "learning_rate": 1e-4,
            "architecture": "ESM-MLM",
            "dataset": "circularly-permuted-sequences",
            "epochs": 3,
            "mask_prob": 0.15,
            "batch_size": 64,  # adjust based on your DataLoader
            "optimizer": "AdamW",
            "loss_fn": "CrossEntropyLoss(ignore_index=-100)",
        },
    )
    return run

def main(args):
    print('> Loading Model, Tokenizer:')
    model = setup_model(args.model_name)
    print('> Loading Dataset:')
    dataset = UniRef50Dataset(
    fasta_path=args.fasta_path,
    tokenizer_name=args.tokenizer_name,
    max_length=args.max_length
)
    print('> Loading Dataset splits:')
    train_dataset, val_dataset, test_dataset = generate_dataset_splits(dataset, seed=args.seed)
    print('> Loading DataLoaders:')
    train_loader, val_loader, test_loader = generate_dataloader(train_dataset, val_dataset, test_dataset, 64)
    print('> Starting Training Loop:')
    training_loop(args.run_name, model, train_loader, val_loader) 
    print('> Finetuning Workflow Complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune ESM model on circularly permuted sequences.")
    
    parser.add_argument("--fasta_path", type=str, default = "/home/ubuntu/uniref50_subset.fasta", help="Path to UniRef50 FASTA file")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Hugging Face ESM tokenizer name")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Hugging Face ESM model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--run_name", type=str, default="esm_cp_ft_test_run", help="Name for W&B run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)
