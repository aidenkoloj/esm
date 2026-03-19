# Import ESM classes from HF transformers
from transformers import EsmTokenizer, EsmForMaskedLM, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torch.nn.functional as F
import random
import wandb
import argparse
import os
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Setup the model
def setup_model(model_name="facebook/esm2_t30_150M_UR50D", checkpoint_path=None):
    """
    Initialize model for fine-tuning
    """
    
    model = EsmForMaskedLM.from_pretrained(model_name)

    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path)
        
    # Freeze parameters that don't participate in MLM forward pass
    frozen = {
        "esm.encoder.emb_layer_norm_after.bias",
        "esm.contact_head.regression.weight",
        "esm.contact_head.regression.bias",
    }
    
    for name, param in model.named_parameters():
        if name in frozen or "contact_head" in name:
            param.requires_grad = False
    
    return model

# Load weights from a finetuning model checkpoint
def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from checkpoint
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to load weights into
    checkpoint_path : str
        Path to checkpoint file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Loading from 'model_state_dict' key")
        
        # Print additional checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            print(f"Training loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
            
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Loading from 'state_dict' key")
    else:
        state_dict = checkpoint
        print("Loading checkpoint as raw state dict")
    
    # Load the state dict
    print("Loading fine-tuned weights...")
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✓ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Continuing with base model weights...")

# Resume training from a checkpoint
def resume_training_from_checkpoint(checkpoint_path, optimizer=None):
    """
    Load training state from checkpoint for resuming training
    Enhanced to handle batch counting
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    training_state = {}
    
    if 'epoch' in checkpoint:
        training_state['start_epoch'] = checkpoint['epoch']
        print(f"Resuming from epoch: {checkpoint['epoch']}")
    else:
        training_state['start_epoch'] = 0
        
    # Handle batch counting for proper resuming
    if 'total_batches_processed' in checkpoint:
        training_state['total_batches_processed'] = checkpoint['total_batches_processed']
        print(f"Resuming from batch: {checkpoint['total_batches_processed']:,}")
    else:
        training_state['total_batches_processed'] = 0
        
    if 'optimizer_step_count' in checkpoint:
        training_state['optimizer_step_count'] = checkpoint['optimizer_step_count']
        print(f"Optimizer step count: {checkpoint['optimizer_step_count']:,}")
    else:
        training_state['optimizer_step_count'] = 0
        
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
    if 'train_loss' in checkpoint:
        training_state['last_train_loss'] = checkpoint['train_loss']
        
    if 'best_val_loss' in checkpoint:
        training_state['best_val_loss'] = checkpoint['best_val_loss']
    elif 'val_loss' in checkpoint:
        training_state['best_val_loss'] = checkpoint['val_loss']
    else:
        training_state['best_val_loss'] = float('inf')
        
    return training_state



# Create the dataset to train on 
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
    def __init__(self, fasta_path, tokenizer_name, max_length, cp = True):
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
    print(f'Number of sequences in dataset: {dataset_size:,}')
    
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

def generate_dataloader(train_dataset, val_dataset, test_dataset, args):
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
    
    
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,   # replaces shuffle=True
        pin_memory=True,
        prefetch_factor=2,
        num_workers=4            # 4 per GPU, not 16 total
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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

def generate_training_mask(input_ids, special_ids, mask_token_id, mask_prob):
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
    special_ids = special_ids.to(input_ids.device)
    
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

def training_loop(args, model, train_loader, val_loader, special_ids, mask_token_id):
    """
    Training loop for fine-tuning ESM on circularly-permuted sequences
    Now validates and saves checkpoints every 100K batches instead of every epoch
    """
    # Move model to the GPUs (process group already initialized in main)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # SINGLE GPU LOGGING: only rank 0 initializes wandb and logs metrics
    if dist.get_rank() == 0:
        run = wandb_run(args)
        config = run.config
    else:
        # All other ranks still need the config values — broadcast from rank 0
        # by constructing a plain namespace with the same defaults
        import types
        config = types.SimpleNamespace(
            accumulation_steps=8,
            learning_rate=1e-4,
            epochs=10000,
            mask_prob=0.15,
        )

    # Define optimizer; give the optimizer the model parameters and learning rate as inputs. 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    optimizer_step_count = 0
    total_batches_processed = 0
    
    # Validation and checkpoint intervals
    # getattr = get attribute from args, set default if no attribute
    validation_interval = args.validation_interval  # Default 100K batches
    log_interval = args.log_interval  # Log every 1K batches
    
    # Handle resuming from checkpoint
    if args.resume_checkpoint:
        # SINGLE GPU LOGGING: only rank 0 prints resume info
        if dist.get_rank() == 0:
            print(f"Resuming training from: {args.resume_checkpoint}")
        training_state = resume_training_from_checkpoint(args.resume_checkpoint, optimizer)
        start_epoch = training_state.get('start_epoch', 0)
        best_val_loss = training_state.get('best_val_loss', float('inf'))
        total_batches_processed = training_state.get('total_batches_processed', 0)
        optimizer_step_count = training_state.get('optimizer_step_count', 0)
        # SINGLE GPU LOGGING: only rank 0 prints resume state
        if dist.get_rank() == 0:
            print(f"Resuming from epoch {start_epoch}, batch {total_batches_processed}, best val loss: {best_val_loss:.4f}")

    checkpoint_dir = args.checkpoint_dir
    # SINGLE GPU LOGGING: only rank 0 creates the checkpoint directory
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Running loss tracking for logging
    running_loss = 0.0
    running_loss_count = 0
    
    for epoch in range(start_epoch, config.epochs):
        train_loader.sampler.set_epoch(epoch)
        # Time epoch start
        epoch_start_time = time.time()
        
        # Set model to train mode
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Loop through the batches in the dataloader
        for step, batch in enumerate(train_loader):
            total_batches_processed += 1
            
            # SINGLE GPU LOGGING: only rank 0 prints progress and loss
            if dist.get_rank() == 0:
                if total_batches_processed % log_interval == 0:
                    print(f'Epoch {epoch+1}, Global Batch {total_batches_processed:,}, Epoch Batch {step+1}/{len(train_loader)}')
                    if running_loss_count > 0:
                        avg_running_loss = running_loss / running_loss_count
                        print(f'  Running avg loss (last {running_loss_count} batches): {avg_running_loss:.4f}')
            
            # Move the inputs from the batch to the GPU
            input_ids = batch['input_ids'].to(device)
            # Generate masked inputs and ground truth labels
            masked_input_ids, ground_truth_labels = generate_training_mask(input_ids, special_ids, mask_token_id, config.mask_prob)
            # Get the attention mask (masks out padding tokens during self-attention)
            attention_mask = batch['attention_mask'].to(device)

            # Get outputs from the model
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=ground_truth_labels
            )

            # Scale loss for gradient accumulation
            loss = outputs.loss / config.accumulation_steps
            loss.backward()
            
            # Track running loss for logging (rank 0 only)
            if dist.get_rank() == 0:
                true_loss_value = outputs.loss.item()
                running_loss += true_loss_value
                running_loss_count += 1
                epoch_loss += true_loss_value
                epoch_batches += 1
    
            # Step optimizer every accumulation_steps
            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

                # SINGLE GPU LOGGING: only rank 0 logs to wandb
                if dist.get_rank() == 0:
                    if optimizer_step_count % 1000 == 0:
                        avg_loss_for_log = running_loss / max(running_loss_count, 1)
                        wandb.log({
                            "train_loss": avg_loss_for_log,
                            "global_batch": total_batches_processed,
                            "optimizer_step": optimizer_step_count,
                            "epoch": epoch + 1
                        }, step=optimizer_step_count)
            
            # **VALIDATION AND CHECKPOINT SAVING EVERY 100K BATCHES**
            if total_batches_processed % validation_interval == 0:
                # SINGLE GPU LOGGING: only rank 0 prints validation header
                if dist.get_rank() == 0:
                    print(f"\n=== VALIDATION at {total_batches_processed:,} batches ===")
                validation_start_time = time.time()
                
                # Run validation
                model.eval()
                val_total_loss = 0
                val_batches = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        # Move batch to device
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                
                        # Mask for MLM (same as training)
                        val_masked_input_ids, val_ground_truth_labels = generate_training_mask(
                            val_input_ids, special_ids, mask_token_id, config.mask_prob
                        )
                
                        # Forward pass
                        val_outputs = model(
                            input_ids=val_masked_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_ground_truth_labels
                        )
                
                        # Collect loss
                        val_total_loss += val_outputs.loss.item()
                        val_batches += 1
                
                # SINGLE GPU LOGGING: only rank 0 computes, logs, and prints validation results
                if dist.get_rank() == 0:
                    avg_val_loss = val_total_loss / max(val_batches, 1)
                    avg_train_loss = running_loss / max(running_loss_count, 1)
                    validation_time = time.time() - validation_start_time
                    print(f"Validation completed in {validation_time:.2f} seconds")
                    print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                    
                    wandb.log({
                        "val_loss": avg_val_loss,
                        "train_loss_epoch_avg": avg_train_loss,
                        "global_batch": total_batches_processed,
                        "epoch": epoch + 1,
                        "validation_time_seconds": validation_time
                    }, step=total_batches_processed)

                    # SINGLE GPU LOGGING: only rank 0 saves checkpoints
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        
                        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_batch_{total_batches_processed}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'total_batches_processed': total_batches_processed,
                            'optimizer_step_count': optimizer_step_count,
                            'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                            'best_val_loss': best_val_loss,
                            'args': args,
                            'config': dict(config) if hasattr(config, 'items') else config
                        }, checkpoint_path)
                        print(f"✓ NEW BEST MODEL! Checkpoint saved: {checkpoint_path}")
                        wandb.log({"best_checkpoint_saved": total_batches_processed})
                        
                        latest_best_path = os.path.join(checkpoint_dir, f"latest_best_{args.run_name}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'total_batches_processed': total_batches_processed,
                            'optimizer_step_count': optimizer_step_count,
                            'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                            'best_val_loss': best_val_loss,
                            'args': args,
                            'config': dict(config) if hasattr(config, 'items') else config
                        }, latest_best_path)
                        
                    else:
                        print(f"Validation loss did not improve (current: {avg_val_loss:.4f}, best: {best_val_loss:.4f})")
                    
                    # SINGLE GPU LOGGING: only rank 0 saves regular checkpoints
                    if getattr(args, 'save_regular_checkpoints', False):
                        regular_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{total_batches_processed}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'total_batches_processed': total_batches_processed,
                            'optimizer_step_count': optimizer_step_count,
                            'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                            'best_val_loss': best_val_loss,
                            'args': args,
                            'config': dict(config) if hasattr(config, 'items') else config
                        }, regular_checkpoint_path)
                        print(f"✓ Regular checkpoint saved: {regular_checkpoint_path}")
                    
                    # Reset running loss tracking
                    running_loss = 0.0
                    running_loss_count = 0
                    print("=== Resuming training ===\n")

                # Set model back to training mode (all ranks)
                model.train()

        # SINGLE GPU LOGGING: only rank 0 prints and logs epoch summary
        if dist.get_rank() == 0:
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Time: {epoch_time/3600:.2f} hours")
            print(f"  Avg Loss: {avg_epoch_loss:.4f}")
            print(f"  Batches Processed: {epoch_batches}")
            print(f"  Total Batches So Far: {total_batches_processed:,}")
            
            wandb.log({
                "epoch_time_hours": epoch_time / 3600,
                "epoch_avg_loss": avg_epoch_loss,
                "epoch_batches": epoch_batches,
                "epoch": epoch + 1
            }, step=epoch + 1)

    # SINGLE GPU LOGGING: only rank 0 saves the final model
    if dist.get_rank() == 0:
        final_path = os.path.join(checkpoint_dir, f"esm_cp_finetuned_{args.run_name}_final.pt")
        torch.save({
            'epoch': config.epochs,
            'total_batches_processed': total_batches_processed,
            'optimizer_step_count': optimizer_step_count,
            'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'args': args,
            'config': dict(config) if hasattr(config, 'items') else config
        }, final_path)
        
        print(f"\nTraining completed! Final model saved as: {final_path}")
        print(f"Total batches processed: {total_batches_processed:,}")
        print(f"Best validation loss: {best_val_loss:.4f}")



def wandb_run(args):
    run = wandb.init(
        entity="aidenkzj",  # Change to your W&B entity
        project="esm_cp_finetune_2",  # Change to your project name
        name=args.run_name,  # Set run name dynamically
        config={
            "accumulation_steps": 8,
            "learning_rate": 1e-4,
            "architecture": "ESM-MLM",
            "dataset": "circularly-permuted-sequences",
            "epochs": 10000,
            "mask_prob": 0.15,
            "batch_size": args.batch_size,
            "max_length": args.max_length,# adjust based on your DataLoader
            "optimizer": "AdamW",
            "loss_fn": "CrossEntropyLoss(ignore_index=-100)",
        },
    )
    return run

def main(args):
    # Initialize distributed process group before anything else so
    # dist.get_rank() is available throughout main and training_loop
    dist.init_process_group(backend="nccl")

    # SINGLE GPU LOGGING: only rank 0 prints setup progress
    if dist.get_rank() == 0:
        print('> Loading Model, Tokenizer:')
    model = setup_model(args.model_name, args.checkpoint_path)
    if dist.get_rank() == 0:
        print('> Loading Dataset:')
    dataset = UniRef50Dataset(
        fasta_path=args.fasta_path,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length
    )
    if dist.get_rank() == 0:
        print('> Loading Dataset splits:')
    train_dataset, val_dataset, test_dataset = generate_dataset_splits(dataset, seed=args.seed)
    if dist.get_rank() == 0:
        print('> Loading DataLoaders:')
    train_loader, val_loader, test_loader = generate_dataloader(train_dataset, val_dataset, test_dataset, args)
    if dist.get_rank() == 0:
        print('> Getting special tokens:')
    special_ids, mask_token_id = get_special_token_ids(args.tokenizer_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    special_ids_tensor = torch.tensor(special_ids, device=device)
    if dist.get_rank() == 0:
        print('> Starting Training Loop:')
    training_loop(args, model, train_loader, val_loader, special_ids_tensor, mask_token_id)
    if dist.get_rank() == 0:
        print('> Finetuning Workflow Complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune ESM model on circularly permuted sequences.")
    
    parser.add_argument("--fasta_path", type=str, default = "/orcd/home/002/aidenkzj/uniref50_filtered512.fasta", help="Path to UniRef50 FASTA file")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/esm2_t30_150M_UR50D", help="Hugging Face ESM tokenizer name")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D", help="Hugging Face ESM model name")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--run_name", type=str, default="esm_cp_ft_test_run", help="Name for W&B run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Checkpoint loading arguments
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to load model weights from")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from (loads optimizer state too)")
    parser.add_argument("--checkpoint_dir", type=str, default="/orcd/home/002/aidenkzj/esm/esm/cp_finetuning/checkpoints", help="Directory to save checkpoints")
    
    # Validation and logging intervals
    parser.add_argument("--validation_interval", type=int, default=100000, help="Run validation every N batches (default: 100,000)")
    parser.add_argument("--log_interval", type=int, default=1000, help="Print progress every N batches (default: 1,000)")
    
    # Checkpoint saving strategy - FIXED
    parser.add_argument("--save_regular_checkpoints", action="store_true", help="Save checkpoints every validation interval regardless of loss improvement")
    
    args = parser.parse_args()
    main(args)
