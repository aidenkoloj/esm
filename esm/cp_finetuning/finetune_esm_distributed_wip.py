# Import ESM classes from HF transformers
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
import random
import wandb
import argparse
import os
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import glob

# Setup the model
def setup_model(model_name="facebook/esm2_t30_150M_UR50D",resume_checkpoint=None):
    """
    Initialize model for fine-tuning
    """
    
    model = EsmForMaskedLM.from_pretrained(model_name)

    if resume_checkpoint is not None:
        print(f"Loading checkpoint from: {resume_checkpoint}")
        load_checkpoint(model, resume_checkpoint)
        
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

def get_shard_files(shards_dir):

    pattern = os.path.join(shards_dir, "shard_*.fasta")

    shard_files = []
    
    shard_files.extend(glob.glob(pattern))
    
    shard_files = sorted(shard_files)
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shards_dir}")
    return shard_files

def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from checkpoint
      
    Loads the checkpoint model weights to the pretrained model
    Model weights are stored in the checkpoint 'model_state_dict' key
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
 
    # Print additional checkpoint info if available
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'shard' in checkpoint:
        print(f"Checkpoint from shard: {checkpoint['shard']}")
    if 'train_loss' in checkpoint:
        print(f"Training loss: {checkpoint['train_loss']:.4f}")
    # Load the state dict
    print("Loading fine-tuned weights...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Continuing with base model weights...")

def resume_training_from_checkpoint(checkpoint_path, optimizer=None):
    """
    Load training state from checkpoint for resuming training AND loads the checkpoint optimizer
    Returns a dictionary that contains relevant training states:
    -epoch
    -total_batches_processed
    -optimizer_step_count
    -
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    training_state = {}
    
    if 'epoch' in checkpoint:
        training_state['epoch'] = checkpoint['epoch']
        print(f"Resuming from epoch: {checkpoint['epoch']}")
    else:
        training_state['epoch'] = 0
        
    if 'shard_idx' in checkpoint:
        training_state['shard_idx'] = checkpoint['shard_idx']
        print(f"Resuming from shard: {checkpoint['shard_idx']:,}")
    else:
        training_state['shard_idx'] = 0
        
    # Handle batch counting for proper resuming
    if 'total_batches_processed' in checkpoint:
        training_state['total_batches_processed'] = checkpoint['total_batches_processed']
        print(f"Resuming from total batches processed: {checkpoint['total_batches_processed']:,}")
    else:
        training_state['total_batches_processed'] = 0
        
    if 'optimizer_step_count' in checkpoint:
        training_state['optimizer_step_count'] = checkpoint['optimizer_step_count']
        print(f"Optimizer step count: {checkpoint['optimizer_step_count']:,}")
    else:
        training_state['optimizer_step_count'] = 0
        
    
    # Load the checkpoint optimizer
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
    # Return all the training state data
    return training_state



# Create the dataset to train on 
class UniRef50Dataset_Shard(Dataset):

    def __init__(self, fasta_path, tokenizer_name, max_length, cp = True):
        self.sequences = self.read_fasta(fasta_path)
        # Could cache tokenizer in future versions
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.cp = cp
        self.shard_idx = int(os.path.splitext(os.path.basename(fasta_path))[0].split("_")[1])


    
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
            # Could CP the tokenized seq, but that would make begin and end tokens in the middle of the sequence
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

        

def generate_train_dataloader(train_dataset, args):
    """
    Create DataLoaders for training
    """
    train_sampler = DistributedSampler(train_dataset)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,   # replaces shuffle=True
        pin_memory=True,
        prefetch_factor=2,
        num_workers=2,
        persistent_workers=True
    )
    
    return train_loader



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

def training_loop(args, model, special_ids, mask_token_id):
    """
    Training loop for fine-tuning ESM on circularly-permuted sequences
    Now validates and saves checkpoints every 100K batches instead of every epoch
    """
    
    # Move model to the GPUs (process group already initialized in main)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    #line not needed; sets GPU for all tasks, but this is already done in main()
    #torch.cuda.set_device(local_rank)
    
    print(f"RANK {dist.get_rank()} | GPU {torch.cuda.current_device()}", flush=True)
    
    #awkward thing with not starting a bunch of simultaneous runs
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
    dist.barrier()
    
    model.to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # Define optimizer; give the optimizer the model parameters and learning rate as inputs. 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Initialize training states
    
    epoch = 0
    optimizer_step_count = 0
    total_batches_processed = 0
    shard_start = 0
    
    running_loss = 0.0
    #EMA smoothing factor
    alpha = 0.01
    
    log_interval = args.log_interval  # Log every N optimizer steps
    
    # Handle resuming from checkpoint
    if args.resume_checkpoint:
        # SINGLE GPU LOGGING: only rank 0 prints resume info
        if dist.get_rank() == 0:
            print(f"Resuming training from: {args.resume_checkpoint}")
        training_state = resume_training_from_checkpoint(args.resume_checkpoint, optimizer)
        shard_start = training_state.get('shard_idx',0)+1
        epoch = training_state.get('epoch', 0)
        total_batches_processed = training_state.get('total_batches_processed', 0)
        optimizer_step_count = training_state.get('optimizer_step_count', 0)
        # SINGLE GPU LOGGING: only rank 0 prints resume state
        if dist.get_rank() == 0:
            print(f"Resuming from shard {shard_start}, epoch {epoch}, batch {total_batches_processed}")
            
    checkpoint_dir = args.checkpoint_dir
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)        
    
    # SINGLE GPU LOGGING: only rank 0 creates the checkpoint directory
    
    
    # Put in a shuffle later for each epoch
    shard_files = get_shard_files(args.shards)
    
    for shard in range(shard_start, len(shard_files)):
        
        shard_start_time = time.time()
        
        shard_path = shard_files[shard] 
        
        if dist.get_rank() == 0:
            print('> Loading Dataset:', flush=True)
        
        if shard > shard_start:
            del shard_dataset, train_loader
            torch.cuda.empty_cache()
        
        shard_dataset = UniRef50Dataset_Shard(
        fasta_path=shard_path,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length
        )
        
        train_loader = generate_train_dataloader(shard_dataset, args)
        
        train_loader.sampler.set_epoch(epoch)
        
        if dist.get_rank() == 0:
            print('> Loading DataLoaders:', flush=True)
        
    
        # Set model to train mode
        model.train()
        
        
        # Loop through the batches in the dataloader
        for step, batch in enumerate(train_loader):
            total_batches_processed += 1            
            
            # SINGLE GPU LOGGING: only rank 0 prints progress and loss
            if dist.get_rank() == 0:
                if total_batches_processed % log_interval == 0:
                    print(f'Shard {shard}, Epoch {epoch}, Total Batches Processed: {total_batches_processed}, Current Shard Batches Processed: {step + 1}/{len(train_loader)}')
            
            
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
            true_loss_value = outputs.loss.item()
            # if running_loss == 0.0:
            #     running_loss = true_loss_value
            # else:
            #     running_loss = (1 - alpha) * running_loss + alpha * true_loss_value
            running_loss = true_loss_value
    
            # Step optimizer every N accumulation_steps
            if (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

                # SINGLE GPU LOGGING: only rank 0 logs to wandb
                if dist.get_rank() == 0:
                    if optimizer_step_count % log_interval == 0:
                        wandb.log({
                            "running_train_loss": running_loss,
                            "global_batch": total_batches_processed,
                            "optimizer_step": optimizer_step_count,
                            "shard_idx": shard,
                            "epoch": epoch + 1
                        }, step=optimizer_step_count)
        #wait for all gpus to finish shard
        dist.barrier()
        if dist.get_rank() == 0:
        # SINGLE GPU LOGGING: only rank 0 saves checkpoints
            ### Add code to remove all previous checkpoints to free up disk memory.
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_shard_{shard}_batch_{total_batches_processed}.pt")
            torch.save({
                'epoch': epoch + 1,
                'shard_idx' : shard,
                'total_batches_processed': total_batches_processed,
                'optimizer_step_count': optimizer_step_count,
                'model_state_dict': model.module.state_dict(),  # .module unwraps DDP
                'optimizer_state_dict': optimizer.state_dict(),
                'running_train_loss': running_loss,
                'args': args,
                'config': dict(config) if hasattr(config, 'items') else config
            }, checkpoint_path)
            print(f"✓ Regular checkpoint saved: {checkpoint_path}")
            
        # Remove previous checkpoints to free disk memory
        if dist.get_rank() == 0:  # Only rank 0 manages cleanup
            try:
                checkpoint_files = sorted(
                    [f for f in os.listdir(checkpoint_dir) 
                     if f.startswith(f"checkpoint_shard") and f.endswith('.pt')],
                    key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x))
                )

                # Keep only the latest checkpoint (customize NUM_CHECKPOINTS_TO_KEEP as needed)
                NUM_CHECKPOINTS_TO_KEEP = 2
                if len(checkpoint_files) > NUM_CHECKPOINTS_TO_KEEP:
                    for old_checkpoint in checkpoint_files[:-NUM_CHECKPOINTS_TO_KEEP]:
                        old_path = os.path.join(checkpoint_dir, old_checkpoint)
                        os.remove(old_path)
                        print(f"Removed old checkpoint: {old_checkpoint}")

            except Exception as e:
                print(f"Error during checkpoint cleanup: {e}")
                    
                    
        dist.barrier()
        # SINGLE GPU LOGGING: only rank 0 prints and logs epoch summary
        if dist.get_rank() == 0:
            shard_time = time.time() - shard_start_time
            print(f"\nShard {shard} Summary:")
            print(f"  Time: {shard_time/3600:.2f} hours")
            print(f"  Avg Loss: {running_loss:.4f}")
            print(f"  Total Batches So Far: {total_batches_processed:,}")
            
            wandb.log({
                "shard": shard,
                "shard_time_hours": shard_time / 3600,
                "running_loss": running_loss,
                "total_batches_processed": total_batches_processed,
                "epoch": epoch 
            }, step=optimizer_step_count)



def wandb_run(args):
    run = wandb.init(
        entity="aidenkzj",  # Change to your W&B entity
        project="esm_cp_finetune_shards",  # Change to your project name
        name=args.run_name,  # Set run name dynamically
        config={
            "accumulation_steps": 8,
            "learning_rate": 1e-4,
            "architecture": "ESM-MLM",
            "dataset": "circularly-permuted-sequences",
            "epochs": 10000,
            "mask_prob": 0.15,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "optimizer": "AdamW",
            "loss_fn": "CrossEntropyLoss(ignore_index=-100)",
        },
    )
    return run

def main(args):
    # Initialize distributed process group before anything else so
    # dist.get_rank() is available throughout main and training_loop
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
     
    dist.init_process_group(backend="nccl")
# switch above line to: dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    # SINGLE GPU LOGGING: only rank 0 prints setup progress
    if dist.get_rank() == 0:
        print('> Loading Model, Tokenizer:', flush=True)
        
    model = setup_model(args.model_name, args.resume_checkpoint)
    
    if dist.get_rank() == 0:
        print('> Getting special tokens:', flush=True)
        
    special_ids, mask_token_id = get_special_token_ids(args.tokenizer_name)
    special_ids_tensor = torch.tensor(special_ids)
    
    if dist.get_rank() == 0:
        print('> Starting Training Loop:', flush=True)
    training_loop(args, model, special_ids_tensor, mask_token_id)
    
    
    
    if dist.get_rank() == 0:
        print('> Finetuning Workflow Complete.', flush=True)
    dist.destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune ESM model on circularly permuted sequences.")
    
    parser.add_argument("--shards", type=str, default = "/orcd/home/002/aidenkzj/uniref50_length512_shards", help="Path to UniRef50 shards")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/esm2_t30_150M_UR50D", help="Hugging Face ESM tokenizer name")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D", help="Hugging Face ESM model name")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--run_name", type=str, default="esm_cp_ft_test_run", help="Name for W&B run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch")
    
    # Checkpoint loading arguments
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from (loads optimizer state too)")
    parser.add_argument("--checkpoint_dir", type=str, default="/orcd/home/002/aidenkzj/esm/esm/cp_finetuning/checkpoints", help="Directory to save checkpoints")
    
    # Validation and logging intervals
    parser.add_argument("--log_interval", type=int, default=5, help="Print progress every N optimizer steps (default: 50)")
    

    args = parser.parse_args()
    main(args)
