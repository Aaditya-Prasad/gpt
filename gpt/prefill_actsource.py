import argparse
from pathlib import Path
import json
from typing import Literal, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

from gpt.interface import DataInterface

from research.cajal.activations.interfaces import ActivationSourceInterface
from research.cajal.activations_v2.utils.process_model_activations import process_model_activations

BOS_TOKEN_ID = 199998
PAD_TOKEN_ID = 199999
START_TOKEN_ID = 200006

"""
Point this file at processed tokens from generate.py and it will prefill the model over the tokens and save activations

Processed outputs should be in a dir like output_dir/exp_name with structure:

output_dir/exp_name
    tokens/
    data.json
    config.yaml
    prompts_and_completions.json

This file will add a third subdir to the output dir called "activations". activations will have subdirs for the different layers collected, e.g.

output_dir/exp_name/activations/
    layer_0/
    layer_1/
    ...
    layer_n/

Each layer subdir will have a file called "activations.npy" which is a numpy array of shape (num_tokens, head_dim)

Activations which are -1 are padded activations. 
"""

class ActivationSourceFromRollouts(ActivationSourceInterface):
    def __init__(self, 
                 data_path: Path, 
                 model: AutoModelForCausalLM, 
                 layer: int=15,
                 llm_batch_size: int=32,
                 max_samples: int=None):
        self.dataset = DataInterface(data_path)
        self.model = model
        self.layer = layer
        self.llm_batch_size = llm_batch_size
        self.i = 0
        self.buffer = torch.empty(0, self.model.config.hidden_size, device=self.model.device)
        # Store the actual maximum number of samples
        self.max_samples = max_samples if max_samples is not None else self.dataset.data("num_prompts")

    def get_batch(self, batch_size: int, return_sources: bool = False, toks_to_mask=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get a batch of activations from the source.
        
        Args:
            batch_size: The number of samples to get
            return_sources: Whether to return indexing information (not used, returns None)
            toks_to_mask: Tokens to mask (not used in this implementation)
            
        Returns:
            Tuple of (activations, row_indices, col_indices)
            Since we don't track source indices, returns (activations, None, None)
        """
        # populate buffer until we have enough activations
        while self._buffer_size() < batch_size:
            # Calculate actual batch size to avoid going past dataset end
            batch_start = self.i
            batch_end = min(self.i + self.llm_batch_size, self.max_samples)
            actual_batch_size = batch_end - batch_start
            
            if actual_batch_size <= 0:  # No more data
                break
                
            batch = self.dataset.load(uids = list(range(batch_start, batch_end)), load_type="tokens")
            token_ids = batch["tokens"]
            # Fix numpy array warning - convert list of arrays to single array first
            if isinstance(token_ids, list):
                token_ids = np.array(token_ids)
            token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.model.device)
            mask = token_ids == PAD_TOKEN_ID
            
            # Fix 1: Properly access hidden states from model output
            with torch.no_grad():
                outputs = self.model(token_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            
            # Fix 2: Properly flatten and mask activations
            # hidden_states[self.layer] is (batch_size, seq_len, hidden_dim)
            layer_activations = hidden_states[self.layer]
            # Flatten batch and sequence dimensions, then apply mask
            flat_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
            flat_mask = mask.reshape(-1)
            valid_activations = flat_activations[~flat_mask]
            
            self.buffer = torch.cat([self.buffer, valid_activations])
            
            # Fix 4: Increment by actual batch size processed
            self.i += actual_batch_size
            
        acts_to_return = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        
        # Return v2 interface format: (activations, row_indices, col_indices)
        # We don't track source indices, so return None for both
        return acts_to_return, None, None

    def _buffer_size(self) -> int:
        return self.buffer.shape[0]

def main(args):
    output_path = Path(args.data_path) / args.exp_name
    ds = DataInterface(output_path)
    N = ds.data("num_prompts")
    model_name = ds.data("model") if args.model == "auto" else "openai/gpt-oss-" + args.model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        # Use standard attention implementation - will use sdpa (scaled dot product attention) if available
        attn_implementation="sdpa",
    )
    model.eval()

    # Fix 3: Calculate the actual total number of non-padded tokens
    # First, find the actual number of available samples using binary search
    print("Finding actual dataset size...")
    actual_N = N
    
    # Quick check if the metadata is correct
    try:
        _ = ds.load(uids=[N-1], load_type="tokens")
        # If this works, metadata is correct
    except IndexError:
        # Binary search to find actual size
        left, right = 0, N - 1
        while left <= right:
            mid = (left + right) // 2
            try:
                _ = ds.load(uids=[mid], load_type="tokens")
                left = mid + 1  # mid is valid, search higher
            except IndexError:
                right = mid - 1  # mid is invalid, search lower
        actual_N = left  # left will be the first invalid index
        print(f"Actual dataset size: {actual_N} (metadata says {N})")
    
    print("Calculating total number of non-padded tokens...")
    total_valid_tokens = 0
    for i in tqdm(range(0, actual_N, args.batch_size)):
        batch_end = min(i + args.batch_size, actual_N)
        try:
            batch = ds.load(uids=list(range(i, batch_end)), load_type="tokens")
            token_ids = batch["tokens"]
            # Fix numpy array warning - convert list of arrays to single array first
            if isinstance(token_ids, list):
                token_ids = np.array(token_ids)
            token_ids = torch.tensor(token_ids, dtype=torch.int64)
            # Count non-padded tokens
            valid_mask = token_ids != PAD_TOKEN_ID
            total_valid_tokens += valid_mask.sum().item()
        except IndexError as e:
            print(f"Warning: Could not load batch starting at {i}: {e}")
            break
    print(f"Total non-padded tokens: {total_valid_tokens}")

    source = ActivationSourceFromRollouts(
        data_path=output_path, 
        model=model, 
        layer=args.layer, 
        llm_batch_size=args.batch_size,
        max_samples=actual_N
    )

    process_model_activations(
        # Required arguments
        activations_source=source,      # Your activation source, adheres to ActivationSourceInterface
        save_dir=output_path / "activations", # Where to save
        d_model=model.config.hidden_size,                      # Activation dimension
        
        # Size configuration
        total_size=total_valid_tokens,    # Total activations (actual number of non-padded tokens)
        n_shards=100,                      # Number of shards to create
        
        # Processing configuration
        batch_size=256,                    # Batch size for source
        # NOTE: I didn't check what this batch size does.
        buffer_size=50000,                 # Activations to buffer before flush
        max_workers=8,                     # Parallel threads for writing
        
        # Storage options
        save_indices=False,                # Don't save row/col indices since we don't track them
        dtype=np.float32,                  # Data type for storage
        
        # Partitioning
        partition_key=0,                   # Partition identifier
        
        # Dynamic sizing (when total_size=None)
        initial_shard_size=100000,         # Starting size per shard
        shard_growth_factor=2.0,           # Growth multiplier
        max_shard_size=10_000_000,         # Maximum shard size
        
        # Other options
        # shuffle=True,                      # Shuffle within shards
        # save_config=True,                  # Save configuration
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data dir", default="../outputs/")
    parser.add_argument("--exp_name", help="Experiment name", default="tmp")
    parser.add_argument("--model", help="Model name", default="auto") # auto, 20b, 120b
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=15) 
    args = parser.parse_args()

    main(args)