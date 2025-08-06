import argparse
from pathlib import Path
import json
from typing import Literal, Optional, Tuple, List, Dict, Any

import os
os.environ['HF_HOME'] = '/mnt/polished-lake/home/emichaud/.cache/huggingface'
# Then set others based on HF_HOME  
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ['HF_HOME']

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
                 forward_batch_size: int=32,
                 max_samples: int=None):
        self.dataset = DataInterface(data_path)
        self.model = model
        self.layer = layer
        self.forward_batch_size = forward_batch_size
        self.i = 0  # index into documents, not tokens
        self.buffer = torch.empty(0, self.model.config.hidden_size, device=self.model.device)

    def get_batch(self, 
                  batch_size: int, 
                  return_sources: bool = False, 
                  toks_to_mask: List[int] = [PAD_TOKEN_ID],
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get a batch of activations from the source.
        
        Args:
            batch_size: The number of samples to get
            return_sources: Whether to return indexing information (not used, returns None)
            toks_to_mask: Tokens to mask 
            
        Returns:
            Tuple of (activations, row_indices, col_indices)
            Since we don't track source indices, returns (activations, None, None)
        """
        if return_sources:
            raise NotImplementedError("return_sources is not implemented")
        
        while len(self.buffer) < batch_size:
            batch = self.dataset.load(uids=list(range(self.i, self.i + self.forward_batch_size)), load_type="tokens")
            token_ids = torch.tensor(batch["tokens"], dtype=torch.int64, device=self.model.device)
            
            mask = token_ids < 0
            for tok_id in toks_to_mask:
                mask = mask | (token_ids == tok_id)
            
            with torch.no_grad():
                outputs = self.model(token_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            
            layer_activations = hidden_states[self.layer]
            flat_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
            flat_mask = mask.reshape(-1)
            valid_activations = flat_activations[~flat_mask]

            self.buffer = torch.cat([self.buffer, valid_activations])
            self.i += self.forward_batch_size
            
        acts_to_return = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]

        return acts_to_return, None, None  # No source indices


def main(args):
    output_path = Path(args.data_path) / args.exp_name
    ds = DataInterface(output_path)
    N = ds.data("num_prompts") # N is the number of PROMPTS in the dataset, not the number of tokens
    model_name = ds.data("model") if args.model == "auto" else "openai/gpt-oss-" + args.model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="kernels-community/vllm-flash-attn3",
    )
    model.eval()

    source = ActivationSourceFromRollouts(
        data_path=output_path, 
        model=model, 
        layer=args.layer, 
        forward_batch_size=args.forward_batch_size,
    )

    process_model_activations(
        # Required arguments
        activations_source=source,             # Your activation source, adheres to ActivationSourceInterface
        save_dir=output_path / "activations",  # Where to save
        d_model=model.config.hidden_size,      # Activation dimension
        total_size=args.total_tokens,          # Go through dataset until exhausted
        n_shards=100,                          # Number of shards to create
        batch_size=256,                        # Batch size for source
        buffer_size=50000,                     # Activations to buffer before flush
        max_workers=8,                         # Parallel threads for writing
        save_indices=False,                    # Since not supported
        tokens_to_mask=[PAD_TOKEN_ID],         # Tokens to mask
        dtype=np.float32,                      # Data type for storage
        partition_key=0,                       # Partition identifier
        initial_shard_size=100000,             # Starting size per shard
        shard_growth_factor=2.0,               # Growth multiplier
        max_shard_size=10_000_000,             # Maximum shard size
        # shuffle=True,                        # Shuffle within shards
        # save_config=True,                    # Save configuration
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../outputs/", help="Data directory")
    parser.add_argument("--exp_name", default="tmp", help="Experiment name")
    parser.add_argument("--model", default="auto", choices=["auto", "20b", "120b"], help="Model")
    parser.add_argument("--total_tokens", type=int, default=None, help="Number of tokens to save")
    parser.add_argument("--forward_batch_size", type=int, default=8, help="Forward batch size")
    parser.add_argument("--layer", type=int, default=15, help="Layer index")
    args = parser.parse_args()

    main(args)