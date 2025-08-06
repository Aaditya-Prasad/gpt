import argparse
from pathlib import Path
import json
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

from gpt.interface import DataInterface

from research.cajal.activations.interfaces import ActivationSourceInterface
from research.cajal.activations_v2.utils.process_model_activations import process_model_activations

PAD_TOKEN_ID = 199999

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
                 llm_batch_size: int=32):
        self.dataset = DataInterface(data_path)
        self.model = model
        self.layer = layer
        self.i = 0
        self.buffer = torch.empty(0, self.model.config.hidden_size, device=self.model.device)

    def get_batch(self, n_samples: int) -> torch.Tensor:
        # populate buffer until we have enough activations
        while self._buffer_size() < n_samples:
            batch = self.dataset.load(uids = list(range(self.i, min(self.i + self.llm_batch_size, self.dataset.data("num_prompts")))), load_type="tokens")
            token_ids = batch["tokens"]
            token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.model.device)
            mask = token_ids == PAD_TOKEN_ID
            with torch.no_grad():
                _, _, activations = self.model(token_ids, output_hidden_states=True)
            self.buffer = torch.cat([self.buffer, activations[self.layer][~mask]])
            self.i += self.llm_batch_size
        acts_to_return = self.buffer[:n_samples]
        self.buffer = self.buffer[n_samples:]
        return acts_to_return

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
        attn_implementation="kernels-community/vllm-flash-attn3",
    )
    model.eval()

    source = ActivationSourceFromRollouts(
        data_path=output_path, 
        model=model, 
        layer=args.layer, 
        llm_batch_size=args.batch_size
    )

    process_model_activations(
        # Required arguments
        activations_source=source,      # Your activation source, adheres to ActivationSourceInterface
        save_dir=output_path / "activations", # Where to save
        d_model=model.config.hidden_size,                      # Activation dimension
        
        # Size configuration
        total_size=N,             # Total activations (None for dynamic)
        n_shards=100,                      # Number of shards to create
        
        # Processing configuration
        batch_size=256,                    # Batch size for source
        # NOTE: I didn't check what this batch size does.
        buffer_size=50000,                 # Activations to buffer before flush
        max_workers=8,                     # Parallel threads for writing
        
        # Storage options
        save_indices=True,                 # Save row/col indices
        dtype=np.float32,                  # Data type for storage
        
        # Partitioning
        partition_key=0,                   # Partition identifier
        
        # Dynamic sizing (when total_size=None)
        initial_shard_size=100000,         # Starting size per shard
        shard_growth_factor=2.0,           # Growth multiplier
        max_shard_size=10_000_000,         # Maximum shard size
        
        # Other options
        shuffle=True,                      # Shuffle within shards
        save_config=True,                  # Save configuration
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data dir", default="../outputs/")
    parser.add_argument("--exp_name", help="Experiment name", default="tmp")
    parser.add_argument("--model", help="Model name", default="auto") # auto, 20b, 120b
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=15) 
    args = parser.parse_args()

    main(args)