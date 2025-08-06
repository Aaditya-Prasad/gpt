import argparse
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

from gpt.interface import DataInterface
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



def main(args):
    output_path = Path(args.data_path) / args.exp_name
    ds = DataInterface(output_path)
    N = ds.data("num_prompts")
    model_name = ds.data("model") if args.model == "auto" else "openai/gpt-oss-" + args.model

    layers = [int(l) for l in args.layers.split(",")]
    layer_activations = {k: [] for k in layers}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="kernels-community/vllm-flash-attn3",
    )

    model.eval()

    for i in range(0, N, args.batch_size):
        batch = ds.load(uids = list(range(i, min(i + args.batch_size, N))), load_type="tokens")
        token_ids = batch["tokens"]
        token_ids = torch.tensor(token_ids, dtype=torch.int64, device=model.device)
        mask = token_ids == PAD_TOKEN_ID
        with torch.no_grad():
            outputs, _, activations = model(token_ids, output_hidden_states=True)

            for layer in layers:
                layer_activations[layer].append(activations[layer])


    # Create activations directory
    activations_dir = output_path / "activations"
    activations_dir.mkdir(exist_ok=True)
    for layer in layers:
        activations_dir = activations_dir / f"layer_{layer}"
        activations_dir.mkdir(exist_ok=True)
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data dir", default="../outputs/")
    parser.add_argument("--exp_name", help="Experiment name", default="tmp")
    parser.add_argument("--model", help="Model name", default="auto") # auto, 20b, 120b
    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)
    parser.add_argument("-l", "--layers", type=str, default="15") # 15 for the 20b by default
    args = parser.parse_args()

    main(args)