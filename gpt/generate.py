import argparse
from dataclasses import dataclass
import yaml
from vllm import LLM, SamplingParams
from typing import Literal
import json
from pathlib import Path
from einops import rearrange
import numpy as np
from datetime import date

"""
Point this file at a dataset and it will spin up a vLLM server on this node, generate completions for each prompt, and save the tokens + outputs to disk


data.json will include the following keys:
- partition_size
- num_prompts
- model
"""

PAD_TOKEN_ID = 199999
BOS_TOKEN_ID = 199998

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.6
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = -1
    skip_special_tokens: bool = False

    def to_sampling_params(self, stop_token_ids):
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            skip_special_tokens=self.skip_special_tokens,
            stop_token_ids=stop_token_ids,
        )   

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

from typing import Iterable, Tuple, Literal, Optional

from openai_harmony import (
    load_harmony_encoding, HarmonyEncodingName,
    Conversation, Message, Role,
    SystemContent, ReasoningEffort
)

ReasoningStr = Literal["low", "medium", "high"]

def get_token_ids(
    prompt: str,
    encoding,
    reasoning: ReasoningStr = "low",
    *,
    knowledge_cutoff: str = "2024-06",
    required_channels: Iterable[str] = ("analysis", "commentary", "final"),
    date_str: Optional[str] = None,
) -> Tuple[list[int], list[int]]:
    """
    Build a minimal Harmony conversation (system + user) and render tokens to
    start assistant sampling. Returns (prefill_token_ids, stop_token_ids).

    `reasoning` can be "low" | "medium" | "high".
    """

    reasoning_map = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
    }
    effort = reasoning_map.get(str(reasoning).lower())
    if effort is None:
        raise ValueError("reasoning must be 'low', 'medium', or 'high'")

    if date_str is None:
        date_str = date.today().isoformat()  # consider passing your own clock

    system = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_knowledge_cutoff(knowledge_cutoff)
        .with_conversation_start_date(date_str)
        .with_reasoning_effort(effort)
        .with_required_channels(list(required_channels))
    )

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.USER, prompt),
    ])

    # Tokens to prefill the model before it starts generating as the assistant
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    return prefill_ids

def main(args):

    model = "openai/gpt-oss-" + args.model
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    llm = LLM(model=model,
              tensor_parallel_size=args.tp,
              max_seq_len_to_capture=3072,
              max_model_len=3072,
    )

    prompts = json.load(open(args.data_path))
    N = len(prompts)
    token_ids = [get_token_ids(p["prompt"], encoding, reasoning=args.reasoning) for p in prompts]


    gc = GenerationConfig() if args.config == "None" else GenerationConfig.from_yaml(args.config)
    sp = gc.to_sampling_params(stop_token_ids=stop_token_ids)

    outputs = llm.generate(
        prompt_token_ids=token_ids,
        sampling_params=sp,
    )
    
    prompts_and_completions = [
        {**p, "completion": o.outputs[0].text} for p, o in zip(prompts, outputs)
    ]

    token_ids = [
        input_toks + o.outputs[0].token_ids for input_toks, o in zip(token_ids, outputs) # Currently not prepending BOS_TOKEN_ID!!
    ]

    output_dir = Path(args.output_dir) / args.exp_name  
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "prompts_and_completions.json", "w") as f:
        json.dump(prompts_and_completions, f)

    with open(output_dir / "data.json", "w") as f:
        json.dump(
            {
                "model": model,
                "partition_size": args.partition_size,
                "num_prompts": N
            }, f)

    max_len = max(len(toks) for toks in token_ids)
    out_ids = np.full((N, max_len), PAD_TOKEN_ID, dtype=np.int64)
    for i, toks in enumerate(token_ids):
        out_ids[i, :len(toks)] = toks

    partitions = np.array_split(out_ids, N // args.partition_size + 1)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)


    for i, partition in enumerate(partitions):
        np.save(tokens_dir / f"tokens_{i}.npy", partition)

    gc.to_yaml(output_dir / "config.yaml")



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name", default="20b") # 20b or 120b
    parser.add_argument("--data_path", help="Path to data file", default="inputs/example.json")
    parser.add_argument("--output_dir", help="Path to output directory", default="../outputs")
    parser.add_argument("-e", "--exp_name", help="Experiment name", default="tmp")
    parser.add_argument("--config", help="Path to generation config YAML", default="None")
    parser.add_argument("-p", "--partition_size", help="Partition size", type=int, default=1024)
    parser.add_argument("-t", "--tp", help="Tensor parallel size", type=int, default=1)
    parser.add_argument("-r", "--reasoning", help="Reasoning effort", type=str, default="low")
    args = parser.parse_args()

    main(args)