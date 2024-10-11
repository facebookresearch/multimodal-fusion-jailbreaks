# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import load_dataset
import torch
import json
from tqdm import trange
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
from .evaluation import judge_jailbreakbench


@torch.no_grad()
def generate_gcg(
    processor: ChameleonProcessor,
    model: ChameleonForConditionalGeneration,
    output_path: str,
    batch_size=16,
    gcg_suffix="!" * 10,
    split="test",
    output_filename="jbb_completions.json",
    single_prompt=None,
):
    if single_prompt is None:
        dataset = load_dataset("javirandor/jbb-chameleon7b", split=split)

        behaviors = dataset["Behavior"]
        prompts = [p + f" {gcg_suffix}" for p in dataset["Goal"]]
    else:
        prompts = [single_prompt + f" {gcg_suffix}"]
        behaviors = [None]

    # Initialize empty list to store results
    results = []

    # Calculate number of batches
    num_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )

    # Perform batched inference
    for i in trange(num_batches):
        # Get current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_behaviors = behaviors[start_idx:end_idx]
        batch_prompts = prompts[start_idx:end_idx]

        # Generate completions for current batch
        inputs = processor(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        inputs = inputs.to(model.device)

        output = model.generate(
            **inputs, max_new_tokens=250, use_shortcut=False, do_sample=False
        )

        generations = [
            processor.decode(out, skip_special_tokens=False)
            .split("<reserved08706>")[-1]
            .split("</s>")[0]
            for out in output
        ]

        # Store results for current batch
        batch_results = [
            {
                "behavior": b,
                "prompt": p.replace(f" {gcg_suffix}", "") if gcg_suffix != "" else p,
                "generation": g,
                "gcg_suffix": gcg_suffix,
            }
            for b, p, g in zip(batch_behaviors, batch_prompts, generations)
        ]
        results.extend(batch_results)

    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results stored in {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jailbreakbench script")
    parser.add_argument(
        "--generation_model",
        type=str,
        required=False,
        default="facebook/chameleon-7b",
        help="Model handle",
    )
    parser.add_argument(
        "--gcg_logfile",
        type=str,
        required=False,
        default=None,
        help="Path to GCG logfile",
    )
    parser.add_argument(
        "--gcg_suffix",
        type=str,
        required=False,
        default="",
        help="GCG suffix to append to prompt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test/",
        help="Directory where output files will be saved",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name for the generations file. By default it will be jbb_completions(_30b).json",
    )
    args = parser.parse_args()

    output_filename = args.output_filename

    processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

    causal_shortcut_model = ChameleonForConditionalGeneration.from_pretrained(
        args.generation_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if output_filename is None:
        output_filename = (
            "jbb_completions.json"
            if "7" in args.generation_model
            else "jbb_completions_30b.json"
        )

    if args.gcg_logfile is not None:
        with open(args.gcg_logfile, "r") as f:
            gcg_logfile = json.load(f)

        gcg_suffix = gcg_logfile["controls"][-1]
    else:
        gcg_suffix = args.gcg_suffix

    assert gcg_suffix is not None, "Could not load gcg suffix"

    # Create generations
    os.makedirs(args.output_path, exist_ok=True)
    completions_test = generate_gcg(
        processor=processor,
        model=causal_shortcut_model,
        output_path=args.output_path,
        output_filename=output_filename,
        gcg_suffix=gcg_suffix,
        split="test",
    )

    completions_train = generate_gcg(
        processor=processor,
        model=causal_shortcut_model,
        output_path=args.output_path,
        output_filename=output_filename.replace(".json", "_train.json"),
        gcg_suffix=gcg_suffix,
        split="train",
    )

    # Judge completions
    del causal_shortcut_model
    torch.cuda.empty_cache()

    eval_completions = judge_jailbreakbench(
        completions=completions_test,
        output_path=args.output_path,
        output_filename=output_filename.replace(".json", "_judge.json"),
    )

    eval_completions_train = judge_jailbreakbench(
        completions=completions_train,
        output_path=args.output_path,
        output_filename=output_filename.replace(".json", "_train_judge.json"),
    )
