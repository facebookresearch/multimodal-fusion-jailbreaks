# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import load_dataset
from jailbreakbench.classifier import (
    Llama3JailbreakJudge,
    Llama3RefusalJudge,
)
import torch
import json
from tqdm import trange
from utils import load_json, chunk
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor


@torch.no_grad()
def generate(
    processor: ChameleonProcessor,
    model: ChameleonForConditionalGeneration,
    output_path: str,
    batch_size=16,
    split="test",
    use_shortcut=False,
    filter_prompts: list = None,
    output_filename="jbb_completions.json",
    adv_image_path="",
):
    dataset = load_dataset("javirandor/jbb-chameleon7b", split=split)

    if filter_prompts is not None:
        dataset = dataset.select(filter_prompts)

    behaviors = dataset["Behavior"]
    prompts = [p + "<image>" for p in dataset["Goal"]]

    # Initialize empty list to store results
    results = []

    # Calculate number of batches
    num_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )

    adv_image = torch.load(adv_image_path).squeeze(0)

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
            images=[torch.rand(3, 512, 512)] * len(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        inputs["pixel_values"] = adv_image.expand(len(batch_prompts), -1, -1, -1)

        inputs = inputs.to(model.device)

        output = model.generate(
            **inputs, max_new_tokens=200, use_shortcut=use_shortcut, do_sample=False
        )

        generations = [
            processor.decode(out, skip_special_tokens=False)
            .split("<reserved08706>")[-1]
            .split("</s>")[0]
            for out in output
        ]

        # Store results for current batch
        batch_results = [
            {"behavior": b, "prompt": p, "generation": g}
            for b, p, g in zip(batch_behaviors, batch_prompts, generations)
        ]
        results.extend(batch_results)

    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results stored in {output_file}")
    return results


def judge_jailbreakbench(
    completions: list | None,
    output_path: str,
    completions_path: str | None = None,
    output_filename: str = "jbb_completions_judge.json",
):
    assert (
        completions is not None or completions_path is not None
    ), "You must provide completions or a path to load them from"

    if completions is None:
        completions = load_json(completions_path)

    # Judge violations
    jailbreak_judge = Llama3JailbreakJudge(api_key=os.environ["TOGETHER_API_KEY"])

    batches = chunk(completions, 1)

    eval_completions = []

    for batch in batches:
        is_jailbroken = jailbreak_judge(
            [p["prompt"] for p in batch], [p["generation"] for p in batch]
        )

        eval_batch = [{**c, "is_jailbroken": ib} for c, ib in zip(batch, is_jailbroken)]
        eval_completions.extend(eval_batch)

    # Judge refusals
    refusal_judge = Llama3RefusalJudge(api_key=os.environ["TOGETHER_API_KEY"])

    batches = chunk(eval_completions, 1)
    final_eval_completions = []
    for batch in batches:
        is_refusal = refusal_judge(
            [p["prompt"] for p in batch], [p["generation"] for p in batch]
        )
        final_batch = [{**c, "is_refusal": ir} for c, ir in zip(batch, is_refusal)]
        final_eval_completions.extend(final_batch)
    eval_completions = final_eval_completions

    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "w") as f:
        json.dump(eval_completions, f, indent=4)

    return eval_completions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jailbreakbench script")
    parser.add_argument(
        "--generation_model",
        type=str,
        required=True,
        help="Path to generation model",
    )
    parser.add_argument(
        "--adv_image_path",
        type=str,
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        required=False,
        default=None,
        help="Path to precomputed generations, only judges will be executed",
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use: train or test",
    )
    args = parser.parse_args()

    output_filename = args.output_filename

    if args.generations_path is None:
        processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        dtype = torch.bfloat16

        causal_shortcut_model = ChameleonForConditionalGeneration.from_pretrained(
            args.generation_model,
            device_map="auto",
            torch_dtype=dtype,
        )

        if output_filename is None:
            output_filename = (
                "jbb_completions.json"
                if "7" in args.generation_model
                else "jbb_completions_30b.json"
            )

        # Create generations
        os.makedirs(args.output_path, exist_ok=True)
        completions = generate(
            processor=processor,
            model=causal_shortcut_model,
            output_path=args.output_path,
            adv_image_path=args.adv_image_path,
            split=args.split,
            output_filename=output_filename,
        )

    # Judge completions
    del causal_shortcut_model
    torch.cuda.empty_cache()

    if output_filename is None:
        output_filename = (
            "jbb_completions_30b.json"
            if "30" in args.generation_model
            else "jbb_completions.json"
        )

    eval_completions = judge_jailbreakbench(
        completions=completions,
        output_path=args.output_path,
        output_filename=output_filename.replace(".json", "_judge.json"),
    )
