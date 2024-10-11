# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import load_dataset
import torch
import json
from tqdm import trange
from utils import load_json, compute_loss
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor


def chunk(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


@torch.no_grad()
def generate_jailbreakbench(
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

        output = model(**inputs, use_shortcut=False)

        losses = compute_loss(output["logits"], inputs["input_ids"], reduction="none")

        # Store results for current batch
        batch_results = [
            {"behavior": b, "prompt": p, "perplexity": torch.exp(loss).item()}
            for b, p, loss in zip(batch_behaviors, batch_prompts, losses)
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
        completions = generate_jailbreakbench(
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
