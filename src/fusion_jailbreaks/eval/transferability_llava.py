# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import load_dataset
import torch
import json
from tqdm import trange, tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import numpy as np


@torch.no_grad()
def generate_llava(
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    output_path: str,
    batch_size=16,
    split="test",
    filter_prompts: list = None,
    output_filename="jbb_completions.json",
    adv_image_path="",
):
    dataset = load_dataset("javirandor/jbb-chameleon7b", split=split)

    if filter_prompts is not None:
        dataset = dataset.select(filter_prompts)

    behaviors = dataset["Behavior"]
    prompts = [
        f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for p in dataset["Goal"]
    ]

    # Initialize empty list to store results
    results = []

    # Calculate number of batches
    num_batches = len(prompts) // batch_size + (
        1 if len(prompts) % batch_size != 0 else 0
    )

    adv_image = torch.load(adv_image_path).squeeze(0)
    # Return to [0, 255] space by undoing normalization
    adv_image = Image.fromarray(
        np.uint8(((adv_image.permute(1, 2, 0) + 1) / 0.0078).float().numpy())
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
            images=[adv_image] * len(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        inputs = inputs.to(model.device)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        generations = [
            processor.decode(out, skip_special_tokens=False)
            .split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            .split("<|eot_id|>")[0]
            .strip()
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


@torch.no_grad()
def generate_llava_gcg(
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    output_path: str,
    batch_size=16,
    split="test",
    filter_prompts: list = None,
    output_filename="jbb_completions.json",
    gcg_suffix="",
):
    dataset = load_dataset("javirandor/jbb-chameleon7b", split=split)

    if filter_prompts is not None:
        dataset = dataset.select(filter_prompts)

    behaviors = dataset["Behavior"]
    prompts = [
        f"<|start_header_id|>user<|end_header_id|>\n\n{p} {gcg_suffix}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for p in dataset["Goal"]
    ]

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

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        generations = [
            processor.decode(out, skip_special_tokens=False)
            .split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            .split("<|eot_id|>")[0]
            .strip()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jailbreakbench script")
    parser.add_argument(
        "--generation_model",
        type=str,
        required=True,
        help="Path to generation model",
    )
    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained(args.generation_model)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.generation_model, device_map="auto", torch_dtype=torch.float16
    )

    # For all single prompts
    PATHS = [
        "../../../data/results_single_prompt/vocab",
        "../../../data/results_single_prompt/embed",
    ]
    for PATH in PATHS:
        for folder in tqdm(os.listdir(PATH)):
            results_folder = os.path.join(PATH, folder)
            generations_folder = os.path.join(results_folder, "generations")

            prompt_id = int(folder.split("_")[0].replace("prompt", ""))

            completions = generate_llava(
                processor=processor,
                model=model,
                output_path=generations_folder,
                batch_size=1,
                adv_image_path=os.path.join(results_folder, "images/BEST"),
                split="test",
                output_filename="jbb_completions_{}.json".format(
                    args.generation_model.split("/")[-1]
                ),
                filter_prompts=[prompt_id],
            )

    # For transferability
    PATHS = [
        "../../../data/results_universal",
    ]
    for PATH in PATHS:
        for folder in tqdm(os.listdir(PATH)):
            if "noprevent" not in folder or "onlysure" in folder:
                continue

            results_folder = os.path.join(PATH, folder)
            generations_folder = os.path.join(results_folder, "generations")

            completions = generate_llava(
                processor=processor,
                model=model,
                output_path=generations_folder,
                batch_size=1,
                adv_image_path=os.path.join(results_folder, "images/BEST"),
                split="test",
                output_filename="jbb_completions_{}.json".format(
                    args.generation_model.split("/")[-1]
                ),
            )

            completions = generate_llava(
                processor=processor,
                model=model,
                output_path=generations_folder,
                adv_image_path=os.path.join(results_folder, "images/BEST"),
                batch_size=1,
                split="train",
                output_filename="jbb_completions_train_{}.json".format(
                    args.generation_model.split("/")[-1]
                ),
            )
