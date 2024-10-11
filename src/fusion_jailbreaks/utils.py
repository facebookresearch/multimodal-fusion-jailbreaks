# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
import torch.nn.functional as F
from datasets import load_dataset
import os


def find_root_path():
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Name of the root directory to find
    root_dir_name = "fusion-jailbreaks"

    # Traverse up until the root directory is found
    while os.path.basename(current_dir) != root_dir_name:
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # This condition checks if we have reached the filesystem root
            raise Exception(f"Root directory named '{root_dir_name}' not found.")
        current_dir = parent_dir

    return current_dir


def chunk(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def prepare_train_dataset(
    num_prompts: int = 10,
    filter_train_prompts: list | None = None,
    use_longer_target: bool = False,
    image_first: bool = False,
    train_split: str = "train",
):
    dataset = load_dataset("javirandor/jbb-chameleon7b", split=train_split)

    if filter_train_prompts is not None:
        dataset = dataset.select(filter_train_prompts)
    elif num_prompts <= len(dataset):
        dataset = dataset.select([i for i in range(num_prompts)])
    else:
        raise Exception(
            "The training dataset has {} entries and you asked for {}".format(
                len(dataset), num_prompts
            )
        )

    prompts = (
        ["<image>" + p for p in dataset["Goal"]]
        if image_first
        else [p + "<image>" for p in dataset["Goal"]]
    )
    targets = (
        [t + ":\n\n" for t in dataset["Target"]]
        if use_longer_target
        else dataset["Target"]
    )

    return prompts, targets


def prepare_inputs(
    processor,
    prompts,
    target_prefixes=[],
    prevent_prefixes=[],
    prevent_recovery=[],
    dtype=torch.bfloat16,
):
    """
    Given prompts and prefixes model should generate, creates the input_ids and labels for forward pass.
    """
    targets = [
        processor.tokenizer.encode(target_str, add_special_tokens=False)
        for target_str in target_prefixes
    ]

    prevent_tokens = [
        processor.tokenizer.encode(target_str, add_special_tokens=False)
        for target_str in prevent_prefixes
    ]

    prevent_recovery = [
        processor.tokenizer.encode(target_str, add_special_tokens=False)
        for target_str in prevent_recovery
    ]

    input_ids = []
    target_labels = []
    prevent_labels = []

    for i in range(len(prompts)):
        recover_tokens = prevent_recovery[i] if len(prevent_recovery) > 0 else []
        input_ids.append(prompts[i]["input_ids"][0] + targets[i] + recover_tokens)

        # Set labels to -100 for input part and target ids for target part
        if len(targets) > 0:
            target_labels.append(
                [-100] * len(prompts[i]["input_ids"][0])
                + targets[i]
                + [-100] * len(recover_tokens)
            )

        if len(prevent_tokens) > 0 and len(prevent_recovery) > 0:
            prevent_labels.append(
                [-100] * len(prompts[i]["input_ids"][0])
                + prevent_tokens[i]
                + [-100] * (len(targets[i]) - len(prevent_tokens[i]))
                + recover_tokens
            )
        elif len(prevent_tokens) > 0:
            prevent_labels.append(
                [-100] * len(prompts[i]["input_ids"][0])
                + prevent_tokens[i]
                + [-100] * (len(targets[i]) - len(prevent_tokens[i]))
            )
        elif len(prevent_recovery) > 0:
            prevent_labels.append(
                [-100] * len(prompts[i]["input_ids"][0])
                + [-100] * (len(targets[i]))
                + recover_tokens
            )

    inputs = processor.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")

    def prepare_labels(inputs, labels):
        if len(labels) == 0:
            return labels

        labels = processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        ).input_ids

        labels = torch.where(
            (inputs["attention_mask"] == 0) | (labels == -100),
            -100,
            labels,
        )

        return labels

    target_labels = prepare_labels(inputs, target_labels)
    prevent_labels = prepare_labels(inputs, prevent_labels)

    inputs["pixel_values"] = torch.stack(
        [torch.tensor(i["pixel_values"][0]) for i in prompts]
    ).to(dtype)

    if len(prevent_labels) == 0:
        prevent_labels = None

    return inputs, target_labels, prevent_labels


def compute_logprobs(logits, targets):
    """
    Given logits and targets, compute logprobs for target sequence
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()

    targets = shift_targets.to(shift_logits.device)
    log_probabilities = F.log_softmax(shift_logits, dim=-1)
    # Step 2: Create a mask for valid target positions (where targets are not -100)
    valid_targets_mask = shift_targets != -100
    # Use the mask to filter out invalid targets
    valid_targets = shift_targets[valid_targets_mask]
    # Gather the log probabilities of the valid targets
    batch_indices, sequence_indices = torch.where(valid_targets_mask)
    gathered_log_probabilities = log_probabilities[
        batch_indices, sequence_indices, valid_targets
    ]
    # Step 3: Sum the log probabilities for each entry in the batch
    # First, create a tensor to store the sums, initialized to zero
    log_prob_sums = torch.zeros(shift_logits.size(0), device=shift_logits.device)
    # Use index_add to add the gathered log probabilities to the corresponding positions in log_prob_sums
    log_prob_sums.index_add_(
        0, batch_indices.to(shift_logits.device), gathered_log_probabilities
    )
    # Now log_prob_sums contains the sum of the log probabilities for each entry in the batch
    return log_prob_sums


def compute_loss(logits, labels, reduction="mean"):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, logits.shape[-1])
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Failed to parse JSON in file {file_path}.")
        return None
