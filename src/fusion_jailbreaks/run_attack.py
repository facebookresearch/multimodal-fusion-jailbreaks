# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import datetime
import os
import sys
import requests
from PIL import Image

import torch
import yaml
from datasets import load_dataset

import wandb
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
)

from config import ConfigClass
from utils import (
    prepare_inputs,
    compute_logprobs,
    compute_loss,
    prepare_train_dataset,
    find_root_path,
)
from eval import generate, judge_jailbreakbench
import shutil


def main(config, config_file):
    date = str(datetime.datetime.now())

    ROOT_PATH = find_root_path()
    RESULTS_PATH = os.path.join(
        ROOT_PATH, config.results_folder, config.experiment_name
    )

    IMAGE_SAVE_PATH = os.path.join(RESULTS_PATH, "images/")
    GENERATIONS_SAVE_PATH = os.path.join(RESULTS_PATH, "generations/")

    # if os.path.exists(os.path.join(IMAGE_SAVE_PATH, "BEST")):
    #     raise Exception("There is already a completed experiment for this config")

    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
    os.makedirs(GENERATIONS_SAVE_PATH, exist_ok=True)

    # Copy config file
    shutil.copyfile(config_file, os.path.join(RESULTS_PATH, config_file.split("/")[-1]))

    processor = ChameleonProcessor.from_pretrained(config.tokenizer_path)

    # In our code, we used a custom split without images containing people.
    # For simplicity, we load this split here. Note it may contain people images.
    data = load_dataset(
        "pufanyi/MIMICIT",
        "CGD_Images",
    )

    dtype = torch.bfloat16

    causal_shortcut_model = ChameleonForConditionalGeneration.from_pretrained(
        config.model_path,
        device_map="auto",
        torch_dtype=dtype,
        shortcut_type=config.shortcut_type,
        softmax_temperature=config.softmax_temperature,
    )

    causal_shortcut_model.model.shortcut.load_state_dict(
        torch.load(
            config.shortcut_checkpoint,
            map_location="cuda:{}".format(torch.cuda.device_count() - 1),
        ),
        assign=True,
    )

    causal_shortcut_model.model.shortcut.to(
        device="cuda:{}".format(torch.cuda.device_count() - 1), dtype=dtype
    )

    causal_shortcut_model.model.vqmodel.eval()
    causal_shortcut_model.eval()

    for param in causal_shortcut_model.parameters():
        param.requires_grad = False

    for param in causal_shortcut_model.model.parameters():
        param.requires_grad = False

    if isinstance(config.image, str) and "http" in config.image:
        image = Image.open(
            requests.get(
                config.image,
                stream=True,
            ).raw
        )
    elif config.image == "random":
        image = data["train"][0]["image"]
    else:
        image = data["train"][int(config.image)]["image"]

    # GET SEQUENCES
    processor.tokenizer.padding_side = "left"

    # Prepare input_ids and labels
    orig_prompts, target_prefixes = prepare_train_dataset(
        config.num_train_prompts,
        config.filter_train_prompts,
        config.use_longer_target,
        config.image_first,
        config.train_split,
    )
    if config.target_prefixes is not None:
        target_prefixes = target_prefixes
        assert len(target_prefixes) == len(
            orig_prompts
        ), "Your target prefixes do not match your prompts"

    prompts = [
        processor(text=p, images=image, return_tensors=None) for p in orig_prompts
    ]

    inputs, target_labels, prevent_labels = prepare_inputs(
        processor,
        prompts,
        target_prefixes,
        prevent_prefixes=[]
        if config.prevent_prefixes is False
        else [config.prevent_prefixes_str] * len(prompts),
        prevent_recovery=[]
        if config.prevent_recovery is False
        else [config.prevent_recovery_str] * len(prompts),
    )

    processor.tokenizer.padding_side = "left"
    # END GET SEQUENCES

    if config.use_embed_reg:
        with torch.no_grad():
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            _, original_vq_embeds, original_shortcut_embeddings = causal_shortcut_model(
                **inputs.to(causal_shortcut_model.device),
                use_shortcut=True,
                use_cache=False,
                output_vqembeds=True,
            )

    embed_norm = torch.nn.CosineEmbeddingLoss()

    num_iterations = config.num_iterations
    learning_rate = config.starting_lr
    best_loss = 1000000

    wandb_config = {
        **dataclasses.asdict(config),
        "config_file": config_file,
    }
    wandb_config["prompts"] = orig_prompts
    wandb_config["targets"] = target_prefixes

    wandb.init(
        # set the wandb project where this run will be logged
        project="JBBChameleon",
        name=config.experiment_name + "_" + date,
        # track hyperparameters and run metadata
        config=wandb_config,
    )

    adv_image = torch.tensor(prompts[0]["pixel_values"]).clone().detach().to(dtype)

    if config.image == "random":
        adv_image = torch.rand_like(adv_image).clone().detach()

    if config.use_pgd:
        starting_image = adv_image.clone()

    for iteration in range(1, num_iterations + 1):
        # Decay in learning rate
        if iteration % 120 == 0 and config.lr_decay:
            learning_rate = max(learning_rate / 2, config.starting_lr / 10)

        def adv_forward_pass(
            model, inputs, labels, adv_image, use_shortcut=True, output_vqembeds=False
        ):
            inputs["pixel_values"] = adv_image.expand(
                inputs["input_ids"].shape[0], -1, -1, -1
            )
            inputs["labels"] = labels

            output = model(
                **inputs.to(model.device),
                use_shortcut=use_shortcut,
                use_cache=False,
                output_vqembeds=output_vqembeds,
            )

            if output_vqembeds:
                return output[0], output[1], output[2]
            return output, None, None

        adv_image.requires_grad = True
        if adv_image.grad is not None:
            adv_image.grad.zero_()

        output_target, vq_embeds, shortcut_embeddings = adv_forward_pass(
            causal_shortcut_model,
            inputs,
            target_labels,
            adv_image,
            output_vqembeds=True if config.use_embed_reg else False,
        )

        loss = -output_target.loss

        if config.use_embed_reg:
            embedpenalty = embed_norm(
                vq_embeds[:, :, :].reshape(-1, vq_embeds.shape[-1]),
                original_vq_embeds[:, :, :].reshape(-1, original_vq_embeds.shape[-1]),
                torch.ones(
                    vq_embeds[:, :, :].reshape(-1, vq_embeds.shape[-1]).shape[0]
                ).to(vq_embeds.device),
            )

            loss -= embedpenalty

        del vq_embeds
        del shortcut_embeddings

        output_prevent_loss = None
        if prevent_labels is not None:
            loss_prevent = compute_loss(output_target.logits, prevent_labels)
            loss += loss_prevent * config.prevent_loss_weight
            output_prevent_loss = loss_prevent.item()

        grad = torch.autograd.grad(loss, adv_image)[0].clone().detach()

        output_target_logits = output_target.logits.detach()
        output_target_loss = output_target.loss.item()

        target_logprobs = compute_logprobs(output_target_logits, target_labels)
        prevent_logprobs = (
            compute_logprobs(output_target.logits, prevent_labels)
            if prevent_labels is not None
            else None
        )

        del output_target

        # Compute probs under quantized model
        with torch.no_grad():
            output_quant, _, _ = adv_forward_pass(
                causal_shortcut_model,
                inputs,
                target_labels,
                adv_image,
                use_shortcut=False,
            )
            target_logprobs_quant = compute_logprobs(output_quant.logits, target_labels)

            quant_loss = output_quant.loss.item()

            if prevent_labels is not None:
                prevent_logprobs_quant = compute_logprobs(
                    output_quant.logits, prevent_labels
                )

                loss_prevent_quant = compute_loss(output_quant.logits, prevent_labels)
                quant_loss -= loss_prevent_quant.item()

        del output_quant

        total_loss = (
            output_target_loss
            if output_prevent_loss is None
            else output_target_loss - output_prevent_loss
        )

        wandb.log(
            {
                "target-loss": output_target_loss,
                "prevent-loss": output_prevent_loss,
                "total-loss": total_loss,
                "quant-loss": quant_loss,
                # "l2-loss": l2penalty,
                "target_logprobs/min": target_logprobs.min(),
                "target_logprobs/mean": target_logprobs.mean(),
                "target_logprobs/max": target_logprobs.max(),
                "prevent_logprobs/min": prevent_logprobs.min()
                if prevent_labels is not None
                else None,
                "prevent_logprobs/mean": prevent_logprobs.mean()
                if prevent_labels is not None
                else None,
                "prevent_logprobs/max": prevent_logprobs.max()
                if prevent_labels is not None
                else None,
                "target_logprobs_quant/min": target_logprobs_quant.min(),
                "target_logprobs_quant/max": target_logprobs_quant.max(),
                "target_logprobs_quant/mean": target_logprobs_quant.mean(),
                "prevent_logprobs_quant/min": prevent_logprobs_quant.min()
                if prevent_labels is not None
                else None,
                "prevent_logprobs_quant/max": prevent_logprobs_quant.max()
                if prevent_labels is not None
                else None,
                "prevent_logprobs_quant/mean": prevent_logprobs_quant.mean()
                if prevent_labels is not None
                else None,
                "lr": learning_rate,
            }
        )

        save_loss = quant_loss if config.shortcut_type == "vocab" else total_loss

        if save_loss < best_loss:
            torch.save(
                adv_image.cpu().detach(),
                os.path.join(IMAGE_SAVE_PATH, "BEST"),
            )
            best_loss = quant_loss

        if iteration % 100 == 0:
            print("Storing at step {}".format(iteration))
            torch.save(
                adv_image.cpu().detach(),
                os.path.join(IMAGE_SAVE_PATH, "step{}_auto".format(iteration)),
            )

        # Update image with average gradients
        with torch.no_grad():
            if config.only_top_grads is not None:
                # Calculate the gradient magnitude
                grad_magnitude = torch.abs(grad)
                # Determine the threshold for the top N% of gradients
                k = int(
                    config.only_top_grads * grad_magnitude.numel()
                )  # N% of the total number of elements
                threshold = torch.kthvalue(
                    grad_magnitude.view(-1), k=grad_magnitude.numel() - k + 1
                ).values
                # Create a mask for the top 20% gradients
                mask = grad_magnitude >= threshold
                # Calculate update direction based on the mask
                update_direction = mask.to(dtype)
            else:
                mask = torch.ones_like(grad).to(dtype)

            # Apply mask
            grad = mask * grad

            if config.use_sign is True:
                update_direction = learning_rate * grad.sign()
            else:
                update_direction = grad

            adv_image = adv_image.detach() + update_direction

            # Project to epsilon ball
            if config.use_pgd:
                perturbation_mask = torch.clamp(
                    adv_image - starting_image,
                    min=-config.pgd_ball,
                    max=config.pgd_ball,
                )
                adv_image = starting_image + perturbation_mask

            adv_image = torch.clamp(adv_image, min=-1, max=1).detach()

        del grad
        torch.cuda.empty_cache()

    torch.save(adv_image, os.path.join(IMAGE_SAVE_PATH, "FINAL.pth"))

    train_filter = None
    if config.filter_train_prompts is not None:
        train_filter = config.filter_train_prompts
    elif config.num_train_prompts is not None:
        train_filter = [i for i in range(config.num_train_prompts)]

    if config.test_split is not None:
        completions = generate(
            processor=processor,
            model=causal_shortcut_model,
            batch_size=1,
            split=config.test_split,
            output_path=GENERATIONS_SAVE_PATH,
            adv_image_path=os.path.join(IMAGE_SAVE_PATH, "BEST"),
            output_filename="jbb_completions_30b.json"
            if "30b" in config.model_path
            else "jbb_completions.json",
        )

        completions_shortcut = generate(
            processor=processor,
            model=causal_shortcut_model,
            output_path=GENERATIONS_SAVE_PATH,
            adv_image_path=os.path.join(IMAGE_SAVE_PATH, "BEST"),
            use_shortcut=True,
            split=config.test_split,
            batch_size=1,
            output_filename="jbb_completions_shortcut_30b.json"
            if "30b" in config.model_path
            else "jbb_completions_shortcut.json",
        )

    completions_train = generate(
        processor=processor,
        model=causal_shortcut_model,
        output_path=GENERATIONS_SAVE_PATH,
        batch_size=1,
        split=config.train_split,
        filter_prompts=train_filter,
        adv_image_path=os.path.join(IMAGE_SAVE_PATH, "BEST"),
        output_filename="jbb_completions_30b_train.json"
        if "30b" in config.model_path
        else "jbb_completions_train.json",
    )

    completions_train_shortcut = generate(
        processor=processor,
        model=causal_shortcut_model,
        output_path=GENERATIONS_SAVE_PATH,
        use_shortcut=True,
        split=config.train_split,
        filter_prompts=train_filter,
        batch_size=1,
        adv_image_path=os.path.join(IMAGE_SAVE_PATH, "BEST"),
        output_filename="jbb_completions_train_shortcut_30b.json"
        if "30b" in config.model_path
        else "jbb_completions_train_shortcut.json",
    )

    # Run evaluations
    if config.run_eval:
        if config.test_split is not None:
            eval_completions = judge_jailbreakbench(
                completions=completions,
                output_path=GENERATIONS_SAVE_PATH,
                output_filename="jbb_completions_30b_judge.json"
                if "30b" in config.model_path
                else "jbb_completions_judge.json",
            )

            eval_completions_shortcut = judge_jailbreakbench(
                completions=completions_shortcut,
                output_path=GENERATIONS_SAVE_PATH,
                output_filename="jbb_completions_shorcut_30b_judge.json"
                if "30b" in config.model_path
                else "jbb_completions_shortcut_judge.json",
            )

        eval_completions_train = judge_jailbreakbench(
            completions=completions_train,
            output_path=GENERATIONS_SAVE_PATH,
            output_filename="jbb_completions_train_30b_judge.json"
            if "30b" in config.model_path
            else "jbb_completions_train_judge.json",
        )

        eval_completions_train_shortcut = judge_jailbreakbench(
            completions=completions_train,
            output_path=GENERATIONS_SAVE_PATH,
            output_filename="jbb_completions_train_shortcut_30b_judge.json"
            if "30b" in config.model_path
            else "jbb_completions_train_shortcut_judge.json",
        )


if __name__ == "__main__":
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        print("Usage: python script.py config.yaml [image value]")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    if len(sys.argv) == 3:
        image_value = sys.argv[2]
        config_dict["image"] = image_value  # Overwrite the 'image' value
        config_dict["experiment_name"] = (
            config_file.split("/")[-1]
            .replace(".yaml", "")
            .replace("image2", f"image{image_value}")
        )

    config = ConfigClass(**config_dict)

    if config.experiment_name == "":
        config.experiment_name = config_file.split("/")[-1].replace(".yaml", "")

    main(config, config_file)
