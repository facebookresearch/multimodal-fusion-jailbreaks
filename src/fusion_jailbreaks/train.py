# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import datetime
import sys
from io import BytesIO
from typing import Optional, List
import os

import requests
import torch
import yaml
from datasets import load_dataset
from PIL import Image
from torch import nn
import wandb

from transformers import (
    ChameleonConfig,
    ChameleonForConditionalGeneration,
    ChameleonModel,
    ChameleonProcessor,
    ChameleonVQVAE,
)
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonImageVocabularyMapping,
)

from utils import find_root_path


@dataclasses.dataclass
class ConfigClass:
    model_size: str = "7b"
    shortcut_type: str = "vocab"
    batch_size: int = 64
    num_epochs: int = 1
    images_filter: List[int] | None = None
    starting_lr: float = 0.0007
    experiment_name: str = ""
    config_file: str = ""
    loss: str | None = None


def main(config: ConfigClass, config_file: str):
    date = str(datetime.datetime.now())
    ROOT_PATH = find_root_path()

    SAVE_PATH = os.path.join(
        ROOT_PATH, "data/models/", config.experiment_name + "_" + date
    )
    os.makedirs(SAVE_PATH, exist_ok=False)

    processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

    shortcut_model = ChameleonModel.from_pretrained(
        "facebook/chameleon-{}".format(config.model_size),
        device_map="auto",
        shortcut_type=config.shortcut_type,
    )

    del shortcut_model.layers
    torch.cuda.empty_cache()

    shortcut_model.vqmodel.eval()
    shortcut_model.shortcut.train()

    for param in shortcut_model.parameters():
        param.requires_grad = False

    for param in shortcut_model.shortcut.parameters():
        param.requires_grad = True

    # In our code, we used a custom split without images containing people.
    # For simplicity, we load this split here. Note it may contain people images.
    data = load_dataset(
        "pufanyi/MIMICIT",
        "CGD_Images",
    )

    if config.images_filter is not None:
        data["train"] = data["train"].select(config.images_filter)

    # Define the loss function and optimizer
    if "embed" in shortcut_model.shortcut_type:
        criterion = nn.MSELoss()
        if config.loss == "cosine":
            criterion = nn.CosineEmbeddingLoss()
    elif "vocab" in shortcut_model.shortcut_type:
        criterion = nn.CrossEntropyLoss()

    print("Training these weights:")
    for name, param in shortcut_model.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = torch.optim.Adam(
        shortcut_model.shortcut.parameters(), lr=config.starting_lr
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project="AdvChameleonTrain",
        name=config.experiment_name + "_" + date,
        # track hyperparameters and run metadata
        config={
            **dataclasses.asdict(config),
            "config_file": config_file,
        },
    )

    iteration = 0

    # Loop through the dataset
    for epoch in range(config.num_epochs):
        for i in range(0, len(data["train"]), config.batch_size):
            try:
                # for i in range(1000):
                # Get the current batch of images
                batch = data["train"][
                    i : min(i + config.batch_size, len(data["train"]))
                ]
                # Convert the images to tensors and move them to the device
                pixel_values = processor.image_processor(
                    images=batch["image"], return_tensors="pt"
                )["pixel_values"].to(shortcut_model.device)

                with torch.no_grad():
                    vq_embeds, image_tokens = shortcut_model.get_image_tokens(
                        pixel_values
                    )

                if "embed" in shortcut_model.shortcut_type:
                    shortcut = shortcut_model.shortcut(vq_embeds)
                    shortcut = shortcut.view(-1, shortcut.shape[-1])
                    with torch.no_grad():
                        target = shortcut_model.embed_tokens(image_tokens)
                        target = target.view(-1, target.shape[-1])

                elif "vocab" in shortcut_model.shortcut_type:
                    shortcut = shortcut_model.shortcut(vq_embeds).view(
                        -1, shortcut_model.vocab_size
                    )
                    target = image_tokens.flatten().long()

                # Calculate the loss
                if config.loss == "cosine":
                    loss = criterion(
                        shortcut,
                        target,
                        torch.ones(shortcut.shape[0]).to(shortcut.device),
                    )
                else:
                    loss = criterion(
                        shortcut,
                        target,
                    )

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch": epoch,
                    },
                    step=iteration,
                )

                # Print the loss at each iteration
                print(f"Loss at iteration {iteration}: {loss.item()}", flush=True)

                iteration += 1
                if iteration % 100 == 0:
                    torch.save(
                        shortcut_model.shortcut.state_dict(),
                        os.path.join(SAVE_PATH, f"{iteration}.pth"),
                    )

            except Exception as e:
                print("Error for i={}: ".format(i), e)
                pass

    # Store the shortcut linear layer at the end
    torch.save(
        shortcut_model.shortcut.state_dict(), os.path.join(SAVE_PATH, "FINAL.pth")
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_shortcut.py config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    config = ConfigClass(**config_dict)

    config.experiment_name = config_file.split("/")[-1].replace(".yaml", "")
    print(config)

    main(config, config_file)
