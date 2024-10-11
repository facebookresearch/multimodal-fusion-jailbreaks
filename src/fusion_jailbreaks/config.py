# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import dataclasses


@dataclasses.dataclass
class ConfigClass:
    model_path: str = "facebook/chameleon-7b"
    tokenizer_path: str = "facebook/chameleon-7b"
    shortcut_type: str = "vocab"
    train_split: str = "train"
    test_split: str | None = "test"
    results_folder: str = "data/results/"
    lr_decay: bool = True
    shortcut_checkpoint: str = ""
    softmax_temperature: float = 1.0
    target_prefixes: List[str] | None = None
    prevent_prefixes: bool = False
    prevent_recovery: bool = False
    prevent_recovery_str: str = "I"
    prevent_prefixes_str: str = "I"
    prompts: List[str] = dataclasses.field(default_factory=list)
    image: int | str = 2
    num_iterations: int = 1000
    starting_lr: float = 0.01
    use_pgd: bool = False
    pgd_ball: float = 0.1
    prevent_loss_weight: float = 0.1
    experiment_name: str = ""
    config_file: str = ""
    use_sign: bool = True
    dtype: Optional[str] = None
    only_top_grads: Optional[float] = None
    run_eval: bool = False
    num_train_prompts: int = 12
    use_embed_reg: bool = False
    use_longer_target: bool = False
    image_first: bool = False
    filter_train_prompts: list = None
    # Deprecated
    model_size: str = "7b"
