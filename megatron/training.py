# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from its original version
#

"""Pretrain utilities."""
import time
from datetime import datetime
from functools import partial

import math
import sys

import torch
import deepspeed
import numpy as np
import wandb

import logging

logging.basicConfig(level=logging.WARNING)

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
)

from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)
from megatron.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    load_data_sampling_weights,
    load_local_samples_seen_per_dataset,
    load_global_samples_seen_per_dataset
)
from megatron.data.data_utils import build_train_valid_test_data_iterators
from megatron.initialize import initialize_megatron, _set_random_seed
from megatron.learning_rates import AnnealingLR
from megatron.logging import tb_wandb_log, training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    get_total_params,
    CharCounter,
)
from megatron.model.gpt2_model import cross_entropy, cross_entropy_per_sample
from eval_tasks import run_eval_harness
from megatron.data.data_sampling_utils import get_data_sampling_weighter

import random

from megatron.data.agent import load_hds

REQUIRES_VALID_DATASET = [
    "naive_validation"
]


def pretrain(neox_args):
    """Main training program.

    This function will run the following in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model.

    Arguments:
        neox_args: an instance of NeoXArgs containing the configuration for pretrain

    """
    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(
        use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
    )

    # Initialize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args)

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        neox_args=neox_args, use_cache=False, iteration=neox_args.iteration
    )

    print("******************************MODLE**************************************************")
    print(model)
    print("******************************MODLE**************************************************")
    name_list = ["module.sequential.10"]
    sum_n = 0
    all_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_num += param.numel()
            param_count = param.numel()
            print(f"Layer name: {name}, Trainable params: {param_count}")
            for name_ in name_list:
                if name_ in name:
                    sum_n += param_count
                    print(f"Layer name: {name}, Trainable params: {param_count}")
    print(f"List Trainable params: {sum_n}")
    print(all_num)
    print("******************************MODLE**************************************************")
    timers("model and optimizer").stop()

    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
        dataset_names,
    ) = build_train_valid_test_data_iterators(neox_args=neox_args)
    timers("train/valid/test data iterators").stop()

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])
    print_rank_0("training ...")

    if neox_args.hds is not None:
        ac_agent, neox_args = load_hds(neox_args=neox_args, model=model)
        model = (model, ac_agent)

    iteration = neox_args.iteration
    data_sampling_weights = None
    if neox_args.do_train and neox_args.train_iters > 0:
        # edge case: save step 0 checkpoint if requested and we're starting from step 0
        if neox_args.save and 0 in neox_args.save_iters and iteration == 0:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        if neox_args.validation_based_reward:
            if neox_args.use_named_train_datasets and neox_args.mixed_batches:
                iteration = train_named_datasets_mixed_batch_validation_reward(
                    dataset_names=dataset_names,
                    neox_args=neox_args,
                    timers=timers,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_data_iterator=train_data_iterator,
                    valid_data_iterator=valid_data_iterator,
                )
            elif neox_args.use_named_train_datasets and neox_args.mixed_minibatches:
                iteration = train_named_datasets_mixed_minibatch_validation_reward(
                    dataset_names=dataset_names,
                    neox_args=neox_args,
                    timers=timers,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_data_iterator=train_data_iterator,
                    valid_data_iterator=valid_data_iterator,
                )
        else:
            if neox_args.mixed_minibatches:
                iteration, data_sampling_weights = train_mixed_minibatch(
                    neox_args=neox_args,
                    timers=timers,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_data_iterator=train_data_iterator,
                    valid_data_iterator=valid_data_iterator,
                )
            else:
                iteration = train(
                    neox_args=neox_args,
                    timers=timers,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_data_iterator=train_data_iterator,
                    valid_data_iterator=valid_data_iterator,
                )

    if neox_args.hds is not None:
        model, ac_agent = model
    if neox_args.do_valid:
        prefix = "the end of training for val data"
        if neox_args.use_named_eval_datasets:
            evaluate_named_datasets_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterators=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )
        else:
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )

    if neox_args.save and iteration != 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            data_sampling_weights=data_sampling_weights,
        )
        if neox_args.hds is not None:
            ac_agent.save_ckpts(neox_args, iteration)

    if neox_args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        if neox_args.use_named_eval_datasets:
            evaluate_named_datasets_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterators=test_data_iterator,
                model=model,
                iteration=iteration,
                verbose=True,
                timers=timers,
                chart_name="test",
            )
        else:
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterator=test_data_iterator,
                model=model,
                iteration=iteration,
                verbose=True,
                timers=timers,
                chart_name="test",
            )


# hds
def _get_batch(neox_args, tokenizer, keys, data, datatype):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    if neox_args.hds is not None:
        dataset_names = data_b["dataset_name"]
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=keys,
        data=data,
        datatype=datatype,
    )


# hds
def get_hds_mixed_batch(neox_args, data_iterators, weights=None, agent=None):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    data = []
    data_names = []
    for sub_domain, sub_iter in data_iterators.items():
        for _ in range(int(neox_args.hds["warmup_batch_size"])):
            data_ = next(sub_iter)
            data.append(data_["text"])
            data_names += [sub_domain] * data_["text"].shape[0]
    data = {"text": torch.cat(data, dim=0)}
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    if neox_args.hds is not None:
        dataset_names = data_b["dataset_name"]
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    dataset_names = data_names

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return dataset_names, tokens, labels, loss_mask, attention_mask, position_ids


def get_hds_batch_names(neox_args, weights=None, warmup=False):
    if neox_args.hds is not None:
        dataset_names = neox_args.hds["datasets_names"]
        domain_num = neox_args.hds["domain_num"]
    else:
        dataset_names = []
        for train_path in neox_args.train_data_paths:
            name = train_path.split("/")[-2]
            dataset_names.append(name)
        domain_num = 1
    batch_names = []
    if warmup:
        # if random.random() > 0.5:
        #     # # iterate over full batch size, then split to each device
        #     # if neox_args.gradient_accumulation_steps % len(dataset_names) != 0:
        #     #     raise ValueError("gradient_accumulation_steps must be a multiple of dataset_names!")
        #     times = neox_args.world_size * neox_args.gradient_accumulation_steps // len(dataset_names)
        #     for dataset_name in dataset_names:
        #         batch_names += [dataset_name] * times
        #     local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        #     weights_way = [1/22]*22
        # else:

        # for _ in range(domain_num):
        #     batch_names += dataset_names
        # if random.random() >= 0.05:
        #     random_values = np.random.rand(22)
        #     normalized_values = random_values / random_values.sum()
        #     weights = normalized_values.tolist()
        # batch_names += random.choices(dataset_names, weights=weights, k=int(neox_args.world_size * neox_args.gradient_accumulation_steps - len(dataset_names) * domain_num))
        # local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        # return local_batches, weights

        for _ in range(domain_num):
            batch_names += dataset_names
        # iterate over full batch size, then split to each device
        batch_names += random.choices(dataset_names, weights=weights, k=int(
            neox_args.world_size * neox_args.gradient_accumulation_steps - len(dataset_names) * domain_num))
        local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        return local_batches, weights

    else:
        for _ in range(domain_num):
            batch_names += dataset_names
        # iterate over full batch size, then split to each device
        batch_names += random.choices(dataset_names, weights=weights, k=int(
            neox_args.world_size * neox_args.gradient_accumulation_steps - len(dataset_names) * domain_num))
        local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        return local_batches


def get_hds_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    if neox_args.hds is not None:
        dataset_names = data_b["dataset_name"]
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_named_datasets_mixed_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    if neox_args.hds is not None:
        dataset_names = data_b["dataset_name"]
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    dataset_names = data["dataset_name"]

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return dataset_names, tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator."""
    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
        neox_args, neox_args.tokenizer, keys, data, datatype
    )

    # unpack data
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


# hds
def forward_step(data_iterator, model, neox_args, timers, return_logits=False):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        neox_args=neox_args, data_iterator=data_iterator
    )
    if timers is not None:
        timers("batch generator").stop()

    outputs = model((tokens, position_ids, attention_mask))

    loss = cross_entropy(
        outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
    )
    if return_logits:
        return loss, outputs

    if neox_args.hds is not None:
        return loss, tokens

    return loss


def get_model(neox_args, use_cache=False):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu.
    model = GPT2ModelPipe(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        use_cache=use_cache,
    )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
            "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_optimizer(model, neox_args):
    """Set up the optimizer."""
    if neox_args.no_load_optim:
        return None, None
    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model, neox_args)
    print_rank_0(
        f'Configuring Optimizer type: {neox_args.optimizer_type} with params: {neox_args.optimizer["params"]}'
    )

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False

    # Filter out params that don't require a grad (for soft prompt tuning, etc.)
    _param_groups = []
    for param_group in param_groups:
        trainable_params = [p for p in param_group["params"] if p.requires_grad]
        param_group["params"] = trainable_params
        _param_groups.append(param_group)
    param_groups = _param_groups

    if neox_args.optimizer_type.lower() in ["cpu_adam", "cpu_torch_adam"]:
        if neox_args.optimizer == "cpu_torch_adam":
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "onebitadam":
        assert neox_args.deepspeed
        optimizer = None
        # onebitadam needs to be instantiated within the deepspeed engine to work :|
    elif neox_args.optimizer_type.lower() == "sm3":
        from .optimizers import SM3

        optimizer = SM3(param_groups, **neox_args.optimizer["params"])
    elif neox_args.optimizer_type.lower() == "madgrad_wd":
        from .optimizers import madgrad_wd

        optimizer = madgrad_wd(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    elif neox_args.optimizer_type.lower() == "adam":
        # Use Adam
        if neox_args.use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                adam_optimizer = bnb.optim.Adam8bit
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            try:
                # default to apex as it's slightly faster
                from apex.optimizers import FusedAdam as Adam
            except ImportError:
                # if apex isn't installed, use deepspeed's FusedAdam
                print(
                    "WARNING: APEX not installed - defaulting to deepspeed's fused adam"
                )
                from deepspeed.ops.adam import FusedAdam as Adam
            adam_optimizer = Adam
        optimizer = adam_optimizer(
            param_groups,
            weight_decay=neox_args.weight_decay,
            **neox_args.optimizer["params"],
        )
    else:
        raise ValueError(f"Optimizer type {neox_args.optimizer_type} not recognized")

    if neox_args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer, param_groups
    else:
        raise ValueError("Must be using deepspeed to run neox")


def get_learning_rate_scheduler(optimizer, neox_args):
    """Build the learning rate scheduler."""
    if neox_args.no_load_optim:
        # TODO: this should be configured as a separate arg
        return None
    if neox_args.deepspeed and neox_args.optimizer_type.lower() == "onebitadam":
        print_rank_0(
            "WARNING: onebitadam requires the lr scheduler be built by deepspeed - "
            "Make sure one is added to your deepspeed config"
        )
        return None

    # Add linear learning rate scheduler.
    if neox_args.lr_decay_iters is not None:
        num_iters = neox_args.lr_decay_iters
    else:
        num_iters = neox_args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = neox_args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=neox_args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=neox_args.lr_decay_style,
        last_iter=init_step,
        min_lr=neox_args.min_lr,
        use_checkpoint_lr_scheduler=neox_args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=neox_args.override_lr_scheduler,
    )

    return lr_scheduler


def setup_model_and_optimizer(neox_args, use_cache=False, iteration=None):
    """Setup model and optimizer."""
    model = get_model(neox_args=neox_args, use_cache=use_cache)
    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if neox_args.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iteration=iteration,
        )
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    else:
        neox_args.iteration = 0

    return model, optimizer, lr_scheduler


def backward_step(neox_args, timers, optimizer, model, loss):
    """Backward step."""

    # Backward pass.
    timers("backward-backward").start()
    if neox_args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError("Must be using deepspeed to run neox")
    timers("backward-backward").stop()

    if neox_args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


def train_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
    else:
        losses = []
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            loss = forward_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=data_iterator,
                model=model,
            )
            timers("forward").stop()
            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            timers("backward").stop()
            # Update parameters.
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def train_step_pipe(neox_args, timers, model, data_iterator):
    """Single training step with DeepSpeed's pipeline parallel engine."""

    assert neox_args.deepspeed
    loss = model.train_batch(data_iter=data_iterator)
    loss_dict = {"lm_loss": loss}
    # Don't break Megatron's timers because we changed code paths.
    for t in [
        "forward",
        "backward",
        "allreduce",
        "optimizer",
        "batch generator",
        "data loader",
        "data sampling update"
    ]:
        timers(t).reset()
    return loss_dict


"""hds"""


def train_step_named_datasets_mixed_batch(neox_args, timers, data_iterator, model, optimizer, lr_scheduler, ac_agent,
                                          weights_sampler=None):
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        raise NotImplementedError("Mixed batch not supported with pipeline parallelism")
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
    else:
        """hds"""

        batch_dataset_names = []

        if neox_args.hds is not None:
            weights = ac_agent.warmup_weighter()
        else:
            weights = weights_sampler()
        local_batch_names, weights_way = get_hds_batch_names(neox_args, warmup=True, weights=weights)
        # local_batch_names = get_hds_batch_names(neox_args, warmup=True, weights=weights)

        losses = []

        for batch_name in local_batch_names:
            batch_iterator = data_iterator[batch_name]

            timers("forward").start()
            timers("batch generator").start()

            # hds

            tokens, labels, loss_mask, attention_mask, position_ids = get_hds_batch(neox_args, batch_iterator)

            timers("batch generator").stop()
            outputs = model((tokens, position_ids, attention_mask))

            # TODO: implement per_sample_loss for mixed batch
            """hds"""

            loss = cross_entropy(
                outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
            )

            losses.append(loss)

            timers("forward").stop()

            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()

            if neox_args.hds is not None:
                ac_agent.update_pools(neox_args=neox_args,
                                      timers=timers,
                                      model=model,
                                      loss=loss,
                                      domain_name=batch_name,
                                      tokens=tokens,
                                      weights=weights_way)
            else:
                backward_step(
                    neox_args=neox_args,
                    timers=timers,
                    optimizer=optimizer,
                    model=model,
                    loss=loss,
                )

            """hds"""
            timers("backward").stop()
            # Update parameters.
            timers("optimizer").start()

            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")

            timers("optimizer").stop()
            batch_dataset_names.append(batch_name)

        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean(),
        }  # reduces losses across machines for logging
    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return batch_dataset_names, reduced_loss, skipped_iter

    #     batch_dataset_names = []
    #     for _ in range(neox_args.gradient_accumulation_steps):
    #         # Forward model for one step.
    #         timers("forward").start()
    #         timers("batch generator").start()
    #
    #         cur_weights = ac_agent.get_weights()
    #
    #         # hds
    #         time1 = time.time()
    #         names, tokens, labels, loss_mask, attention_mask, position_ids = get_hds_mixed_batch(
    #             neox_args=neox_args, data_iterators=data_iterator, weights=cur_weights, agent=ac_agent
    #         )
    #
    #         print(names, type(names))
    #         time2 = time.time()
    #         print(f"Get Data Cost: {time.time() - time1} s")
    #         # names, tokens, labels, loss_mask, attention_mask, position_ids = get_named_datasets_mixed_batch(
    #         #     neox_args=neox_args, data_iterator=data_iterator
    #         # )
    #
    #
    #         timers("batch generator").stop()
    #         outputs = model((tokens, position_ids, attention_mask))
    #
    #         time3 = time.time()
    #         print(f"Forward Cost: {time.time() - time2} s")
    #
    #         # TODO: implement per_sample_loss for mixed batch
    #         """hds"""
    #         per_sample_loss = cross_entropy_per_sample(
    #             outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
    #         )
    #
    #         loss = per_sample_loss.mean()
    #
    #
    #         # loss = cross_entropy(
    #         #     outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
    #         # )
    #
    #         time4 = time.time()
    #         print(f"Loss computing Cost: {time.time() - time3} s")
    #
    #
    #
    #         timers("forward").stop()
    #
    #
    #
    #
    #         # Calculate gradients, reduce across processes, and clip.
    #         timers("backward").start()
    #
    #
    #         # backward_step(
    #         #     neox_args=neox_args,
    #         #     timers=timers,
    #         #     optimizer=optimizer,
    #         #     model=model,
    #         #     loss=loss,
    #         # )
    #
    #         if neox_args.hds is not None:
    #             ac_agent.update_pools(neox_args, timers, model, per_sample_loss, names)
    #         else:
    #             backward_step(
    #                 neox_args=neox_args,
    #                 timers=timers,
    #                 optimizer=optimizer,
    #                 model=model,
    #                 loss=loss,
    #             )
    #
    #         time5 = time.time()
    #         print(f"Backward Cost: {time.time() - time4} s")
    #
    #         # losses.append(reduce_losses([loss]).mean().clone().detach().cpu().item())
    #
    #         time6 = time.time()
    #         print(f"Reduce loss Cost: {time.time() - time5} s")
    #
    #         """hds"""
    #         timers("backward").stop()
    #         # Update parameters.
    #         timers("optimizer").start()
    #
    #         if neox_args.deepspeed:
    #             model.step()
    #         else:
    #             raise ValueError("Must be using deepspeed to run neox")
    #
    #         time7 = time.time()
    #         print(f"Model step Cost: {time.time() - time6} s")
    #
    #         timers("optimizer").stop()
    #         batch_dataset_names.extend(names)
    #
    #         print(f"Empty cache Cost: {time.time() - time7} s")
    #
    #     reduced_loss = {
    #         "lm_loss": np.array(losses).mean(),
    #     }  # reduces losses across machines for logging
    # if neox_args.precision == "fp16" and model.optimizer.overflow:
    #     skipped_iter = 1
    # else:
    #     skipped_iter = 0
    #
    # return batch_dataset_names, reduced_loss, skipped_iter


def train(
        neox_args,
        timers,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    # if using named train datasets, prepare dataloaders, dataiterators and dicts to track iterations and epochs
    if neox_args.use_named_train_datasets:
        neox_args.dataset_iterations = {name: 0 for name in train_data_iterator.keys()}
        train_dataloaders = {name: dataloader for name, dataloader in train_data_iterator.items()}
        train_data_iterator = {name: iter(dataloader) for name, dataloader in train_dataloaders.items()}
        neox_args.dataset_epochs = {name: dataloader.dataset._completed_epochs for name, dataloader in
                                    train_dataloaders.items()}
        dataset_names = list(train_data_iterator.keys())
        data_sampling_weights = get_data_sampling_weighter(
            dataset_names=list(train_dataloaders.keys()),
            weights=neox_args.train_data_weights,
            warmup_steps=neox_args.data_sampling_warmup_steps,
            update_frequency=neox_args.data_sampling_update_frequency,
            update_method=neox_args.data_sampling_method
        )
    else:
        data_sampling_weights = None

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)
    while iteration < neox_args.train_iters:

        if neox_args.use_named_train_datasets:
            # print(f"ITERATION: {iteration} -- RANK {torch.distributed.get_rank()} -- WEIGHT {neox_args.train_data_weights}")
            batch_name = random.choices(dataset_names, weights=data_sampling_weights(), k=1)[0]
            # print(f"ITERATION: {iteration} -- RANK {torch.distributed.get_rank()} USING DATASET {batch_name}")
            batch_iterator = train_data_iterator[batch_name]
        else:
            batch_iterator = train_data_iterator

        # if using named train datasets, allow for restarting a dataset
        if neox_args.use_named_train_datasets:
            try:
                loss_dict, skipped_iter = train_step(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
            except StopIteration:
                # stop timers from attempted iteration
                timers("forward").stop()
                timers("batch generator").stop()
                # print(f"ITERATION: {iteration} -- RANK {torch.distributed.get_rank()} RESTARTING DATASET {batch_name}")
                # reinitialize dataset
                train_dataloaders[batch_name].dataset._reinitialize()
                # update completed epochs
                neox_args.dataset_epochs[batch_name] = train_dataloaders[batch_name].dataset._completed_epochs
                # wait for all processes
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # create iterator from dataloader
                train_data_iterator[batch_name] = iter(train_dataloaders[batch_name])
                # continue with new batch
                batch_iterator = train_data_iterator[batch_name]
                loss_dict, skipped_iter = train_step(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
        else:
            loss_dict, skipped_iter = train_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=batch_iterator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
        iteration += 1

        # if using named train datasets, update the iteration count
        if neox_args.use_named_train_datasets:
            neox_args.dataset_iterations[batch_name] += 1

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # update data sampling weights
        if neox_args.use_named_train_datasets:
            reward = loss_dict["lm_loss"].item()
            timers("data sampling update").start()
            data_sampling_weights.update(iteration, **{"dataset_name": batch_name, "reward": reward})
            timers("data sampling update").stop()

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
            data_sampling_weights=data_sampling_weights
        )

        # Checkpointing
        if neox_args.save and iteration in neox_args.save_iters:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluation
        if (
                neox_args.eval_interval
                and iteration % neox_args.eval_interval == 0
                and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            if neox_args.use_named_eval_datasets:
                evaluate_named_datasets_and_print_results(
                    neox_args=neox_args,
                    prefix=prefix,
                    forward_step_func=forward_step,
                    data_iterators=valid_data_iterator,
                    model=model,
                    iteration=iteration,
                    verbose=False,
                    timers=timers,
                )
            else:
                evaluate_and_print_results(
                    neox_args=neox_args,
                    prefix=prefix,
                    forward_step_func=forward_step,
                    data_iterator=valid_data_iterator,
                    model=model,
                    iteration=iteration,
                    verbose=False,
                    timers=timers,
                )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

    return iteration


def train_mixed_minibatch(
        neox_args,
        timers,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
):
    """Train the model function."""

    if neox_args.hds is not None:
        model, ac_agent = model
    else:
        ac_agent = None
    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    assert (neox_args.use_named_train_datasets), "Mixed minibatch training requires named datasets"

    # Set up data sampling variables
    neox_args.dataset_iterations = {name: 0 for name in train_data_iterator.keys()}
    train_dataloaders = {name: dataloader for name, dataloader in train_data_iterator.items()}
    train_data_iterator = {name: iter(dataloader) for name, dataloader in train_dataloaders.items()}
    neox_args.dataset_epochs = {name: dataloader.dataset._completed_epochs for name, dataloader in
                                train_dataloaders.items()}
    dataset_names = list(train_data_iterator.keys())
    data_sampler_kwargs = {}
    if neox_args.data_sampling_method == "smoothed_mean":
        data_sampler_kwargs['smoothing_factor'] = neox_args.data_sampling_smoothing_factor

    if neox_args.hds is not None:
        data_sampling_weights = get_data_sampling_weighter(
            dataset_names=list(train_dataloaders.keys()),
            weights=neox_args.train_data_weights,
            warmup_steps=neox_args.data_sampling_warmup_steps,
            update_frequency=neox_args.data_sampling_update_frequency,
            update_method=neox_args.data_sampling_method,
            **data_sampler_kwargs
        )

        if neox_args.hds["warmup_iter"] > 0:
            data_sampler_kwargs_ = {}
            data_sampler_kwargs_['smoothing_factor'] = neox_args.data_sampling_smoothing_factor
            ac_agent.warmup_weighter = get_data_sampling_weighter(
                dataset_names=list(train_dataloaders.keys()),
                weights=neox_args.train_data_weights,
                warmup_steps=0,
                update_frequency=neox_args.data_sampling_update_frequency,
                update_method="smoothed_mean",
                **data_sampler_kwargs_
            )
    else:
        data_sampler_kwargs_ = {}
        data_sampler_kwargs_['smoothing_factor'] = neox_args.data_sampling_smoothing_factor
        data_sampling_weights = get_data_sampling_weighter(
            dataset_names=list(train_dataloaders.keys()),
            weights=neox_args.train_data_weights,
            warmup_steps=neox_args.data_sampling_warmup_steps,
            update_frequency=neox_args.data_sampling_update_frequency,
            update_method="smoothed_mean",
            **data_sampler_kwargs_
        )

    # if saving, need to specify number of iterations per device
    if neox_args.save:
        neox_args.local_samples_seen_per_dataset = {name: 0 for name in train_data_iterator.keys()}

    # if loading from checkpoint
    if neox_args.load:
        # get data sampling weights
        data_sampling_weights = load_data_sampling_weights(neox_args)

        # load local samples seen per dataset
        neox_args.local_samples_seen_per_dataset = load_local_samples_seen_per_dataset(neox_args)
        # iterate through datasets to correctly initialize the dataloaders
        for dataset_name, samples_seen in neox_args.local_samples_seen_per_dataset.items():
            local_batches = samples_seen / neox_args.train_micro_batch_size_per_gpu
            if not local_batches.is_integer():
                local_batches = math.ceil(local_batches)
                print(
                    f"Warning: Loaded samples seen per dataset on rank {torch.distributed.get_rank()} is not a multiple of micro batch size. "
                    f"Samples seen / train_micro_batch_size_per_gpu: {samples_seen} / {neox_args.train_micro_batch_size_per_gpu}. "
                    f"Rounding up to {local_batches}. "
                )

            # calculate how many epochs have been completed
            local_completed_epochs = int(local_batches // len(train_dataloaders[dataset_name]))
            # calculate how many batches need to be skipped in the current epoch
            local_start_iter = int(local_batches % len(train_dataloaders[dataset_name]))

            print(
                f"RANK {torch.distributed.get_rank()} -- Skipping {local_start_iter} batches in epoch {local_completed_epochs} of {dataset_name}")

            if local_completed_epochs > 0:
                train_dataloaders[dataset_name].dataset.seed += local_completed_epochs
                train_dataloaders[dataset_name].dataset._reinitialize(override_process_dataset=True)
                neox_args.dataset_epochs[dataset_name] = train_dataloaders[dataset_name].dataset._completed_epochs
                train_dataloaders[dataset_name].dataset._completed_epochs = local_completed_epochs

            train_dataloaders[dataset_name].batch_sampler.start_iter = local_start_iter
            train_data_iterator[dataset_name] = iter(train_dataloaders[dataset_name])

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # batch_iterator = train_data_iterator[dataset_name]
            # for _ in range(int(local_batches)):
            #     try:
            #         data = next(batch_iterator)
            #     except:
            #         # reinitialize dataset
            #         train_dataloaders[dataset_name].dataset.seed += 1
            #         train_dataloaders[dataset_name].dataset._reinitialize(override_process_dataset=True)
            #         # update completed epochs
            #         neox_args.dataset_epochs[dataset_name] = train_dataloaders[dataset_name].dataset._completed_epochs
            #         # create iterator from dataloader
            #         train_data_iterator[dataset_name] = iter(train_dataloaders[dataset_name])
            #         # continue with new batch
            #         batch_iterator = train_data_iterator[dataset_name]
            #         data = next(batch_iterator)

        # update global iteration count
        global_samples_seen_per_dataset = load_global_samples_seen_per_dataset(neox_args)
        for dataset_name, samples_seen in global_samples_seen_per_dataset.items():
            global_batches = samples_seen / neox_args.train_batch_size
            neox_args.dataset_iterations[dataset_name] = global_batches

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    # Data sampling warmup loop
    if iteration < neox_args.data_sampling_warmup_steps:
        ### hds
        if neox_args.hds is not None:
            ac_agent.warmup_flag = True

        # change dataset iterations to tensor to allow for distributed accumulation
        neox_args.dataset_iterations = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in
                                        train_data_iterator.keys()}

        # get mixed_batch data loaders
        neox_args.mixed_batches = True
        neox_args.mixed_minibatches = False
        # timers("train/valid/test data iterators").start()
        # (
        #     warmup_train_data_iterator,
        #     _,
        #     _,
        #     _,
        # ) = build_train_valid_test_data_iterators(neox_args=neox_args)
        # timers("train/valid/test data iterators").stop()

        # iterate through warmup steps

        # hds
        while iteration < neox_args.data_sampling_warmup_steps:

            print(f"-------------------------------Iteration {iteration}------------------------")
            time1 = time.time()

            # batch_dataset_names, loss_dict, skipped_iter = train_step_named_datasets_mixed_batch(
            #     neox_args=neox_args,
            #     timers=timers,
            #     data_iterator=warmup_train_data_iterator,
            #     model=model,
            #     optimizer=optimizer,
            #     lr_scheduler=lr_scheduler,
            # )
            if neox_args.hds is not None:
                batch_dataset_names, loss_dict, skipped_iter = train_step_named_datasets_mixed_batch(
                    neox_args=neox_args,
                    timers=timers,
                    # hds
                    data_iterator=train_data_iterator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ac_agent=ac_agent
                )
            else:
                batch_dataset_names, loss_dict, skipped_iter = train_step_named_datasets_mixed_batch(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=train_data_iterator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ac_agent=ac_agent,
                    weights_sampler=data_sampling_weights
                )
            iteration += 1

            print(f"Total Cost: {time.time() - time1} s")
            print(f"-----------------------------------------------------------------------------")

            for name in batch_dataset_names:
                neox_args.dataset_iterations[name.replace("train_", "")] += 1 / neox_args.train_batch_size
                neox_args.local_samples_seen_per_dataset[name.replace("train_", "")] += 1

            overflow_monitor.check(skipped_iter)  # check for repeated overflow
            if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
                noise_scale_logger.update()

            # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
            # may have no tunable parameters on a specific rank
            if optimizer.param_groups:
                lr = optimizer.param_groups[0].get("lr", 0)
            else:
                lr = 0

            # Logging.
            report_memory_flag = training_log(
                neox_args=neox_args,
                timers=timers,
                loss_dict=loss_dict,
                total_loss_dict=total_loss_dict,
                learning_rate=lr,
                iteration=iteration,
                loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
                report_memory_flag=report_memory_flag,
                skipped_iter=skipped_iter,
                model=model,
                optimizer=optimizer,
                noise_scale_logger=noise_scale_logger,
                data_sampling_weights=data_sampling_weights,
                ac_agent=ac_agent
            )

            # Checkpointing
            if neox_args.save and iteration in neox_args.save_iters:
                save_checkpoint(
                    neox_args=neox_args,
                    iteration=iteration,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
                if neox_args.hds is not None:
                    ac_agent.save_ckpts(neox_args, iteration)

            # Evaluation
            if (
                    neox_args.eval_interval
                    and iteration % neox_args.eval_interval == 0
                    and neox_args.do_valid
            ):
                prefix = "iteration {}".format(iteration)
                time2 = time.time()
                evaluate_named_datasets_and_print_results(
                    neox_args=neox_args,
                    prefix=prefix,
                    forward_step_func=forward_step,
                    data_iterators=valid_data_iterator,
                    model=model,
                    iteration=iteration,
                    verbose=True,
                    timers=timers,
                )
                print(f"---------------EVALUATION COST: {time.time() - time2} s----------------------------")

        # del warmup_train_data_iterator

        # reset seeds, dataloaders, and iterators
        _set_random_seed(neox_args.seed)
        neox_args.mixed_batches = False
        neox_args.mixed_minibatches = True
        # timers("train/valid/test data iterators").start()
        # (
        #     train_data_iterator,
        #     _,
        #     _,
        #     _,
        # ) = build_train_valid_test_data_iterators(neox_args=neox_args)
        # train_dataloaders = {name: dataloader for name, dataloader in train_data_iterator.items()}
        # train_data_iterator = {name: iter(dataloader) for name, dataloader in train_dataloaders.items()}
        # neox_args.dataset_epochs = {name: dataloader.dataset._completed_epochs for name, dataloader in train_dataloaders.items()}
        # timers("train/valid/test data iterators").stop()

        # gather dataset_iterations across processes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            for name in neox_args.dataset_iterations.keys():
                torch.distributed.all_reduce(neox_args.dataset_iterations[name])
                neox_args.dataset_iterations[name] = int(math.ceil(neox_args.dataset_iterations[name].tolist()))

        # iterate through datasets to correctly initialize the dataloaders
        # hds use the same loader in warmup and training
        # for dataset_name, samples_seen in neox_args.local_samples_seen_per_dataset.items():
        #     local_batches = samples_seen // neox_args.train_micro_batch_size_per_gpu
        #
        #     batch_iterator = train_data_iterator[dataset_name]
        #     for _ in range(int(local_batches)):
        #         try:
        #             data = next(batch_iterator)
        #         except:
        #             # reinitialize dataset
        #             train_dataloaders[dataset_name].dataset.seed += 1
        #             train_dataloaders[dataset_name].dataset._reinitialize(override_process_dataset=True)
        #             # update completed epochs
        #             neox_args.dataset_epochs[dataset_name] = train_dataloaders[dataset_name].dataset._completed_epochs
        #             # create iterator from dataloader
        #             train_data_iterator[dataset_name] = iter(train_dataloaders[dataset_name])
        #             # continue with new batch
        #             batch_iterator = train_data_iterator[dataset_name]
        #             data = next(batch_iterator)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # hds
    ac_agent.warmup_flag = False
    ac_agent.warmup_P_losses = None
    ac_agent.warmup_Q_losses = None
    ac_agent.init_target_network()

    # Main training loop
    while iteration < neox_args.train_iters:
        print(f"******************Iteration: {iteration}************************")
        time1 = time.time()
        cur_weight = ac_agent.get_weights()
        local_batch_names = get_hds_batch_names(neox_args, weights=cur_weight, warmup=False)

        losses = []

        for batch_name in local_batch_names:
            batch_iterator = train_data_iterator[batch_name]

            try:
                loss, tokens = forward_step_only(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model
                )
            except StopIteration:
                train_dataloaders[batch_name].dataset._reinitialize()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                train_data_iterator[batch_name] = iter(train_dataloaders[batch_name])
                batch_iterator = train_data_iterator[batch_name]
                loss, tokens = forward_step_only(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model
                )

            losses.append(loss)
            neox_args.local_samples_seen_per_dataset[batch_name] += 1

            # hds
            backward_step_only(
                neox_args=neox_args,
                timers=timers,
                loss=loss,
                tokens=tokens,
                domain_name=batch_name,
                model=model,
                optimizer=optimizer,
                ac_agent=ac_agent
            )

            neox_args.dataset_iterations[batch_name] += 1 / len(local_batch_names)

        loss_dict = {
            "lm_loss": reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

        if neox_args.precision == "fp16" and model.optimizer.overflow:
            skipped_iter = 1
        else:
            skipped_iter = 0

        # -----------------------------------hds-------------------------------------------------
        if "load_path" not in neox_args.hds.keys():
            timers("hds_update").start()
            ac_agent.update()
            timers("hds_update").stop()
        # -----------------------------------hds-------------------------------------------------

        iteration += 1

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # gather batch losses
        timers("data sampling update").start()
        # print(f"ITERATION: {iteration} -- RANK {torch.distributed.get_rank()} -- PRE BATCH LOSSES {batch_losses}")
        # if torch.distributed.is_initialized():
        #     for b in batch_losses.keys():
        #         torch.distributed.all_reduce(batch_losses[b])
        # scale batch losses
        # batch_losses = {b: batch_losses[b].item()/batch_name_counts[b] for b in batch_losses.keys()}
        # print(f"ITERATION: {iteration} -- RANK {torch.distributed.get_rank()} -- POST BATCH LOSSES {batch_losses}")

        # update data sampling weights
        # reward = loss_dict["lm_loss"].item()
        data_sampling_weights.group_update(iteration, **{"state": None, "agent": ac_agent, "training": True})
        # for batch_name in batch_losses.keys():
        #     data_sampling_weights.update(iteration, **{"dataset_name":batch_name, "reward":batch_losses[batch_name]})
        timers("data sampling update").stop()
        # data_sampling_weights.update(iteration, **{"dataset_name":batch_name, "reward":reward})
        # timers("data sampling update").stop()

        # print("---------------------------------------------DATA WEIGHTS---------------------------------------")
        # print(ac_agent.data_weights)
        # print("---------------------------------------------DATA WEIGHTS---------------------------------------")
        # for _ in ac_agent.pool:
        #     print(_)
        # print("---------------------------------------------DATA POOL---------------------------------------")

        print(f"Total Cost: {time.time() - time1} s")

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
            data_sampling_weights=data_sampling_weights,
            ac_agent=ac_agent
        )

        # Checkpointing
        if neox_args.save and iteration in neox_args.save_iters:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                data_sampling_weights=data_sampling_weights
            )
            ac_agent.save_ckpts(neox_args, iteration)

        # Evaluation
        if (
                neox_args.eval_interval
                and iteration % neox_args.eval_interval == 0
                and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            time2 = time.time()
            evaluate_named_datasets_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterators=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=True,
                timers=timers,
            )
            print(f"---------------EVALUATION COST: {time.time() - time2} s----------------------------")

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

        # sync processes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    return iteration, data_sampling_weights


def train_named_datasets_mixed_batch_validation_reward(
        dataset_names,
        neox_args,
        timers,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    # if using named train datasets, prepare dataloaders, dataiterators and dicts to track iterations and epochs
    neox_args.dataset_iterations = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in dataset_names}
    neox_args.dataset_epochs = {name: 0 for name in dataset_names}
    data_sampler_kwargs = {}
    if neox_args.data_sampling_method in REQUIRES_VALID_DATASET:
        # initialize validation data iterators with slightly different seed
        neox_args.seed += 1
        (_, reward_dataloaders, _, _) = build_train_valid_test_data_iterators(neox_args=neox_args)
        neox_args.seed -= 1
        data_sampler_kwargs['reward_dataloaders'] = reward_dataloaders
        data_sampler_kwargs['neox_args'] = neox_args

    data_sampling_weights = get_data_sampling_weighter(
        dataset_names=dataset_names,
        weights=neox_args.train_data_weights,
        warmup_steps=neox_args.data_sampling_warmup_steps,
        update_frequency=neox_args.data_sampling_update_frequency,
        update_method=neox_args.data_sampling_method,
        **data_sampler_kwargs
    )
    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)
    while iteration < neox_args.train_iters:
        batch_dataset_names, loss_dict, skipped_iter = train_step_named_datasets_mixed_batch(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        iteration += 1

        for name in batch_dataset_names:
            neox_args.dataset_iterations[name.replace("train_", "")] += 1 / neox_args.train_batch_size

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
            data_sampling_weights=data_sampling_weights
        )

        # Checkpointing
        if neox_args.save and iteration in neox_args.save_iters:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluation
        if (
                neox_args.eval_interval
                and iteration % neox_args.eval_interval == 0
                and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            time2 = time.time()
            evaluate_named_datasets_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterators=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )
            print(f"---------------EVALUATION COST: {time.time() - time2} s----------------------------")

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

        timers("data sampling update").start()
        if neox_args.data_sampling_method in REQUIRES_VALID_DATASET:
            data_sampling_weights.update(iteration, **{"model": model})
        train_data_iterator._index_sampler.sampler.weights = data_sampling_weights()
        timers("data sampling update").stop()

    return iteration


def get_batch_names(neox_args, dataset_names, data_sampling_weights, iteration, ac_agent):
    # iterate over full batch size, then split to each device
    batch_names = random.choices(dataset_names, weights=data_sampling_weights(),
                                 k=neox_args.world_size * neox_args.gradient_accumulation_steps)

    # print(f"ITERATION: {iteration} -- RANK: {torch.distributed.get_rank()} -- BATCH NAMES: {batch_names}")
    # only keep indices for this rank
    local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]

    # create a dict counting the number of times each dataset appears in the batch
    batch_name_counts = {name: batch_names.count(name) for name in dataset_names if name in batch_names}
    # print(f"ITERATION: {iteration} -- RANK: {torch.distributed.get_rank()} -- BATCH NAMES: {batch_names} -- BATCH COUNTS: {batch_name_counts} -- LOCAL BATCHES: {local_batches}")
    return local_batches, batch_name_counts


def get_batch_names_hds(neox_args, dataset_names, ac_agent, gi=0):
    itr_i = 0
    batch_names = []
    if neox_args.gradient_accumulation_steps > 1:
        itr_i, remain_n = divmod(neox_args.hds["train_domain_num"], neox_args.hds["domain_num"])
    if gi == 0 or gi < itr_i:
        domian_num = neox_args.hds["domain_num"]
        for dataset_name in dataset_names:
            batch_names.extend([dataset_name] * domian_num)

        remaining_samples = neox_args.hds["batch_size"] - len(batch_names)
        additional_batch_names = random.choices(
            dataset_names, weights=ac_agent.data_weights, k=remaining_samples
        )
        batch_names.extend(additional_batch_names)
        # local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        batch_name_counts = {name: batch_names.count(name) for name in dataset_names if name in batch_names}
    elif gi == itr_i:
        _, domian_num = divmod(neox_args.hds["train_domain_num"], neox_args.hds["domain_num"])
        for dataset_name in dataset_names:
            batch_names.extend([dataset_name] * domian_num)
        remaining_samples = neox_args.hds["batch_size"] - len(batch_names)
        additional_batch_names = random.choices(
            dataset_names, weights=ac_agent.data_weights, k=remaining_samples
        )
        batch_names.extend(additional_batch_names)
        # local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        batch_name_counts = {name: batch_names.count(name) for name in dataset_names if name in batch_names}
    else:
        additional_batch_names = random.choices(
            dataset_names, weights=ac_agent.data_weights, k=neox_args.hds["batch_size"]
        )
        batch_names.extend(additional_batch_names)
        # local_batches = batch_names[torch.distributed.get_rank()::neox_args.world_size]
        batch_name_counts = {name: batch_names.count(name) for name in dataset_names if name in batch_names}

    return batch_names, batch_name_counts


# hds
def forward_step_only(neox_args, timers, data_iterator, model):
    """Forward step."""

    # Forward model for one step.
    timers("forward").start()
    loss, tokens = forward_step(
        neox_args=neox_args,
        timers=timers,
        data_iterator=data_iterator,
        model=model,
    )
    timers("forward").stop()

    return loss, tokens


def backward_step_only(neox_args, timers, loss, tokens, domain_name, model, optimizer, ac_agent):
    # hds
    if neox_args.hds is not None:
        ac_agent.update_pools(neox_args, timers, model, loss, tokens, domain_name)
        # Update parameters
        timers("optimizer").start()
        if neox_args.deepspeed:
            model.step()
        else:
            raise ValueError("Must be using deepspeed to run neox")
        timers("optimizer").stop()
    return


def single_step_validation(neox_args, model, data_iterator):
    model.eval()
    with torch.no_grad():
        try:
            batch = next(data_iterator)
        except StopIteration:
            raise StopIteration
        tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
            neox_args=neox_args,
            tokenizer=neox_args.tokenizer,
            keys=["text"],
            data=batch,
            datatype=torch.int64
        )
        outputs = model((tokens, position_ids, attention_mask))
        loss = cross_entropy(
            outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
        )
        # When contiguous memory optimizations are enabled, the buffers
        # allocated by the optimizations are deallocated during backward pass
        # in the absence of backward pass the buffers should be reset after each
        # forward pass
        if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
            deepspeed.checkpointing.reset()
    model.train()
    return loss


def train_named_datasets_mixed_minibatch_validation_reward(
        dataset_names,
        neox_args,
        timers,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
):
    """Train the model function."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    # if using named train datasets, prepare dataloaders, dataiterators and dicts to track iterations and epochs
    neox_args.dataset_iterations = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in dataset_names}
    train_dataloaders = {name: dataloader for name, dataloader in train_data_iterator.items()}
    train_data_iterator = {name: iter(dataloader) for name, dataloader in train_dataloaders.items()}
    neox_args.dataset_epochs = {name: dataloader.dataset._completed_epochs for name, dataloader in
                                train_dataloaders.items()}
    data_sampler_kwargs = {}
    # initialize validation data iterators with slightly different seed, and only allow initialization on main process
    neox_args.seed += 1
    neox_args.num_workers = 0
    (_, reward_dataloaders, _, _) = build_train_valid_test_data_iterators(neox_args=neox_args)
    neox_args.seed -= 1
    reward_data_iterators = {name: iter(dataloader) for name, dataloader in reward_dataloaders.items()}

    # track running average of batch loss, reward loss, and generalization reward
    # running_batch_loss = {name: 0.0 for name in dataset_names}
    # running_reward_loss = {name: 0.0 for name in dataset_names}
    # running_generalization_reward = {name: 0.0 for name in dataset_names}
    if neox_args.scale_by_reward_loss_diff:
        last_reward = {name: None for name in dataset_names}

    data_sampling_weights = get_data_sampling_weighter(
        dataset_names=dataset_names,
        weights=neox_args.train_data_weights,
        warmup_steps=neox_args.data_sampling_warmup_steps,
        update_frequency=neox_args.data_sampling_update_frequency,
        update_method=neox_args.data_sampling_method,
        **data_sampler_kwargs
    )
    first_reward_step = True

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)

    while iteration < neox_args.train_iters:
        iteration += 1

        local_batch_names, batch_name_counts = get_batch_names(neox_args, dataset_names, data_sampling_weights,
                                                               iteration)
        batch_losses = {b: torch.tensor(0.0, device=torch.cuda.current_device()) for b in batch_name_counts.keys()}
        reward_losses = {b: torch.tensor(0.0, device=torch.cuda.current_device()) for b in batch_name_counts.keys()}
        losses = []

        for batch_name in local_batch_names:
            batch_iterator = train_data_iterator[batch_name]

            # use try/except to allow for restarting a dataset
            try:
                loss, _ = forward_step_only(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model,
                )

            except StopIteration:
                # stop timers from attempted iteration
                timers("forward").stop()
                timers("batch generator").stop()
                # reinitialize dataset
                train_dataloaders[batch_name].dataset.seed += 1
                train_dataloaders[batch_name].dataset._reinitialize(override_process_dataset=True)
                # update completed epochs
                neox_args.dataset_epochs[batch_name] = train_dataloaders[batch_name].dataset._completed_epochs
                # create iterator from dataloader
                train_data_iterator[batch_name] = iter(train_dataloaders[batch_name])
                # continue with new batch
                batch_iterator = train_data_iterator[batch_name]
                loss, _ = forward_step_only(
                    neox_args=neox_args,
                    timers=timers,
                    data_iterator=batch_iterator,
                    model=model,
                )

            batch_losses[batch_name] += loss.item()
            losses.append(loss)

            timers("data sampling update").start()
            if data_sampling_weights.update_scheduler.requires_update(iteration):
                if first_reward_step:
                    # preserve some GPU space until we're done with the warmup
                    torch.cuda.empty_cache()
                    first_reward_step = False
                # run validation on a single batch to get reward
                try:
                    reward_loss = single_step_validation(neox_args, model, reward_data_iterators[batch_name])
                except StopIteration:
                    reward_data_iterators[batch_name] = iter(reward_dataloaders[batch_name])
                    # continue with new batch
                    reward_loss = single_step_validation(neox_args, model, reward_data_iterators[batch_name])

                # print(f"ITERATION: {iteration} -- RANK: {torch.distributed.get_rank()} -- BATCH NAME: {batch_name} -- REWARD LOSS: {reward_loss.item()}")
                reward_losses[batch_name] += reward_loss.item()

            timers("data sampling update").stop()

            neox_args.dataset_iterations[batch_name] += 1

        loss_dict, skipped_iter = backward_step_only(
            neox_args=neox_args,
            timers=timers,
            losses=losses,
            model=model,
            optimizer=optimizer,
        )

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
            data_sampling_weights=data_sampling_weights
        )

        # Checkpointing
        if neox_args.save and iteration in neox_args.save_iters:
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluation
        if (
                neox_args.eval_interval
                and iteration % neox_args.eval_interval == 0
                and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            evaluate_named_datasets_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=forward_step,
                data_iterators=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

        timers("data sampling update").start()
        if data_sampling_weights.update_scheduler.requires_update(iteration):
            if neox_args.batch_normalized_reward:
                generalization_reward = {}
                for batch_name in batch_name_counts.keys():
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(reward_losses[batch_name])
                        torch.distributed.all_reduce(batch_losses[batch_name])
                    # rescale losses by number of batches
                    reward_losses[batch_name] /= batch_name_counts[batch_name]
                    batch_losses[batch_name] /= batch_name_counts[batch_name]
                    generalization_reward[batch_name] = reward_losses[batch_name] - batch_losses[batch_name]
                    if neox_args.scale_by_reward_loss_diff:
                        if last_reward[batch_name] is not None:
                            reward_loss_diff = last_reward[batch_name] - reward_losses[batch_name]
                            generalization_reward[batch_name] *= reward_loss_diff
                        last_reward[batch_name] = reward_losses[batch_name]
                min_reward = min(generalization_reward.values())
                max_reward = max(generalization_reward.values())
                for batch_name in batch_name_counts.keys():
                    normalized_reward = (generalization_reward[batch_name] - min_reward) / (max_reward - min_reward)
                    data_sampling_weights.update(iteration, **{"dataset_name": batch_name, "reward": normalized_reward})
            else:
                for batch_name in batch_name_counts.keys():
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(reward_losses[batch_name])
                        torch.distributed.all_reduce(batch_losses[batch_name])
                    # rescale losses by number of batches
                    reward_losses[batch_name] /= batch_name_counts[batch_name]
                    batch_losses[batch_name] /= batch_name_counts[batch_name]
                    generalization_reward = reward_losses[batch_name] - batch_losses[batch_name]
                    if neox_args.scale_by_reward_loss_diff:
                        if last_reward[batch_name] is not None:
                            reward_loss_diff = last_reward[batch_name] - reward_losses[batch_name]
                            generalization_reward *= reward_loss_diff
                        last_reward[batch_name] = reward_losses[batch_name]
                    data_sampling_weights.update(iteration,
                                                 **{"dataset_name": batch_name, "reward": generalization_reward})
        timers("data sampling update").stop()

        # update running means
        # for batch_name in batch_name_counts.keys():
        #     running_reward_loss[batch_name] = 0.95 * running_reward_loss[batch_name] + 0.05 * reward_losses[batch_name]
        #     running_batch_loss[batch_name] = 0.95 * running_batch_loss[batch_name] + 0.05 * batch_losses[batch_name]
        #     running_generalization_reward[batch_name] = running_reward_loss[batch_name] - running_batch_loss[batch_name]
        # log batch and reward losses

        # if iteration % 10 == 0:
        #     for dataset_name in dataset_names:
        #         tb_wandb_log(
        #             f"train/batch_loss/{dataset_name}",
        #             running_batch_loss[dataset_name],
        #             iteration,
        #             use_wandb=neox_args.use_wandb,
        #             tensorboard_writer=neox_args.tensorboard_writer,
        #         )
        #         tb_wandb_log(
        #             f"train/reward_loss/{dataset_name}",
        #             running_reward_loss[dataset_name],
        #             iteration,
        #             use_wandb=neox_args.use_wandb,
        #             tensorboard_writer=neox_args.tensorboard_writer,
        #         )
        #         # log generalization reward
        #         tb_wandb_log(
        #             f"train/generalization_reward/{dataset_name}",
        #             running_generalization_reward[dataset_name],
        #             iteration,
        #             use_wandb=neox_args.use_wandb,
        #             tensorboard_writer=neox_args.tensorboard_writer,
        #         )
    return iteration


def evaluate(
        neox_args, forward_step_fn, data_iterator, model, verbose=False, timers=None
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    if neox_args.char_level_ppl:
        data_iterator = CharCounter(data_iterator, neox_args.tokenizer)

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gas into account - default to 1 here if pipe parallel is true
            for _ in range(
                    1
                    if neox_args.is_pipe_parallel
                    else neox_args.gradient_accumulation_steps
            ):
                # Forward evaluation
                output = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                    timers=timers,
                )
                if isinstance(output, tuple):
                    loss, _ = output
                else:
                    loss = output
                losses.append(loss)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])

    if neox_args.char_level_ppl:
        # calculate character level perplexity, if specified
        # if neox_args.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(
            eval_results["lm_loss"] * tokens_per_char
        )

    if neox_args.eval_tasks:
        eval_results.update(
            run_eval_harness(
                model, forward_step_fn, neox_args, eval_tasks=neox_args.eval_tasks
            ).get("results")
        )
    # Move model back to the train mode.
    model.train()
    return eval_results


def evaluate_named_dataset(
        neox_args, forward_step_fn, name, data_iterator, model, verbose=False, timers=None, debug=False
):
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    # print num samples in data_iterator
    if debug:
        print_rank_0(f"num samples in {name} data_iterator: {len(data_iterator)}")
        num_samples = []
        token_list = []
    losses = []

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    if neox_args.char_level_ppl:
        data_iterator = CharCounter(data_iterator, neox_args.tokenizer)

    with torch.no_grad():
        for iteration, batch in enumerate(data_iterator):
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating {} - iter {}".format(name, iteration)
                )

            tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(
                neox_args=neox_args,
                tokenizer=neox_args.tokenizer,
                keys=keys,
                data=batch,
                datatype=datatype,
            )

            outputs = model((tokens, position_ids, attention_mask))
            loss = cross_entropy(
                outputs, (labels, loss_mask), _fp16=neox_args.fp16_lm_cross_entropy
            )
            losses.append(loss)

            if debug:
                num_samples.append(tokens.shape[0])
                token_list.append(tokens)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    if debug:
        # collect all tokens
        all_tokens = torch.cat(token_list, dim=0)
        print(f"rank {torch.distributed.get_rank()} all_tokens: {all_tokens.shape}")
        torch.distributed.all_reduce(all_tokens)
        if torch.distributed.get_rank() == 0:
            count = 0
            # count times first item in token_list is equal to anything in all_tokens
            first_example = token_list[0]
            for example in all_tokens:
                if torch.equal(first_example, example):
                    count += 1
            print(f"rank {torch.distributed.get_rank()} count: {count}")
        # print num samples
        print(f"rank {torch.distributed.get_rank()} num samples: {num_samples}")
        reduced_samples = torch.tensor(sum(num_samples), dtype=torch.long, device=torch.cuda.current_device())
        torch.distributed.all_reduce(reduced_samples, torch.distributed.ReduceOp.SUM)
        print_rank_0(f"eval {name} total samples: {reduced_samples.item()}")

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {f"lm_loss": reduce_losses(losses).mean().item()}
    eval_results[f"lm_loss_ppl"] = math.exp(eval_results[f"lm_loss"])

    if neox_args.char_level_ppl:
        # calculate character level perplexity, if specified
        # if neox_args.char_level_ppl:
        # unwrap the data_iterator
        tokens_per_char = data_iterator.tokens_per_char()
        print_rank_0(f"Counting chars took {data_iterator.total_time} seconds")

        data_iterator = data_iterator.data_iterator
        eval_results["lm_loss_char_lvl_ppl"] = math.exp(
            eval_results["lm_loss"] * tokens_per_char
        )

    # Move model back to the train mode.
    model.train()
    return eval_results


def evaluate_and_print_results(
        neox_args,
        prefix,
        forward_step_func,
        data_iterator,
        model,
        iteration,
        verbose=False,
        timers=None,
        chart_name="validation",
):
    """Helper function to evaluate and dump results on screen."""
    total_loss_dict = evaluate(
        neox_args=neox_args,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
    )
    string = f" {chart_name} results at {prefix} | "
    for k, v in total_loss_dict.items():
        if isinstance(v, dict):
            if neox_args.eval_tasks and "results" in v:
                v = v["results"]
                print(v)
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                tb_wandb_log(
                    f"{chart_name}/{k3}",
                    v2,
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                )
        else:
            string += f"{k} value: {v:.6E} | "
            tb_wandb_log(
                f"{chart_name}/{k}",
                v,
                iteration,
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)


def evaluate_named_datasets_and_print_results(
        neox_args,
        prefix,
        forward_step_func,
        data_iterators,
        model,
        iteration,
        verbose=False,
        timers=None,
        chart_name="validation",
):
    """Helper function to evaluate named datasets and dump results on screen."""
    string = f" {chart_name} results at {prefix} | "

    overall_results = {}

    for name, iterator in data_iterators.items():
        total_loss_dict = evaluate_named_dataset(
            neox_args=neox_args,
            forward_step_fn=forward_step_func,
            name=name,
            data_iterator=iterator,
            model=model,
            verbose=verbose,
            timers=timers,
        )
        overall_results[name] = total_loss_dict
        string += f"\n{name} | "
        for k, v in total_loss_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    k3 = "_".join([k, k2])
                    string += f"{k3} value: {v2:.6E} | "
                    tb_wandb_log(
                        f"{chart_name}/{k3}/{name}",
                        v2,
                        iteration,
                        use_wandb=neox_args.use_wandb,
                        tensorboard_writer=neox_args.tensorboard_writer,
                    )
            else:
                string += f"{k} value: {v:.6E} | "
                tb_wandb_log(
                    f"{chart_name}/{k}/{name}",
                    v,
                    iteration,
                    use_wandb=neox_args.use_wandb,
                    tensorboard_writer=neox_args.tensorboard_writer,
                )

    # calculate average loss and perplexity across all datasets
    avg_lm_loss, weighted_avg_lm_loss = 0, 0
    avg_ppl, weighted_avg_ppl = 0, 0
    record_weighted_dataset = False
    if "validation" in chart_name and hasattr(neox_args, "validation_dataset_weights"):
        dataset_weights = neox_args.validation_dataset_weights
        record_weighted_dataset = True
    elif "test" in chart_name and hasattr(neox_args, "test_dataset_weights"):
        dataset_weights = neox_args.test_dataset_weights
        record_weighted_dataset = True

    for name, results in overall_results.items():
        avg_lm_loss += results["lm_loss"]
        avg_ppl += results["lm_loss_ppl"]
        if record_weighted_dataset:
            weighted_avg_lm_loss += results["lm_loss"] * dataset_weights[name]
            weighted_avg_ppl += results["lm_loss_ppl"] * dataset_weights[name]

    avg_lm_loss /= len(overall_results)
    avg_ppl /= len(overall_results)
    string += f"\navg_loss value: {avg_lm_loss:.6E} | "
    string += f"avg_ppl value: {avg_ppl:.6E} | "
    tb_wandb_log(
        f"{chart_name}/avg_lm_loss",
        avg_lm_loss,
        iteration,
        use_wandb=neox_args.use_wandb,
        tensorboard_writer=neox_args.tensorboard_writer,
    )
    tb_wandb_log(
        f"{chart_name}/avg_ppl",
        avg_ppl,
        iteration,
        use_wandb=neox_args.use_wandb,
        tensorboard_writer=neox_args.tensorboard_writer,
    )

    if record_weighted_dataset:
        string += f"\nweighted_avg_lm_loss value: {weighted_avg_lm_loss:.6E} | "
        string += f"weighted_avg_ppl value: {weighted_avg_ppl:.6E} | "
        tb_wandb_log(
            f"{chart_name}/weighted_avg_lm_loss",
            weighted_avg_lm_loss,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )
        tb_wandb_log(
            f"{chart_name}/weighted_avg_ppl",
            weighted_avg_ppl,
            iteration,
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )

    length = len(string) + 1
    print_rank_0("-" * min(80, length))
    print_rank_0(string)
    print_rank_0("-" * min(80, length))