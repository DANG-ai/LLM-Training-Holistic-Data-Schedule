import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import deepspeed
import random
from typing import List, Optional, Dict, Iterator
from collections import OrderedDict
import os
from megatron.utils import reduce_losses
import copy
from megatron.learning_rates import AnnealingLR
from torch.distributions import Normal


def mean_std_norm(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor = (tensor - mean) / (std + 1e-8)
    return torch.clamp(tensor, -3, 3)


def get_model_weights(model, module_list):
    weights_list = []
    for name, param in model.named_parameters():
        if any(module == name for module in module_list):
            weights_list.append(param.data.clone().detach().cpu())
    return weights_list


def get_model_grad_flat(model, module_list):
    """Retrieve and flatten gradients of specified modules in the model"""
    grad_list = []
    for name, param in model.named_parameters():
        if any(module == name for module in module_list) and param.grad is not None:
            grad_list.append(param.grad.clone().detach().flatten().cpu())
    return torch.cat(grad_list) if grad_list else torch.tensor([])


def compute_domain_gradients(neox_args, timers, model, loss, domain_idx, dnum_dict=None, grad_matrix=None,
                             prev_grad=None, return_loss=False):
    if neox_args.deepspeed:
        timers("backward-backward").start()
        # domain_loss.backward(retain_graph=True)
        model.backward(loss)
        timers("backward-backward").stop()
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")

    # Retrieve current gradients
    current_grad = get_model_grad_flat(model, neox_args.hds["selected_weights_name"])

    # Compute the gradient difference if the previous gradient exists
    if prev_grad is not None:
        domain_grad = current_grad - prev_grad
    else:
        domain_grad = current_grad.clone()

    # Update the previous gradient
    prev_grad = current_grad.clone()

    # Store the gradient in the matrix
    if "load_path" not in neox_args.hds.keys():
        grad_matrix[domain_idx, :] += domain_grad

        if dnum_dict is not None:
            for domain_idx, domain in enumerate(neox_args.hds["datasets_names"]):
                if dnum_dict[domain].item() != 0:
                    grad_matrix[domain_idx, :] /= dnum_dict[domain].to("cpu")

    # return grad_matrix, prev_grad
    return grad_matrix, prev_grad


def calculate_mtld(batch_tokens):
    """
    A simplified proxy for MTLD. Calculates Type-Token Ratio (TTR).
    A proper implementation of MTLD is more complex.
    """
    if batch_tokens.nelement() == 0:
        return 0.0

    # Assuming batch_tokens is a 2D tensor [batch_size, seq_len]
    ttrs = []
    for i in range(batch_tokens.size(0)):
        tokens = [tok for tok in batch_tokens[i].tolist() if tok != -1]  # Filter out padding
        if not tokens:
            continue

        num_types = len(set(tokens))
        num_tokens = len(tokens)

        ttr = num_types / num_tokens if num_tokens > 0 else 0
        ttrs.append(ttr)

    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def compute_rewards(neox_args, timers, model, loss, tokens, domain_idx, prev_grad=None, dnum_dict=None, old_mat=None):
    grad_matrix, prev_grad = compute_domain_gradients(neox_args, timers, model, loss, domain_idx, prev_grad=prev_grad,
                                                      dnum_dict=dnum_dict, grad_matrix=old_mat)

    # r_align calculation
    if dnum_dict is None:
        # This case happens during micro-batches accumulation, only grad is accumulated.
        return grad_matrix, prev_grad

    if "load_path" not in neox_args.hds.keys():
        grad_matrix = grad_matrix.to(torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.all_reduce(grad_matrix)
        grad_matrix = grad_matrix.to("cpu")
        scores_mat_all = grad_matrix @ grad_matrix.T

        diag_elements = torch.diag(scores_mat_all)
        # r_align is the sum of off-diagonal elements
        r_align = scores_mat_all.sum(dim=-1) - diag_elements

        avg_norm = grad_matrix.norm(dim=-1).mean()
        r_align = r_align / (avg_norm + 1e-6)
        r_align = torch.clip(r_align, min=neox_args.hds["dw_min"], max=neox_args.hds["dw_max"])
        r_align = mean_std_norm(r_align)
    else:
        r_align = torch.zeros(len(neox_args.hds["datasets_names"]))

    # r_diversity calculation
    mtld_score = calculate_mtld(tokens)
    # Normalize step
    current_step = neox_args.iteration
    total_steps = neox_args.train_iters
    t_prime = current_step / total_steps if total_steps > 0 else 0

    # As per paper, MTLD_max is seq_len and MTLD_min is 2. Here we use TTR which is [0,1].
    # We will skip normalization for this proxy.
    mtld_norm = mtld_score

    r_diversity_scalar = t_prime / (mtld_norm + 1e-8) if mtld_norm > 0 else 0
    r_diversity = torch.full((len(neox_args.hds["datasets_names"]),), r_diversity_scalar)

    # r_stability is calculated in update_pools based on weight changes
    # Here we return the components that can be calculated now.

    # Combine rewards in update_pools
    return r_align, r_diversity, None


class FullyConnectedWithSkip(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FullyConnectedWithSkip, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        if input_dim != hidden_dim:
            self.skip_fc = nn.Linear(input_dim, hidden_dim)
        else:
            self.skip_fc = None

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.layer_norm(out)
        out = self.activation(out)

        if self.skip_fc is not None:
            identity = self.skip_fc(identity)
        out = out + identity
        return out



class PolicyNet(nn.Module):
    def __init__(self, embed_dim, ff_dim, model_dim, fusion_dim, num_heads=8, num_layers=8, dropout=0.1):
        super(PolicyNet, self).__init__()

        # Domain feature extraction
        self.domain_proj = nn.Linear(3, embed_dim)
        self.domain_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,
            ),
            num_layers=num_layers
        )

        # Weights feature extraction
        self.weights_fc1 = nn.Linear(18, 240)
        self.weights_norm1 = nn.LayerNorm(240)

        # Iteration feature extraction
        self.iter_fc1 = nn.Linear(1, 16)
        self.iter_norm1 = nn.LayerNorm(16)

        # Model feature extraction
        self.model_fc1 = nn.Linear(256, model_dim)
        self.model_norm1 = nn.LayerNorm(model_dim)
        self.model_fc2 = nn.Linear(model_dim, model_dim)
        self.model_norm2 = nn.LayerNorm(model_dim)

        # Final feature fusion
        self.final_fc1 = nn.Linear(int(model_dim+22*embed_dim), fusion_dim)
        self.final_norm1 = nn.LayerNorm(fusion_dim)
        self.final_fc2 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc3 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc4 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc_final = nn.Linear(fusion_dim, 22)
        self.final_act = nn.Softmax(dim=-1)

    def forward(self, x):
        domain_loss_norm, sub_domain_loss_l2_norm, domain_nums, model_weights_l2_norm, sub_weights_l2_norm, iter_num = x
        bs = domain_loss_norm.shape[0]

        # Domain feature extraction

        domain_input = self.domain_proj(torch.cat([
            domain_loss_norm.view(bs, -1, 1),
            sub_domain_loss_l2_norm.view(bs, -1, 1),
            domain_nums.view(bs, -1, 1)
        ], dim=2))  # bs x 22 x embed_dim
        domain_feature = self.domain_encoder(domain_input)  # bs x 22 x embed_dim
        domain_feature = domain_feature.view(bs, -1)  # bs x 22*embed_dim

        # Weights feature extraction
        weights_input = torch.cat([model_weights_l2_norm, sub_weights_l2_norm], dim=1)  # bs x 18
        weights_f1 = F.relu(self.weights_norm1(self.weights_fc1(weights_input)))

        # Iteration feature extraction
        iter_f1 = F.relu(self.iter_norm1(self.iter_fc1(iter_num)))

        # Model feature extraction
        model_feature = torch.cat([weights_f1, iter_f1], dim=1)  # bs x 146
        model_feature = F.relu(self.model_norm1(self.model_fc1(model_feature)))
        model_feature = model_feature + F.relu(self.model_norm2(self.model_fc2(model_feature)))

        # Final feature fusion
        final_feature = torch.cat([domain_feature, model_feature], dim=1)  # bs x (model_dim+22*embed_dim)
        final_feature = F.relu(self.final_norm1(self.final_fc1(final_feature)))
        final_feature = self.final_fc2(final_feature)
        final_feature = self.final_fc3(final_feature)
        final_feature = self.final_fc4(final_feature)
        output = self.final_act(self.final_fc_final(final_feature))
        return output


class QValueNet(nn.Module):
    def __init__(self, embed_dim, ff_dim, model_dim, fusion_dim, num_heads=4, num_layers=8, dropout=0.1):
        super(QValueNet, self).__init__()

        # Domain feature extraction
        self.domain_proj = nn.Linear(4, embed_dim)
        self.domain_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,
            ),
            num_layers=num_layers
        )

        # Weights feature extraction
        self.weights_fc1 = nn.Linear(18, 240)
        self.weights_norm1 = nn.LayerNorm(240)

        # Iteration feature extraction
        self.iter_fc1 = nn.Linear(1, 16)
        self.iter_norm1 = nn.LayerNorm(16)

        # Model feature extraction
        self.model_fc1 = nn.Linear(256, model_dim)
        self.model_norm1 = nn.LayerNorm(model_dim)
        self.model_fc2 = nn.Linear(model_dim, model_dim)
        self.model_norm2 = nn.LayerNorm(model_dim)

        # Final feature fusion
        self.final_fc1 = nn.Linear(int(model_dim+22*embed_dim), fusion_dim)
        self.final_norm1 = nn.LayerNorm(fusion_dim)
        self.final_fc2 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc3 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc4 = FullyConnectedWithSkip(fusion_dim, fusion_dim)
        self.final_fc_final = nn.Linear(fusion_dim, 22)
        self.final_act = nn.Softmax(dim=-1)

    def forward(self, x, action):
        domain_loss_norm, sub_domain_loss_l2_norm, domain_nums, model_weights_l2_norm, sub_weights_l2_norm, iter_num = x
        bs = domain_loss_norm.shape[0]

        # Domain feature extraction

        domain_input = self.domain_proj(torch.cat([
            domain_loss_norm.view(bs, -1, 1),
            sub_domain_loss_l2_norm.view(bs, -1, 1),
            domain_nums.view(bs, -1, 1),
            action.view(bs, -1, 1)
        ], dim=2))  # bs x 22 x embed_dim
        domain_feature = self.domain_encoder(domain_input)  # bs x 22 x embed_dim
        domain_feature = domain_feature.view(bs, -1)  # bs x 22*embed_dim

        # Weights feature extraction
        weights_input = torch.cat([model_weights_l2_norm, sub_weights_l2_norm], dim=1)  # bs x 18
        weights_f1 = F.relu(self.weights_norm1(self.weights_fc1(weights_input)))

        # Iteration feature extraction
        iter_f1 = F.relu(self.iter_norm1(self.iter_fc1(iter_num)))

        # Model feature extraction
        model_feature = torch.cat([weights_f1, iter_f1], dim=1)  # bs x 146
        model_feature = F.relu(self.model_norm1(self.model_fc1(model_feature)))
        model_feature = model_feature + F.relu(self.model_norm2(self.model_fc2(model_feature)))

        # Final feature fusion
        final_feature = torch.cat([domain_feature, model_feature], dim=1)  # bs x (model_dim+22*embed_dim)
        final_feature = F.relu(self.final_norm1(self.final_fc1(final_feature)))
        final_feature = self.final_fc2(final_feature)
        final_feature = self.final_fc3(final_feature)
        final_feature = self.final_fc4(final_feature)
        output = self.final_act(self.final_fc_final(final_feature))
        return output

class hds:
    def __init__(
            self,
            neox_args
    ):
        dataset_names = neox_args.hds["datasets_names"]
        self.update_n = neox_args.hds["update_n"]
        self.gamma = neox_args.hds["gamma"]
        self.tau = neox_args.hds["tau"]
        self.alpha = neox_args.hds.get("alpha", 0.2)
        self.target_entropy = -torch.prod(torch.Tensor((len(dataset_names),)).to('cuda')).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')

        self.selected_params_num = neox_args.hds["selected_params_num"]
        self.dataset_names = dataset_names
        self.domain_2_idx = {s: i for i, s in enumerate(self.dataset_names)}
        self.idx_2_domain = {i: s for i, s in enumerate(self.dataset_names)}
        self.state_dim = len(dataset_names) * 3 + len(neox_args.hds["weights_layers"]) * 2 + 1
        self.action_dim = len(dataset_names)
        self.rewards_dim = len(dataset_names)

        self.online_Q_model1 = QValueNet(self.state_dim, neox_args.hds["Q_hidden_dim"], self.action_dim)
        self.online_Q_model2 = QValueNet(self.state_dim, neox_args.hds["Q_hidden_dim"], self.action_dim)
        self.online_P_model = PolicyNet(self.state_dim, neox_args.hds["P_hidden_dim"], self.action_dim)

        self.target_Q_model1 = QValueNet(self.state_dim, neox_args.hds["Q_hidden_dim"], self.action_dim)
        self.target_Q_model2 = QValueNet(self.state_dim, neox_args.hds["Q_hidden_dim"], self.action_dim)

        self.online_Q_model = QValueNet(embed_dim=256, ff_dim=512, model_dim=256, fusion_dim=1024)
        self.online_P_model = PolicyNet(embed_dim=256, ff_dim=512, model_dim=256, fusion_dim=1024)
        self.target_Q_model = QValueNet(embed_dim=256, ff_dim=512, model_dim=256, fusion_dim=1024)
        self.target_P_model = PolicyNet(embed_dim=256, ff_dim=512, model_dim=256, fusion_dim=1024)

        self.target_Q_model1.load_state_dict(self.online_Q_model1.state_dict())
        self.target_Q_model2.load_state_dict(self.online_Q_model2.state_dict())

        ds_config = {
            "train_batch_size": neox_args.hds["update_n"],
            "train_micro_batch_size_per_gpu": neox_args.hds["train_micro_batch_size_per_gpu"],
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": False},
            "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
        }

        try:
            from apex.optimizers import FusedAdam as Adam
        except ImportError:
            print("WARNING: APEX not installed - defaulting to deepspeed's fused adam")
            from deepspeed.ops.adam import FusedAdam as Adam

        self.optimizer_Q1 = Adam(self.online_Q_model1.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.optimizer_Q2 = Adam(self.online_Q_model2.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.optimizer_P = Adam(self.online_P_model.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.alpha_optim = Adam([self.log_alpha], lr=1e-4)

        self.model_engine_online_Q1, _, _, _ = deepspeed.initialize(model=self.online_Q_model1,
                                                                    model_parameters=self.online_Q_model1.parameters(),
                                                                    config_params=ds_config,
                                                                    optimizer=self.optimizer_Q1)
        self.model_engine_online_Q2, _, _, _ = deepspeed.initialize(model=self.online_Q_model2,
                                                                    model_parameters=self.online_Q_model2.parameters(),
                                                                    config_params=ds_config,
                                                                    optimizer=self.optimizer_Q2)
        self.model_engine_online_P, self.optimizer_online_P, _, _ = deepspeed.initialize(model=self.online_P_model,
                                                                                         model_parameters=self.online_P_model.parameters(),
                                                                                         config_params=ds_config,
                                                                                         optimizer=self.optimizer_P)
        self.model_engine_target_Q1, _, _, _ = deepspeed.initialize(model=self.target_Q_model1,
                                                                    model_parameters=self.target_Q_model1.parameters(),
                                                                    config_params=ds_config)
        self.model_engine_target_Q2, _, _, _ = deepspeed.initialize(model=self.target_Q_model2,
                                                                    model_parameters=self.target_Q_model2.parameters(),
                                                                    config_params=ds_config)

        self.cur_state = None
        self.pool = []
        self.data_weights = neox_args.train_data_weights
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        total_weights = np.sum(neox_args.train_data_weights)
        self._probabilities = {name: weight / total_weights for name, weight in
                               zip(dataset_names, neox_args.train_data_weights)}
        self.warmup_flag = True
        self.critic_loss = None
        self.actor_loss = None
        self.micro_steps = 0
        self.micro_num_dict = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in dataset_names}
        self.old_mat = torch.zeros((len(neox_args.hds["datasets_names"]), neox_args.hds["selected_params_num"]),
                                   dtype=torch.float32)
        self.domain_loss_dict = {name: torch.tensor([0.0, 0.0]).to(torch.cuda.current_device()) for name in
                                 dataset_names}
        self.prev_grad = None
        self.score_w = [0.0] * len(dataset_names)
        self.warmup_weighter = None
        self.warmup_P_losses = None
        self.warmup_Q_losses = None
        self.last_domain_loss = None
        self.last_weights_list = None
        self.last_weights_norm = None
        self.batchsize = neox_args.hds["train_micro_batch_size_per_gpu"]

    def update_domain_loss(self, domain_loss, dname):
        self.domain_loss_dict[dname][0] += domain_loss
        self.domain_loss_dict[dname][1] += 1
        return

    def update_pools(self, neox_args, timers, model, loss, tokens, domain_name, weights=None):
        self.micro_num_dict[domain_name] += 1
        self.update_domain_loss(loss, domain_name)
        if (self.micro_steps + 1) % neox_args.gradient_accumulation_steps == 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                for name in self.micro_num_dict.keys():
                    torch.distributed.all_reduce(self.micro_num_dict[name])
                    torch.distributed.all_reduce(self.domain_loss_dict[name])

            dnum_dict = copy.deepcopy(self.micro_num_dict)

            r_align, r_diversity, self.prev_grad = compute_rewards(neox_args, timers, model, loss, tokens,
                                                                   self.domain_2_idx[domain_name],
                                                                   prev_grad=self.prev_grad, dnum_dict=dnum_dict,
                                                                   old_mat=self.old_mat)

            model_weights = get_model_weights(model, neox_args.hds["weights_layers"])
            model_weights_l2_norm_tensor = torch.tensor([torch.norm(mv, p=2).item() for mv in model_weights])
            model_weights_l2_norm = mean_std_norm(model_weights_l2_norm_tensor)

            if self.last_weights_norm is not None:
                r_stability_scalar = 1.0 / (
                            torch.abs(model_weights_l2_norm_tensor - self.last_weights_norm).sum() + 1e-8)
                r_stability_scalar = min(r_stability_scalar, neox_args.hds.get("r_stability_cap", 5.0))
            else:
                r_stability_scalar = 0.0

            self.last_weights_norm = model_weights_l2_norm_tensor.clone()
            r_stability = torch.full((len(self.dataset_names),), r_stability_scalar)

            w_align = neox_args.hds.get("w_align", 1.0)
            w_diversity = neox_args.hds.get("w_diversity", 10.0)
            w_stability = neox_args.hds.get("w_stability", 10.0)

            final_reward = w_align * r_align + w_diversity * r_diversity + w_stability * r_stability

            domain_loss = torch.tensor([self.domain_loss_dict[dictname][0] / self.domain_loss_dict[dictname][1] if
                                        self.domain_loss_dict[dictname][1] > 0 else 0.0 for dictname in
                                        self.dataset_names]).to("cpu")
            domain_loss_norm = mean_std_norm(domain_loss)

            domain_nums = mean_std_norm(torch.tensor(
                [self.domain_loss_dict[dictname][1] * neox_args.train_micro_batch_size_per_gpu for dictname in
                 self.dataset_names]).to("cpu"))
            if self.last_domain_loss is not None:
                sub_domain_loss_l2_norm = mean_std_norm(domain_loss - self.last_domain_loss)
                sub_weights_l2_norm = mean_std_norm(torch.tensor(
                    [torch.norm(mw - self.last_weights_list[mi], p=2).item() for mi, mw in enumerate(model_weights)]))
            else:
                sub_domain_loss_l2_norm = torch.zeros_like(domain_loss_norm)
                sub_weights_l2_norm = torch.zeros_like(model_weights_l2_norm)

            self.last_domain_loss = domain_loss.clone()
            self.last_weights_list = [mv.clone() for mv in model_weights]

            state = [
                domain_loss_norm,
                sub_domain_loss_l2_norm,
                domain_nums,
                model_weights_l2_norm,
                sub_weights_l2_norm,
                torch.tensor([int((
                                              self.micro_steps + 1) // neox_args.gradient_accumulation_steps) / neox_args.train_iters *
                              neox_args.hds["iter_coe"]])
            ]

            if self.cur_state is not None:
                action_to_store = self.data_weights if not self.warmup_flag else weights
                pool_item = (self.cur_state, action_to_store, final_reward.tolist(), state)
                self.pool.append(pool_item)
                if len(self.pool) > neox_args.hds["pool_size"]:
                    self.pool.pop(0)

            self.cur_state = [si.clone() for si in state]

            self.old_mat.zero_()
            self.domain_loss_dict = {name: torch.tensor([0.0, 0.0]).to(torch.cuda.current_device()) for name in
                                     self.dataset_names}
            self.micro_num_dict = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in
                                   self.dataset_names}

        else:
            dnum_dict = None
            self.old_mat, self.prev_grad = compute_domain_gradients(neox_args, timers, model, loss,
                                                                    self.domain_2_idx[domain_name],
                                                                    prev_grad=self.prev_grad, dnum_dict=dnum_dict,
                                                                    grad_matrix=self.old_mat)
        self.micro_steps += 1
        return

    def init_target_network(self):
        self.target_Q_model1.load_state_dict(self.online_Q_model1.state_dict())
        self.target_Q_model2.load_state_dict(self.online_Q_model2.state_dict())
        return

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if len(self.pool) < self.batchsize:
            return

        self.model_engine_online_Q1.train()
        self.model_engine_online_Q2.train()
        self.model_engine_online_P.train()

        selected_items = random.sample(self.pool, self.batchsize)

        states = [
            torch.cat([torch.as_tensor(item[0][i], dtype=torch.float).view(1, -1) for item in selected_items], dim=0)
            for i in range(len(selected_items[0][0]))]
        actions = torch.cat([torch.as_tensor(item[1], dtype=torch.float).view(1, -1) for item in selected_items], dim=0)
        rewards = torch.cat([torch.as_tensor(item[2], dtype=torch.float).view(1, -1) for item in selected_items], dim=0)
        next_states = [
            torch.cat([torch.as_tensor(item[3][i], dtype=torch.float).view(1, -1) for item in selected_items], dim=0)
            for i in range(len(selected_items[0][3]))]

        device = self.model_engine_online_P.device
        states = [s.to(device) for s in states]
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = [s.to(device) for s in next_states]

        # Update Critic
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.model_engine_online_P.module.sample(next_states)
            q1_next_target = self.model_engine_target_Q1(next_states, next_state_action)
            q2_next_target = self.model_engine_target_Q2(next_states, next_state_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * min_q_next_target

        q1 = self.model_engine_online_Q1(states, actions)
        q2 = self.model_engine_online_Q2(states, actions)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)

        self.model_engine_online_Q1.backward(q1_loss)
        self.model_engine_online_Q1.step()
        self.model_engine_online_Q2.backward(q2_loss)
        self.model_engine_online_Q2.step()

        self.critic_loss = (reduce_losses([q1_loss]).mean() + reduce_losses([q2_loss]).mean()).item() / 2

        # Update Actor
        pi, log_pi = self.model_engine_online_P.module.sample(states)
        q1_pi = self.model_engine_online_Q1(states, pi)
        q2_pi = self.model_engine_online_Q2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.model_engine_online_P.backward(actor_loss)
        self.model_engine_online_P.step()
        self.actor_loss = reduce_losses([actor_loss]).mean().item()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self.soft_update(self.model_engine_online_Q1.module, self.model_engine_target_Q1.module)
        self.soft_update(self.model_engine_online_Q2.module, self.model_engine_target_Q2.module)

    def gen_action(self, state, training=True):
        self.model_engine_online_P.eval()
        device = self.model_engine_online_P.device
        with torch.no_grad():
            state_to_use = self.cur_state if training and self.cur_state is not None else state
            if state_to_use is None:  # Fallback for initial state
                return np.random.dirichlet(np.ones(self.action_dim)).tolist()

            action_, _ = self.model_engine_online_P.module.sample(
                [s.clone().detach().view(1, -1).to(device) for s in state_to_use])

        action = action_.cpu().detach().numpy().flatten()
        self.data_weights = action.tolist()
        return self.data_weights

    def get_weights(self):
        if self.warmup_flag:
            return self.warmup_weighter()

        if self.cur_state:
            self.data_weights = self.gen_action(self.cur_state, training=True)
        else:
            # Fallback to random weights if state is not initialized
            self.data_weights = np.random.dirichlet(np.ones(len(self.dataset_names))).tolist()

        data_weights_tensor = torch.tensor(self.data_weights).to(torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.all_reduce(data_weights_tensor)
            data_weights_tensor /= torch.distributed.get_world_size()

        self.data_weights = data_weights_tensor.cpu().detach().tolist()
        return self.data_weights

    def save_ckpts(self, neox_args, iteration):
        save_dir = os.path.join(neox_args.hds["save_path"], f"model_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        self.model_engine_online_Q1.save_checkpoint(save_dir, tag="online_Q1")
        self.model_engine_online_Q2.save_checkpoint(save_dir, tag="online_Q2")
        self.model_engine_online_P.save_checkpoint(save_dir, tag="online_P")
        self.model_engine_target_Q1.save_checkpoint(save_dir, tag="target_Q1")
        self.model_engine_target_Q2.save_checkpoint(save_dir, tag="target_Q2")
        torch.save(self.log_alpha, os.path.join(save_dir, 'log_alpha.pt'))
        print("All models and optimizers have been saved successfully.")

    def load_ckpts(self, neox_args):
        load_dir = neox_args.hds["load_path"]
        self.model_engine_online_Q1.load_checkpoint(load_dir, tag="online_Q1", load_optimizer_states=False,
                                                    load_lr_scheduler_states=False)
        self.model_engine_online_Q2.load_checkpoint(load_dir, tag="online_Q2", load_optimizer_states=False,
                                                    load_lr_scheduler_states=False)
        self.model_engine_online_P.load_checkpoint(load_dir, tag="online_P", load_optimizer_states=False,
                                                   load_lr_scheduler_states=False)
        self.model_engine_target_Q1.load_checkpoint(load_dir, tag="target_Q1", load_optimizer_states=False,
                                                    load_lr_scheduler_states=False)
        self.model_engine_target_Q2.load_checkpoint(load_dir, tag="target_Q2", load_optimizer_states=False,
                                                    load_lr_scheduler_states=False)

        log_alpha_path = os.path.join(load_dir, 'log_alpha.pt')
        if os.path.exists(log_alpha_path):
            self.log_alpha = torch.load(log_alpha_path)

        print("All models and optimizers have been loaded successfully.")


def load_hds(neox_args, model):
    selected_params_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in neox_args.hds["selected_weights_name"]:
            selected_params_num += param.numel()
    if "selected_params_num" not in neox_args.hds.keys():
        neox_args.hds["selected_params_num"] = selected_params_num
    if neox_args.hds["selected_params_num"] == 0 or neox_args.hds["selected_params_num"] is None:
        raise NotImplementedError("Must have selected weights name for hds model!")
    if "datasets_names" not in neox_args.hds.keys():
        neox_args.hds["datasets_names"] = []
        for train_path in neox_args.train_data_paths:
            name = train_path.split("/")[-2]
            neox_args.hds["datasets_names"].append(name)
        neox_args.hds["datasets_names"] = list(OrderedDict.fromkeys(neox_args.hds["datasets_names"]))
    ac_agent = hds(neox_args)
    if "load_path" in neox_args.hds.keys():
        ac_agent.load_ckpts(neox_args=neox_args)
    return ac_agent, neox_args


class SACWeightUpdater:
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
    ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        total_weights = np.sum(weights)
        self._probabilities = {name: weight / total_weights for name, weight in zip(dataset_names, weights)}
        self.eps = 1 / self.num_datasets
        self.prev_eps = None

    def group_update(self, iteration, state, agent, training):
        return agent.gen_action(state, training)