import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import grad
from mat.utils.util import get_gard_norm, huber_loss, mse_loss, quantile_huber_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check, average_agent_encoders_by_adj, average_attention_params_by_adj

class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents
        self.args = args
        self.num_quants = args.n_quants
        self.n_embd = args.n_embd
        self.truelyDistributed = policy.truelyDistributed
        self.consensusLoss = args.consensusLoss and args.algorithm_name in {"mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd"}
        self.critic_consensus = args.algorithm_name == "consensus_ippo"
        self.avg_critic = args.avg_critic

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.mini_batch_size = args.mini_batch_size
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        # self.gnn_loss_coef = args.gnn_loss_coef
        self.gnn_loss_coef = torch.tensor(args.gnn_loss_coef, dtype=torch.float32, device=device)
        self.consensus_tau = args.consensus_tau
        self.consensus_steps = args.consensus_steps

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self.detach = args.detach

        self.eye = torch.eye(self.num_agents, device=device).unsqueeze(0)

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(self.num_agents, self.num_quants, norm_axes=0, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.num_quants == 1:
            #print('Loss using MSE')
            if self._use_huber_loss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)
        else:
            #print('Loss using quantile_huber_loss')
            if self._use_valuenorm:
                # value_loss_clipped = quantile_huber_loss(self.value_normalizer.normalize(return_batch), value_pred_clipped)
                # value_loss_original = quantile_huber_loss(self.value_normalizer.normalize(return_batch), values)
                value_loss_clipped = quantile_huber_loss(return_batch, value_pred_clipped)
                value_loss_original = quantile_huber_loss(return_batch, values)
            else:
                value_loss_clipped = quantile_huber_loss(return_batch, value_pred_clipped)
                value_loss_original = quantile_huber_loss(return_batch, values)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            # Calculate per-agent policy loss with active masks
            value_loss = (value_loss * active_masks_batch).sum(dim=0) / active_masks_batch.sum(dim=0)
        else:
            # Calculate per-agent policy loss without active masks
            value_loss = value_loss.mean(dim=0)

        return value_loss


    def gnn_consensus_loss(self, x, edge_index):
        total_consensus_loss = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)

        for i in range(self.num_agents):
            # Create mask for valid edges (where edge_index[:, 1, :] != -1)
            valid_mask = (edge_index[:, 1, :] == i) & (edge_index[:, 1, :] != -1)  # [batch_size, max_edges]

            # Get valid source and target indices
            source_indices = edge_index[:, 0, :][valid_mask].long()  # [total_valid_edges]
            edge_indices = edge_index[:, 1, :][valid_mask].long()    # [total_valid_edges]

            # If no valid edges, return zero
            if source_indices.numel() == 0:
                return torch.tensor(0.0, device=edge_index.device)

            # Gather source and edge observations
            source_x = x[:, source_indices, :]
            if self.detach:
                edge_x = x[:, edge_indices, :].detach()
            else:
                edge_x = x[:, edge_indices, :]

            # Compute MSE loss
            total_consensus_loss[i] = F.mse_loss(source_x, edge_x, reduction='mean')

        return total_consensus_loss.unsqueeze(-1)

    def adj_gnn_consensus_loss(self, x, adj):
        """
        x:   [batch_size, n_agents, feature_dim]
        adj: [batch_size, n_agents, n_agents] (0/1 adjacency matrix)
        returns: [n_agents, 1] cosine consensus loss per agent
        """
        # Include self-loops without mutating the sampled adjacency batch.
        eye = self.eye.to(device=adj.device, dtype=adj.dtype)
        adj = (adj + eye).clamp(max=1.0).float()

        # Expand for pairwise cosine distances.
        x_i = x.unsqueeze(2)  # [batch_size, n_agents, 1, feature_dim]
        x_j = x.unsqueeze(1)  # [batch_size, 1, n_agents, feature_dim]

        if self.detach:
            x_j = x_j.detach()

        cosine_sim = F.cosine_similarity(x_i, x_j, dim=-1, eps=1e-8)
        cosine_loss = 1.0 - cosine_sim              # [batch_size, n_agents, n_agents]

        # Mask with adjacency
        masked_loss = cosine_loss * adj             # [batch_size, n_agents, n_agents]

        # Average over each agent's neighbors.
        neighbor_counts = adj.sum(dim=-1)           # [batch_size, n_agents]
        per_agent_loss = masked_loss.sum(dim=-1) / neighbor_counts.clamp_min(1.0)

        # Average across batch
        per_agent_loss = per_agent_loss.mean(dim=0) # [n_agents]

        return per_agent_loss.unsqueeze(-1)         # [n_agents, 1]

    def consensus_weight_matrix(self, adj):
        """
        Build a row-stochastic neighbor mixing matrix from a sampled adjacency batch.
        Empty graphs fall back to all-to-all mixing so the baseline does not silently
        become plain IPPO when a runner has no graph.
        """
        if adj is None:
            adj = torch.ones(self.num_agents, self.num_agents, dtype=torch.float32, device=self.device)
        else:
            adj = adj.to(dtype=torch.float32, device=self.device)
            if adj.dim() == 3:
                adj = adj.mean(dim=0)
            elif adj.dim() != 2:
                adj = adj.view(self.num_agents, self.num_agents)

        if adj.shape != (self.num_agents, self.num_agents):
            adj = torch.ones(self.num_agents, self.num_agents, dtype=torch.float32, device=self.device)

        eye = torch.eye(self.num_agents, dtype=adj.dtype, device=adj.device)
        adj = (adj + eye).clamp(max=1.0)
        if torch.count_nonzero(adj).item() == self.num_agents:
            adj = torch.ones_like(adj)

        row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adj / row_sum

    def critic_parameter_disagreement(self, adj):
        if not hasattr(self.policy.transformer.encoder, "head_"):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        modules = self.policy.transformer.encoder.head_
        weights = self.consensus_weight_matrix(adj)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        norm = torch.zeros((), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            params = [[p.detach() for p in module.parameters()] for module in modules]
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i == j:
                        continue
                    w_ij = weights[i, j]
                    if w_ij <= 0:
                        continue
                    for p_i, p_j in zip(params[i], params[j]):
                        total = total + w_ij * (p_i - p_j).pow(2).mean()
                    norm = norm + w_ij

        return total / norm.clamp_min(1e-8)

    def apply_critic_consensus(self, adj):
        if not hasattr(self.policy.transformer.encoder, "head_"):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        modules = self.policy.transformer.encoder.head_
        weights = self.consensus_weight_matrix(adj)
        tau = float(self.consensus_tau)

        with torch.no_grad():
            for _ in range(max(1, int(self.consensus_steps))):
                snapshots = [[p.detach().clone() for p in module.parameters()] for module in modules]
                for i, module in enumerate(modules):
                    for param_idx, param in enumerate(module.parameters()):
                        mixed = torch.zeros_like(param.data)
                        for j in range(self.num_agents):
                            mixed = mixed + weights[i, j] * snapshots[j][param_idx]
                        param.data.mul_(1.0 - tau).add_(mixed, alpha=tau)

        return self.critic_parameter_disagreement(adj)

    def module_list_parameter_disagreement(self, modules, adj):
        """Weighted neighbor disagreement for identically structured agent modules."""

        weights = self.consensus_weight_matrix(adj)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        norm = torch.zeros((), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            snapshots = [[p.detach() for p in module.parameters()] for module in modules]
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i == j or weights[i, j] <= 0:
                        continue
                    for parameter_i, parameter_j in zip(snapshots[i], snapshots[j]):
                        total = total + weights[i, j] * (
                            parameter_i - parameter_j
                        ).square().mean()
                    norm = norm + weights[i, j]
        return total / norm.clamp_min(1e-8)

    def apply_sr_parameter_consensus(self, adj):
        """Apply the same graph-neighbor D-SGD mixing to every SR agent block."""

        if adj is None:
            consensus_adj = torch.ones(
                self.num_agents,
                self.num_agents,
                dtype=torch.float32,
                device=self.device,
            )
        elif adj.dim() == 3:
            consensus_adj = adj.float().mean(dim=0)
        else:
            consensus_adj = adj.float()

        module_lists = [
            *self.policy.transformer.obs_encoder.consensus_module_lists(),
            self.policy.transformer.encoder.head_,
            self.policy.transformer.decoder.mlp_,
        ]
        for modules in module_lists:
            average_agent_encoders_by_adj(modules, consensus_adj)

        disagreements = [
            self.module_list_parameter_disagreement(modules, consensus_adj)
            for modules in module_lists
        ]
        return torch.stack(disagreements).mean()


    def ppo_update(self, sample, episode, iter_step, obs_dim=None):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, adjcency_matrix_batch, next_obs_batch, edge_index_batch = sample

        # Convert all inputs to proper device and dtype
        def ensure_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(**self.tpdv)
            return x.to(**self.tpdv) if isinstance(x, torch.Tensor) else x

        old_action_log_probs_batch = ensure_tensor(old_action_log_probs_batch)
        adv_targ = ensure_tensor(adv_targ)
        value_preds_batch = ensure_tensor(value_preds_batch)
        return_batch = ensure_tensor(return_batch)
        active_masks_batch = ensure_tensor(active_masks_batch)
        obs_batch = ensure_tensor(obs_batch)
        edge_index_batch = ensure_tensor(edge_index_batch)
        next_obs_batch = ensure_tensor(next_obs_batch)
        share_obs_batch = ensure_tensor(share_obs_batch)
        rnn_states_batch = ensure_tensor(rnn_states_batch)
        rnn_states_critic_batch = ensure_tensor(rnn_states_critic_batch)
        actions_batch = ensure_tensor(actions_batch)
        masks_batch = ensure_tensor(masks_batch)

        if adjcency_matrix_batch is not None:
            adjcency_matrix_batch = ensure_tensor(adjcency_matrix_batch)
            adjcency_matrix_batch = adjcency_matrix_batch.view(-1, self.num_agents, adjcency_matrix_batch.shape[-1])

        if available_actions_batch is not None:
            available_actions_batch = ensure_tensor(available_actions_batch)
            available_actions_batch = available_actions_batch.view(-1, self.num_agents, available_actions_batch.shape[-1])

        # Reshape tensors to [batch_size, num_agents, ...]
        old_action_log_probs_batch = old_action_log_probs_batch.view(-1, self.num_agents, old_action_log_probs_batch.shape[-1])
        adv_targ = adv_targ.view(-1, self.num_agents, adv_targ.shape[-1])
        value_preds_batch = value_preds_batch.view(-1, self.num_agents, value_preds_batch.shape[-1])
        return_batch = return_batch.view(-1, self.num_agents, return_batch.shape[-1])
        active_masks_batch = active_masks_batch.view(-1, self.num_agents, active_masks_batch.shape[-1])

        obs_batch = obs_batch.view(-1, self.num_agents, obs_batch.shape[-1])
        edge_index_batch = edge_index_batch.view(-1, 2, edge_index_batch.shape[-1])
        next_obs_batch = next_obs_batch.view(-1, self.num_agents, next_obs_batch.shape[-1])
        share_obs_batch = share_obs_batch.view(-1, self.num_agents, share_obs_batch.shape[-1])
        rnn_states_batch = rnn_states_batch.view(-1, self.num_agents, rnn_states_batch.shape[-1])
        actions_batch = actions_batch.view(-1, self.num_agents, actions_batch.shape[-1])
        masks_batch = masks_batch.view(-1, self.num_agents, masks_batch.shape[-1])
        rnn_states_critic_batch = rnn_states_critic_batch.view(-1, self.num_agents, rnn_states_critic_batch.shape[-1])

        # Evaluate actions for this agent only
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                graph_context=(
                    adjcency_matrix_batch
                    if self.policy.algorithm_name in {"sr_mappo", "sr_mappo_shared"}
                    else None
                ),
            )

        if action_log_probs.shape[-1] != 1:
            action_log_probs = action_log_probs.sum(-1, keepdim=True)
        if old_action_log_probs_batch.shape[-1] != 1:
            old_action_log_probs_batch = old_action_log_probs_batch.sum(-1, keepdim=True)
        if dist_entropy.shape[-1] != 1:
            dist_entropy = dist_entropy.sum(-1, keepdim=True)

        # Calculate policy loss for this agent
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            min_surr = torch.min(surr1, surr2)
            masked_min_surr = min_surr * active_masks_batch
            policy_loss = -masked_min_surr.sum(dim=0) / (active_masks_batch.sum(dim=0) + 1e-8)
        else:
            policy_loss = -torch.min(surr1, surr2).mean(dim=0)

        # Calculate value loss for this agent
        value_loss = self.cal_value_loss(
                values,
                value_preds_batch,
                return_batch,
                active_masks_batch
            )

        if self.consensusLoss:
            gnn_consensus_loss = self.adj_gnn_consensus_loss(obs_batch[:,:,self.policy.obs_dim:], adjcency_matrix_batch)
            loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + self.gnn_loss_coef * gnn_consensus_loss
        else:
            gnn_consensus_loss = torch.zeros(self.num_agents, 1, dtype=torch.float32, device=self.device)
            loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        auxiliary_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        auxiliary_metrics = {}
        if self.policy.algorithm_name in {"sr_mappo", "sr_mappo_shared"}:
            auxiliary_loss = self.policy.transformer.auxiliary_loss()
            auxiliary_metrics = self.policy.transformer.auxiliary_metrics()
            loss = loss + auxiliary_loss

        # We'll store losses for logging
        policy_losses = []
        value_losses = []

        total_value_loss = 0
        total_policy_loss = 0
        total_gnn_loss = 0

        if self.truelyDistributed:
            # Zero all gradients first
            for optimizer in self.policy.optimizers:
                optimizer.zero_grad()

            # Pre-compute all gradients in parallel
            all_grads = []
            for i in range(self.num_agents):
                grads = torch.autograd.grad(
                    outputs=loss[i],
                    inputs=self.policy.agent_parameters(i),
                    retain_graph=True,  # Need to keep graph for parallel
                    create_graph=False,
                    allow_unused=True
                )
                all_grads.append(grads)

            # Apply gradients and step optimizers
            grad_norms = []
            for i, grads in enumerate(all_grads):
                for param, grad in zip(self.policy.agent_parameters(i), grads):
                    if grad is not None:
                        param.grad = grad
                if self._use_max_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.agent_parameters(i), self.max_grad_norm)
                else:
                    grad_norm = get_gard_norm(self.policy.agent_parameters(i))
                grad_norms.append(torch.as_tensor(grad_norm, device=self.device))
                self.policy.optimizers[i].step()
            grad_norm = torch.stack(grad_norms).mean()

        else:
            # Total loss for this agent
            loss = loss.mean()

            # Zero gradients, backward pass, and optimizer step
            self.policy.optimizers.zero_grad()
            loss.backward()

            if self._use_max_grad_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.transformer.parameters(),
                    self.max_grad_norm
                )
            else:
                grad_norm = get_gard_norm(self.policy.transformer.parameters())

            self.policy.optimizers.step()

        critic_consensus_error = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.critic_consensus:
            critic_consensus_error = self.apply_critic_consensus(adjcency_matrix_batch)

        if self.policy.algorithm_name == "sr_mappo":
            auxiliary_metrics["sr_parameter_consensus_error"] = (
                self.apply_sr_parameter_consensus(adjcency_matrix_batch).detach()
            )

        for _ in range(1):
            if self.policy.algorithm_name == 'mappo_dgnn_dsgd':
                average_agent_encoders_by_adj(self.policy.transformer.obs_encoder.agent_encoders, adjcency_matrix_batch[0])
                average_agent_encoders_by_adj(self.policy.transformer.obs_encoder.node_classifier_heads, adjcency_matrix_batch[0])
                average_attention_params_by_adj(self.policy.transformer.obs_encoder.atts, adjcency_matrix_batch[0])

                # average_attention_params_by_adj(self.policy.transformer.obs_encoder.hop_atts, adjcency_matrix_batch[0])
                # average_attention_params_by_adj(self.policy.transformer.obs_encoder.hop_biases, adjcency_matrix_batch[0])
                average_agent_encoders_by_adj(self.policy.transformer.encoder.head_, adjcency_matrix_batch[0])
                average_agent_encoders_by_adj(self.policy.transformer.decoder.mlp_, adjcency_matrix_batch[0])


        # Return average losses across agents
        avg_value_loss = sum(value_loss) / len(value_loss)
        avg_policy_loss = sum(policy_loss) / len(policy_loss)
        avg_gnn_consensus_loss = sum(gnn_consensus_loss) / len(gnn_consensus_loss)
        avg_grad_norm = grad_norm #sum(grad_norms) / len(grad_norms)

        return (
            avg_value_loss,
            avg_grad_norm,
            avg_policy_loss,
            dist_entropy.mean().item(),
            avg_grad_norm,
            imp_weights,
            avg_gnn_consensus_loss,
            critic_consensus_error,
            auxiliary_loss.detach().mean(),
            auxiliary_metrics,
        )

    def train(self, buffer, episode, obs_dim=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.clone()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = torch.nan

        valid_advantages = advantages_copy[~torch.isnan(advantages_copy)]
        mean_advantages = valid_advantages.mean()
        std_advantages = valid_advantages.std(unbiased=False)

        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['gnn_consensus_loss'] = 0
        train_info['critic_consensus_error'] = 0
        train_info['sr_auxiliary_loss'] = 0
        num_updates = 0

        for i in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch, mini_batch_size=self.mini_batch_size)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, avg_gnn_consensus_loss, critic_consensus_error, auxiliary_loss, auxiliary_metrics \
                    = self.ppo_update(sample, episode, i, obs_dim)

                train_info['value_loss'] += value_loss
                train_info['policy_loss'] += policy_loss
                train_info['dist_entropy'] += dist_entropy
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['gnn_consensus_loss'] += avg_gnn_consensus_loss
                train_info['critic_consensus_error'] += critic_consensus_error
                train_info['sr_auxiliary_loss'] += auxiliary_loss
                for key, value in auxiliary_metrics.items():
                    if key not in train_info:
                        train_info[key] = 0
                    train_info[key] += value
                num_updates += 1

        num_updates = max(num_updates, 1)

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
