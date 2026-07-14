import torch
from mat.utils.util import update_linear_schedule
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.utils.util import check


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.num_quants = args.n_quants
        self.n_embd = args.n_embd
        self.truelyDistributed = args.truelyDistributed
        self.clone_extra_agents_from = getattr(args, "clone_extra_agents_from", None)

        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.obs_dim_ = self.obs_dim

        if self.algorithm_name in {"mappo_dgnn", "mappo_dgnn_dsgd"}:
            from mat.algorithms.mat.algorithm.ma_gnn_transformer_new import MultiAgentGnnTransformer as MAT
            if args.iterations > 0:
                self.obs_dim_ = self.obs_dim+args.n_embd

        elif self.algorithm_name in {"sr_mappo", "sr_mappo_shared"}:
            from mat.algorithms.mat.algorithm.sr_mappo_transformer import SymmetryReducedMAPPO as MAT
            # SR communication is recomputed inside the policy during PPO updates,
            # so the replay buffer stores the raw local observation.
            self.obs_dim_ = self.obs_dim

        else:
            raise NotImplementedError(f"Unsupported algorithm: {self.algorithm_name}")

        self.transformer = MAT(args, self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, device=device,
                               action_type=self.action_type, dec_actor=args.dec_actor,
                               share_actor=args.share_actor, num_quants=args.n_quants)

        if args.env_name == "hands":
            self.transformer.zero_std()

        if not self.truelyDistributed:
            self.optimizers = torch.optim.Adam(self.transformer.parameters(),
                                            lr=self.lr, eps=self.opti_eps,
                                            weight_decay=self.weight_decay)
        else:
            if self.algorithm_name in {"mappo_dgnn", "mappo_dgnn_dsgd"}:
                self.optimizers = [
                    torch.optim.Adam(
                        list(self.transformer.decoder.mlp_[i].parameters()) +
                        list(self.transformer.encoder.head_[i].parameters()) +
                        list(self.transformer.obs_encoder.agent_encoders[i].parameters()) +
                        list(self.transformer.obs_encoder.node_classifier_heads[i].parameters()) +
                        [self.transformer.obs_encoder.atts[k][i] for k in range(self.transformer.obs_encoder.K)],
                        # [self.transformer.obs_encoder.hop_atts[k][i] for k in range(self.transformer.obs_encoder.K)] +
                        # [self.transformer.obs_encoder.hop_biases[k][i] for k in range(self.transformer.obs_encoder.K)],
                        lr=self.lr,
                        eps=self.opti_eps,
                        weight_decay=self.weight_decay,
                    )
                    for i in range(self.num_agents)
                ]
            elif self.algorithm_name == "sr_mappo":
                self.optimizers = [
                    torch.optim.Adam(
                        self.agent_parameters(i),
                        lr=self.lr,
                        eps=self.opti_eps,
                        weight_decay=self.weight_decay,
                    )
                    for i in range(self.num_agents)
                ]
            else:
                raise ValueError(
                    "Distributed optimizer mode is supported only for "
                    "mappo_dgnn and sr_mappo"
                )

    def agent_parameters(self, agent_idx):
        """Returns all parameters specific to one agent"""
        params = []

        # MLP and head parameters
        params.extend(list(self.transformer.decoder.mlp_[agent_idx].parameters()))
        params.extend(list(self.transformer.encoder.head_[agent_idx].parameters()))

        if self.algorithm_name == "sr_mappo":
            params.extend(
                list(self.transformer.obs_encoder.agent_codecs[agent_idx].parameters())
            )
            params.extend(
                list(self.transformer.obs_encoder.agent_fusers[agent_idx].parameters())
            )
            params.extend(
                list(self.transformer.obs_encoder.agent_readouts[agent_idx].parameters())
            )
            return params

        # Original D-GAT observation encoder components.
        params.extend(list(self.transformer.obs_encoder.agent_encoders[agent_idx].parameters()))
        params.extend(list(self.transformer.obs_encoder.node_classifier_heads[agent_idx].parameters()))
        params.extend([self.transformer.obs_encoder.atts[k][agent_idx] for k in range(self.transformer.obs_encoder.K)])
        # params.extend([self.transformer.obs_encoder.hop_atts[k][agent_idx] for k in range(self.transformer.obs_encoder.K)])
        # params.extend([self.transformer.obs_encoder.hop_biases[k][agent_idx] for k in range(self.transformer.obs_encoder.K)])

        return params

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        optimizers = self.optimizers if isinstance(self.optimizers, (list, tuple)) else [self.optimizers]
        update_linear_schedule(optimizers, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    batched_edge_index=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim_)

        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if batched_edge_index is None:
            actions, action_log_probs, values = self.transformer.get_actions(cent_obs,
                                                                            obs,
                                                                            available_actions,
                                                                            deterministic)
            actions = actions.view(-1, self.act_num)
            action_log_probs = action_log_probs.view(-1, self.act_num)
            values = values.view(-1, self.num_quants)

            # unused, just for compatibility
            rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
            rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

        else:
            actions, action_log_probs, values = self.transformer.get_actions(cent_obs,
                                                                            obs,
                                                                            available_actions,
                                                                            deterministic,
                                                                            batched_edge_index)
            actions = actions.view(-1, self.act_num)
            action_log_probs = action_log_probs.view(-1, self.act_num)
            values = values.view(-1, self.num_quants)

            # unused, just for compatibility
            rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
            rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic


    def get_values(
        self,
        cent_obs,
        obs,
        rnn_states_critic,
        masks,
        available_actions=None,
        graph_context=None,
    ):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim_)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if self.algorithm_name in {"sr_mappo", "sr_mappo_shared"}:
            values = self.transformer.get_values(
                cent_obs, obs, available_actions, graph_context=graph_context
            )
        else:
            values = self.transformer.get_values(cent_obs, obs, available_actions)

        values = values.view(-1, self.num_quants)

        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                         available_actions=None, active_masks=None, agent_id=None, action_hats=None,
                         graph_context=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim_)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        # action_hats = action_hats.reshape(-1, self.num_agents, action_hats.shape[-1])

        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if self.algorithm_name in {"sr_mappo", "sr_mappo_shared"}:
            action_log_probs, values, entropy = self.transformer(
                cent_obs,
                obs,
                actions,
                available_actions,
                graph_context=graph_context,
            )
        else:
            action_log_probs, values, entropy = self.transformer(
                cent_obs, obs, actions, available_actions
            )

        action_log_probs = action_log_probs.view(-1, self.num_agents, self.act_num)
        values = values.view(-1, self.num_agents, self.num_quants)
        entropy = entropy.view(-1, self.num_agents, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum(dim=0)/active_masks.sum(dim=0)
            # entropy = entropy.mean(dim=0)
        else:
            entropy = entropy.mean(dim=0)

        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True, batched_edge_index=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        rnn_states_critic = torch.zeros_like(rnn_states_actor)
        if batched_edge_index is None:
            _, actions, _, rnn_states_actor, _ = self.get_actions(cent_obs,
                                                                obs,
                                                                rnn_states_actor,
                                                                rnn_states_critic,
                                                                masks,
                                                                available_actions,
                                                                None,
                                                                deterministic)
        else:
            _, actions, _, rnn_states_actor, _ = self.get_actions(cent_obs,
                                                                obs,
                                                                rnn_states_actor,
                                                                rnn_states_critic,
                                                                masks,
                                                                available_actions,
                                                                batched_edge_index,
                                                                deterministic)

        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def _infer_checkpoint_agent_count(self, state_dict):
        max_idx = -1
        prefixes = (
            "decoder.mlp_.",
            "encoder.head_.",
            "obs_encoder.agent_encoders.",
            "obs_encoder.node_classifier_heads.",
        )

        for key in state_dict.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    tail = key[len(prefix):]
                    idx_str = tail.split(".", 1)[0]
                    if idx_str.isdigit():
                        max_idx = max(max_idx, int(idx_str))

        return max_idx + 1 if max_idx >= 0 else None

    def _clone_dgnn_agent_slot(self, src_agent, dst_agent):
        with torch.no_grad():
            self.transformer.decoder.mlp_[dst_agent].load_state_dict(
                self.transformer.decoder.mlp_[src_agent].state_dict()
            )
            self.transformer.encoder.head_[dst_agent].load_state_dict(
                self.transformer.encoder.head_[src_agent].state_dict()
            )
            self.transformer.obs_encoder.agent_encoders[dst_agent].load_state_dict(
                self.transformer.obs_encoder.agent_encoders[src_agent].state_dict()
            )
            self.transformer.obs_encoder.node_classifier_heads[dst_agent].load_state_dict(
                self.transformer.obs_encoder.node_classifier_heads[src_agent].state_dict()
            )

            if hasattr(self.transformer.obs_encoder, "atts"):
                for k in range(len(self.transformer.obs_encoder.atts)):
                    if src_agent < len(self.transformer.obs_encoder.atts[k]) and dst_agent < len(self.transformer.obs_encoder.atts[k]):
                        self.transformer.obs_encoder.atts[k][dst_agent].copy_(
                            self.transformer.obs_encoder.atts[k][src_agent]
                        )

            if hasattr(self.transformer.obs_encoder, "hop_atts"):
                for k in range(len(self.transformer.obs_encoder.hop_atts)):
                    if src_agent < len(self.transformer.obs_encoder.hop_atts[k]) and dst_agent < len(self.transformer.obs_encoder.hop_atts[k]):
                        self.transformer.obs_encoder.hop_atts[k][dst_agent].copy_(
                            self.transformer.obs_encoder.hop_atts[k][src_agent]
                        )

            if hasattr(self.transformer.obs_encoder, "hop_biases"):
                for k in range(len(self.transformer.obs_encoder.hop_biases)):
                    if src_agent < len(self.transformer.obs_encoder.hop_biases[k]) and dst_agent < len(self.transformer.obs_encoder.hop_biases[k]):
                        self.transformer.obs_encoder.hop_biases[k][dst_agent].copy_(
                            self.transformer.obs_encoder.hop_biases[k][src_agent]
                        )

    def restore(self, model_dir, allow_partial=False):
        transformer_state_dict = torch.load(model_dir, weights_only=True)

        if not allow_partial:
            self.transformer.load_state_dict(transformer_state_dict)
            return

        current_state_dict = self.transformer.state_dict()
        filtered_state_dict = {}
        skipped = []
        adapted = []

        for key, value in transformer_state_dict.items():
            if key not in current_state_dict:
                skipped.append((key, "missing_key"))
                continue

            model_tensor = current_state_dict[key]

            if model_tensor.shape != value.shape:
                # Keep registered attention masks from the target model.
                if key.endswith(".mask"):
                    skipped.append(
                        (
                            key,
                            f"shape_mismatch_mask ckpt={tuple(value.shape)} model={tuple(model_tensor.shape)}",
                        )
                    )
                    continue

                if model_tensor.dim() != value.dim():
                    skipped.append(
                        (
                            key,
                            f"shape_mismatch_rank ckpt={tuple(value.shape)} model={tuple(model_tensor.shape)}",
                        )
                    )
                    continue

                # Build a model-shaped tensor filled with zeros, then copy overlap.
                patched = torch.zeros_like(model_tensor)
                src = value.to(device=patched.device, dtype=patched.dtype)
                overlap_slices = tuple(
                    slice(0, min(model_tensor.size(d), src.size(d))) for d in range(model_tensor.dim())
                )
                patched[overlap_slices] = src[overlap_slices]
                filtered_state_dict[key] = patched
                adapted.append((key, tuple(value.shape), tuple(model_tensor.shape)))
                continue

            filtered_state_dict[key] = value

        missing, unexpected = self.transformer.load_state_dict(filtered_state_dict, strict=False)

        print(
            f"[restore] partial load enabled: loaded {len(filtered_state_dict)} tensors; "
            f"adapted {len(adapted)} mismatched tensors; "
            f"skipped {len(skipped)} mismatched/missing tensors; "
            f"missing_after_load={len(missing)}, unexpected_after_load={len(unexpected)}"
        )

        if adapted:
            print("[restore] first adapted keys:")
            for key, ckpt_shape, model_shape in adapted[:20]:
                print(f"  - {key}: ckpt={ckpt_shape} -> model={model_shape} (zero-filled where needed)")
            if len(adapted) > 20:
                print(f"  ... and {len(adapted) - 20} more.")

        if skipped:
            print("[restore] first skipped keys:")
            for key, reason in skipped[:20]:
                print(f"  - {key}: {reason}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more.")

        # Optional: clone one trained agent into newly added slots for non-shared DGNN policies.
        if self.algorithm_name in ("mappo_dgnn", "mappo_dgnn_dsgd") and self.clone_extra_agents_from is not None:
            ckpt_agents = self._infer_checkpoint_agent_count(transformer_state_dict)
            if ckpt_agents is not None and self.num_agents > ckpt_agents:
                src_agent = int(self.clone_extra_agents_from)
                if src_agent < 0 or src_agent >= ckpt_agents:
                    print(
                        f"[restore] clone_extra_agents_from={src_agent} is outside checkpoint range [0, {ckpt_agents - 1}]. "
                        "Falling back to source agent 0."
                    )
                    src_agent = 0

                cloned_slots = []
                for dst_agent in range(ckpt_agents, self.num_agents):
                    self._clone_dgnn_agent_slot(src_agent, dst_agent)
                    cloned_slots.append(dst_agent)

                print(
                    f"[restore] cloned DGNN agent {src_agent} into new agent slots {cloned_slots} "
                    f"(checkpoint_agents={ckpt_agents}, model_agents={self.num_agents})."
                )

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()
