import time
import csv
import os
import wandb
import numpy as np
from functools import reduce
import torch
import imageio
import gymnasium as gym
import torch.nn.functional as F

from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


def faulty_action(action, faulty_node):
    action_fault = action.copy()
    if faulty_node >= 0:
        action_fault[:, faulty_node, :] = 0.
        # action[:, faulty_node, :] = 0.
    # return action
    return action_fault

GRAPH_ALGORITHMS = {"mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd", "sr_mappo", "consensus_ippo"}
GNN_ALGORITHMS = {"mappo_gnn", "mappo_dgnn", "mappo_dgnn_dsgd"}


class MAGoToGoalRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(MAGoToGoalRunner, self).__init__(config)
        self.reward_list = []
        self.reward_list_training = []
        self.plt_name = f"{self.env_name}_{self.algorithm_name}"
        self.eye = torch.eye(self.num_agents, device=self.device).unsqueeze(0)
        self.eye = self.eye / torch.norm(self.eye, p='fro')  # Normalize entire matrix
        # Accumulate episode-level eval stats across repeated eval() calls in this run.
        self._agg_reached_counts_by_fault = {}
        self._agg_collision_counts_by_fault = {}

    def _flatten_info_dicts(self, obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, np.ndarray):
            out = []
            for x in obj.flat:
                out.extend(self._flatten_info_dicts(x))
            return out
        if isinstance(obj, (list, tuple)):
            out = []
            for x in obj:
                out.extend(self._flatten_info_dicts(x))
            return out
        return []

    def _update_episode_team_flags(self, eval_infos, reached_flags, collision_flags):
        for env_i, env_info in enumerate(eval_infos):
            agent_infos = self._flatten_info_dicts(env_info)
            max_agents = min(self.num_agents, len(agent_infos))
            for agent_i in range(max_agents):
                info_i = agent_infos[agent_i]
                if not isinstance(info_i, dict):
                    continue
                if "reached_goal" in info_i:
                    reached_flags[env_i, agent_i] = bool(info_i["reached_goal"])
                if "collision" in info_i:
                    collision_flags[env_i, agent_i] = collision_flags[env_i, agent_i] or bool(info_i["collision"])

    def _append_eval_team_stats(self, rows):
        if not rows:
            return

        out_path = os.path.join(str(self.run_dir), "eval_team_stats.csv")
        file_exists = os.path.exists(out_path)
        fieldnames = list(rows[0].keys())

        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _append_eval_predator_prey_stats(self, rows):
        if not rows:
            return

        out_path = os.path.join(str(self.run_dir), "eval_predator_prey_stats.csv")
        file_exists = os.path.exists(out_path)
        fieldnames = list(rows[0].keys())

        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _first_env_info(self, eval_infos, env_i):
        if eval_infos is None or env_i >= len(eval_infos):
            return {}
        env_info = eval_infos[env_i]
        if isinstance(env_info, dict):
            return env_info
        if isinstance(env_info, np.ndarray):
            env_info = env_info.tolist()
        if isinstance(env_info, (list, tuple)):
            return env_info[0] if env_info and isinstance(env_info[0], dict) else {}
        return {}

    def _update_predator_prey_episode_stats(
        self,
        eval_infos,
        capture_counts,
        collision_counts,
        final_prey_remaining,
        capture_success,
        max_capture_group,
    ):
        for env_i in range(len(capture_counts)):
            info = self._first_env_info(eval_infos, env_i)
            if not info:
                continue
            capture_counts[env_i] += int(info.get("new_captures", 0))
            collision_counts[env_i] += int(info.get("collision_count", 0))
            final_prey_remaining[env_i] = int(info.get("prey_remaining", final_prey_remaining[env_i]))
            capture_success[env_i] = capture_success[env_i] or bool(info.get("capture_success", False))
            max_capture_group[env_i] = max(max_capture_group[env_i], int(info.get("max_capture_group", 0)))

    def _update_aggregate_eval_stats(self, faulty_node, reached_counts, collision_counts):
        if faulty_node not in self._agg_reached_counts_by_fault:
            self._agg_reached_counts_by_fault[faulty_node] = []
        if faulty_node not in self._agg_collision_counts_by_fault:
            self._agg_collision_counts_by_fault[faulty_node] = []
        self._agg_reached_counts_by_fault[faulty_node].extend(list(reached_counts))
        self._agg_collision_counts_by_fault[faulty_node].extend(list(collision_counts))

    def _log_ci_bar_chart_to_wandb(self, reached_counts, collision_counts, faulty_node, step):
        if len(reached_counts) == 0 or len(collision_counts) == 0:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        def _mean_ci95(x):
            x = np.asarray(x, dtype=np.float64)
            mean = float(np.mean(x))
            if x.size <= 1:
                return mean, 0.0
            sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
            return mean, 1.96 * sem

        reached_rates = np.asarray(reached_counts, dtype=np.float64) / max(self.num_agents, 1)
        collision_rates = np.asarray(collision_counts, dtype=np.float64) / max(self.num_agents, 1)

        reached_mean, reached_ci = _mean_ci95(reached_rates)
        collision_mean, collision_ci = _mean_ci95(collision_rates)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        labels = ["Reached Goal Rate", "Collision Rate"]
        means = [reached_mean, collision_mean]
        cis = [reached_ci, collision_ci]
        ax.bar(labels, means, yerr=cis, capsize=8, color=["#2A9D8F", "#E76F51"])
        ax.set_ylabel("Rate per Episode")
        ax.set_title(
            f"{self.algorithm_name} | Team {self.num_agents} | Faulty {faulty_node}\n"
            "Mean over eval episodes with 95% CI"
        )
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()

        wandb.log(
            {f"faulty_node_{faulty_node}/eval_counts_ci95_bar": wandb.Image(fig)},
            step=step,
        )
        plt.close(fig)

    def _log_ci_summary_table_to_wandb(self, reached_counts, collision_counts, faulty_node, step):
        if len(reached_counts) == 0 or len(collision_counts) == 0:
            return

        def _mean_ci95(x):
            x = np.asarray(x, dtype=np.float64)
            mean = float(np.mean(x))
            if x.size <= 1:
                return mean, 0.0
            sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
            return mean, 1.96 * sem

        reached_mean, reached_ci = _mean_ci95(reached_counts)
        collision_mean, collision_ci = _mean_ci95(collision_counts)

        reached_rate = np.asarray(reached_counts, dtype=np.float64) / max(self.num_agents, 1)
        collision_rate = np.asarray(collision_counts, dtype=np.float64) / max(self.num_agents, 1)
        reached_rate_mean, reached_rate_ci = _mean_ci95(reached_rate)
        collision_rate_mean, collision_rate_ci = _mean_ci95(collision_rate)

        table = wandb.Table(
            columns=[
                "algorithm_name",
                "team_size",
                "faulty_node",
                "metric",
                "mean",
                "ci95",
                "n_episodes",
                "step",
            ]
        )
        n_eps = int(len(reached_counts))
        rows = [
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "reached_goal_count", reached_mean, reached_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "collision_count", collision_mean, collision_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "reached_goal_rate", reached_rate_mean, reached_rate_ci, n_eps, int(step)],
            [self.algorithm_name, int(self.num_agents), int(faulty_node), "collision_rate", collision_rate_mean, collision_rate_ci, n_eps, int(step)],
        ]
        for row in rows:
            table.add_data(*row)

        # Same key across runs makes cross-run comparison panels straightforward.
        wandb.log({"comparison/eval_ci_summary": table}, step=step)

    def get_batch_edge_index(self, edge_index):
        """
        Converts a padded multi-environment edge index into a single batched edge index.

        Args:
            edge_index: Padded tensor of shape [batch_size, 2, max_edges]
                    (invalid edges marked with -1 in the 2nd row).

        Returns:
            batched_edge_index: Merged edge index of shape [2, total_valid_edges]
        """

        batch_size = edge_index.size(0)
        batched_edges = []

        for i in range(batch_size):
            # Step 1: Remove invalid edges (where edge_index[i, 1, :] == -1)
            valid_mask = edge_index[i, 1, :] != -1
            valid_edges = edge_index[i, :, valid_mask]  # Shape [2, num_valid_edges]

            # Step 2: Apply offset to node indices (avoid collisions across batches)
            valid_edges[0, :] += i * self.num_agents  # Offset source nodes
            valid_edges[1, :] += i * self.num_agents  # Offset target nodes

            # Step 3: Collect all valid edges
            batched_edges.append(valid_edges)

        # Step 4: Concatenate all valid edges into [2, total_edges]
        return torch.cat(batched_edges, dim=1).long()

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                actions_fault = faulty_action(actions.cpu().detach().numpy(), self.all_args.faulty_node)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions_fault)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                share_obs = torch.tensor(share_obs, dtype=torch.float32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
                # available_actions = torch.tensor(available_actions, dtype=torch.float32, device=self.device)

                if self.algorithm_name in GRAPH_ALGORITHMS:
                    edge_index = self.envs.get_edge_index_matrix()
                    edge_index = torch.tensor(edge_index, dtype=torch.float32, device=self.device)
                    batch_edge_index = self.get_batch_edge_index(edge_index)

                    adjcency_matrix = self.envs.get_visibility_matrix()
                    adjcency_matrix = torch.tensor(adjcency_matrix, dtype=torch.float32, device=self.device)

                    data = obs, share_obs, rewards, dones, infos, None, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data, batch_edge_index, edge_index, adjcency_matrix)

                else:
                    data = obs, share_obs, rewards, dones, infos, None, \
                        values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic

                    # insert data into buffer
                    self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train(episode)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)

                    if self.use_wandb:
                        wandb.log({"average_episode_rewards": aver_episode_rewards}, step=total_num_steps)

                    #self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    if self.reward_list_training is None:
                        self.reward_list_training = np.array(np.mean(done_episodes_rewards))
                    else:
                        self.reward_list_training = np.hstack((self.reward_list_training, np.array(np.mean(done_episodes_rewards))))

                    np.save(
                        os.path.join(str(self.run_dir), self.plt_name + "_training.npy"),
                        self.reward_list_training,
                    )

                    done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                # self.eval(total_num_steps, self.all_args.eval_faulty_node[0])
                faulty_nodes = self.all_args.eval_faulty_node
                for node in faulty_nodes:
                    self.eval(total_num_steps, node)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        share_obs = torch.tensor(share_obs, dtype=torch.float32, device=self.device)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        # available_actions = torch.tensor(available_actions, dtype=torch.float32, device=self.device)

        if self.algorithm_name == "consensus_ippo":
            adjcency_matrix = self.envs.get_visibility_matrix()
            adjcency_matrix = torch.tensor(adjcency_matrix, dtype=torch.float32, device=self.device)
            self.buffer.adjcency_matrix[0] = adjcency_matrix.clone()

        if self.algorithm_name == "sr_mappo":
            adjcency_matrix = self.envs.get_visibility_matrix()
            adjcency_matrix = torch.as_tensor(
                adjcency_matrix, dtype=torch.float32, device=self.device
            )
            self.buffer.adjcency_matrix[0] = adjcency_matrix

        if self.algorithm_name in GNN_ALGORITHMS:
            self.buffer.obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.obs_dim+self.n_embd), dtype=torch.float32, device=self.device)

            print('obs shape = ', obs.shape)

            adjcency_matrix = self.envs.get_visibility_matrix()
            adjcency_matrix = torch.tensor(adjcency_matrix, dtype=torch.float32, device=self.device)

            edge_index = self.envs.get_edge_index_matrix()
            edge_index = torch.tensor(edge_index, dtype=torch.float32, device=self.device)
            batch_edge_index = self.get_batch_edge_index(edge_index)

            x = self.trainer.policy.transformer.obs_encoder(obs, batch_edge_index)

            obs = torch.cat([obs,x],dim=-1).detach()

            self.buffer.adjcency_matrix[0] = adjcency_matrix.clone()

        # replay buffer
        if (not self.use_centralized_V) and self.algorithm_name.startswith("mat"):
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.clone()
        self.buffer.obs[0] = obs.clone()
        # self.buffer.available_actions[0] = available_actions.clone()

    @torch.no_grad()
    def collect(self, step, batched_edge_index=None):
        self.trainer.prep_rollout()
        if self.algorithm_name == "sr_mappo":
            batched_edge_index = self.buffer.adjcency_matrix[step]
        value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
                        self.buffer.share_obs[step],
                        self.buffer.obs[step],
                        self.buffer.rnn_states[step],
                        self.buffer.rnn_states_critic[step],
                        self.buffer.masks[step],
                        None,
                        batched_edge_index
                    )

        # [self.envs, agents, dim]
        values = value.reshape(self.n_rollout_threads, self.num_agents, -1)
        actions = action.reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs = action_log_prob.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states = rnn_state.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states_critic = rnn_state_critic.reshape(self.n_rollout_threads, self.num_agents, -1)
        # action_hats = action_hat.reshape(self.n_rollout_threads, self.num_agents, -1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, batched_edge_index=None, edge_index=None, adjcency_matrix=None):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if self.all_args.iterations > 0:
            if self.algorithm_name in GNN_ALGORITHMS:
                x = self.trainer.policy.transformer.obs_encoder(obs, batched_edge_index)
                obs = torch.cat([obs,x],dim=-1).detach()

        dones_env = torch.all(dones, dim=1)

        rnn_states[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device=self.device)
        rnn_states_critic[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device=self.device)

        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1), dtype=torch.float32, device=self.device)
        masks[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device=self.device)

        active_masks = torch.ones((self.n_rollout_threads, self.num_agents, 1), dtype=torch.float32, device=self.device)
        active_masks[dones == True] = torch.zeros(((dones == True).sum(), 1), dtype=torch.float32, device=self.device)
        active_masks[dones_env == True] = torch.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device=self.device)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V and self.algorithm_name.startswith("mat"):
            share_obs = obs

        if self.algorithm_name == "consensus_ippo" and self.all_args.consensus_reward_mode == "mean":
            rewards = rewards.mean(dim=1, keepdim=True).repeat(1, self.num_agents, 1)

        self.buffer.insert(share_obs, obs, rnn_states.unsqueeze(-2), rnn_states_critic.unsqueeze(-2), actions, action_log_probs, values, rewards, masks, None,
                            active_masks, available_actions, edge_index, adjcency_matrix)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = torch.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps, faulty_node):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        episode_reached_flags = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=bool)
        episode_collision_flags = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=bool)
        episode_reached_counts = []
        episode_collision_counts = []
        is_predator_prey = self.env_name == "long_range_predator_prey"
        episode_capture_counts = []
        episode_prey_remaining = []
        episode_capture_success = []
        episode_predator_collision_counts = []
        episode_max_capture_groups = []
        pp_capture_counts = np.zeros(self.n_eval_rollout_threads, dtype=np.int64)
        pp_collision_counts = np.zeros(self.n_eval_rollout_threads, dtype=np.int64)
        pp_final_prey_remaining = np.full(self.n_eval_rollout_threads, -1, dtype=np.int64)
        pp_capture_success = np.zeros(self.n_eval_rollout_threads, dtype=bool)
        pp_max_capture_group = np.zeros(self.n_eval_rollout_threads, dtype=np.int64)
        per_episode_rows = []

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
        eval_rnn_states = torch.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.n_embd), dtype=torch.float32, device=self.device)
        eval_masks = torch.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device=self.device)

        all_frames = []
        batch_edge_index=None

        '''render_env = gym.make(self.all_args.scenario,render_mode="rgb_array")
        render_obs = render_env.reset()

        image = render_env.render()[0][0]
        all_frames.append(image)'''

        while True:
            if self.all_args.iterations > 0 and self.algorithm_name in GNN_ALGORITHMS:
                edge_index = self.eval_envs.get_edge_index_matrix()
                edge_index = torch.tensor(edge_index, dtype=torch.float32, device=self.device)
                batch_edge_index = self.get_batch_edge_index(edge_index)

                x = self.trainer.policy.transformer.obs_encoder(eval_obs, batch_edge_index)
                eval_obs = torch.cat([eval_obs,x],dim=-1).detach()
            elif self.algorithm_name == "sr_mappo":
                batch_edge_index = torch.as_tensor(
                    self.eval_envs.get_visibility_matrix(),
                    dtype=torch.float32,
                    device=self.device,
                )

            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(eval_share_obs,
                                        eval_obs,
                                        eval_rnn_states,
                                        eval_masks,
                                        None,
                                        deterministic=True,
                                        batched_edge_index=batch_edge_index)
            eval_actions = eval_actions.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
            eval_rnn_states = eval_rnn_states.reshape(self.n_eval_rollout_threads, self.num_agents, -1)

            # Obser reward and next obs
            eval_actions = faulty_action(eval_actions.cpu().detach().numpy(), faulty_node)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(100*eval_actions)
            if is_predator_prey:
                self._update_predator_prey_episode_stats(
                    eval_infos,
                    pp_capture_counts,
                    pp_collision_counts,
                    pp_final_prey_remaining,
                    pp_capture_success,
                    pp_max_capture_group,
                )
            else:
                self._update_episode_team_flags(eval_infos, episode_reached_flags, episode_collision_flags)


            if self.all_args.use_render:
                self.eval_envs.render(mode="human")
                # print('eval_actions = ', 100*eval_actions)

            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device=self.device)
            eval_share_obs = torch.tensor(eval_share_obs, dtype=torch.float32, device=self.device)
            eval_dones = torch.tensor(eval_dones, dtype=torch.float32, device=self.device)

            '''flat_actions = np.concatenate([eval_actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
            render_obs, render_share_obs, render_rewards, render_dones, render_infos, _ = render_env.step(flat_actions)'''

            # Capture frame
            '''image = render_env.render()[0][0]
            all_frames.append(image)'''

            eval_dones_env = torch.all(eval_dones, dim=1)
            eval_rnn_states[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device=self.device)
            eval_masks = torch.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device=self.device)
            eval_masks[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device=self.device)

            for eval_i in range(self.all_args.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    if is_predator_prey:
                        capture_count = int(pp_capture_counts[eval_i])
                        predator_collision_count = int(pp_collision_counts[eval_i])
                        prey_remaining = int(pp_final_prey_remaining[eval_i])
                        if prey_remaining < 0:
                            prey_remaining = max(int(getattr(self.all_args, "num_prey", 0)) - capture_count, 0)
                        success = bool(pp_capture_success[eval_i])
                        max_capture_group = int(pp_max_capture_group[eval_i])
                        num_prey = max(int(getattr(self.all_args, "num_prey", 1)), 1)

                        episode_capture_counts.append(capture_count)
                        episode_predator_collision_counts.append(predator_collision_count)
                        episode_prey_remaining.append(prey_remaining)
                        episode_capture_success.append(float(success))
                        episode_max_capture_groups.append(max_capture_group)
                        per_episode_rows.append(
                            {
                                "algorithm_name": self.algorithm_name,
                                "num_predators": int(self.num_agents),
                                "num_prey": int(num_prey),
                                "faulty_node": int(faulty_node),
                                "eval_step": int(total_num_steps),
                                "episode_index": int(eval_episode),
                                "captured_prey": capture_count,
                                "capture_rate": float(capture_count / num_prey),
                                "prey_remaining": prey_remaining,
                                "capture_success": int(success),
                                "predator_collision_pairs": predator_collision_count,
                                "max_capture_group": max_capture_group,
                                "episode_reward": float(one_episode_rewards[eval_i]),
                            }
                        )
                        pp_capture_counts[eval_i] = 0
                        pp_collision_counts[eval_i] = 0
                        pp_final_prey_remaining[eval_i] = -1
                        pp_capture_success[eval_i] = False
                        pp_max_capture_group[eval_i] = 0
                    else:
                        reached_count = int(np.sum(episode_reached_flags[eval_i]))
                        collision_count = int(np.sum(episode_collision_flags[eval_i]))
                        episode_reached_counts.append(reached_count)
                        episode_collision_counts.append(collision_count)
                        per_episode_rows.append(
                            {
                                "algorithm_name": self.algorithm_name,
                                "team_size": int(self.num_agents),
                                "faulty_node": int(faulty_node),
                                "eval_step": int(total_num_steps),
                                "episode_index": int(eval_episode),
                                "reached_goal_count": reached_count,
                                "collision_count": collision_count,
                                "reached_goal_rate": float(reached_count / max(self.num_agents, 1)),
                                "collision_rate": float(collision_count / max(self.num_agents, 1)),
                                "episode_reward": float(one_episode_rewards[eval_i]),
                            }
                        )
                    one_episode_rewards[eval_i] = 0
                    episode_reached_flags[eval_i] = False
                    episode_collision_flags[eval_i] = False

            if eval_episode >= self.all_args.eval_episodes:
                key_average = 'faulty_node_' + str(faulty_node) + '/eval_average_episode_rewards'
                key_max = 'faulty_node_' + str(faulty_node) + '/eval_max_episode_rewards'
                if is_predator_prey:
                    num_prey = max(int(getattr(self.all_args, "num_prey", 1)), 1)
                    key_captured = 'faulty_node_' + str(faulty_node) + '/eval_captured_prey'
                    key_capture_rate = 'faulty_node_' + str(faulty_node) + '/eval_capture_rate'
                    key_prey_remaining = 'faulty_node_' + str(faulty_node) + '/eval_prey_remaining'
                    key_capture_success = 'faulty_node_' + str(faulty_node) + '/eval_capture_success_rate'
                    key_predator_collisions = 'faulty_node_' + str(faulty_node) + '/eval_predator_collision_pairs'
                    key_max_capture_group = 'faulty_node_' + str(faulty_node) + '/eval_max_capture_group'
                    eval_env_infos = {
                        key_average: eval_episode_rewards,
                        key_max: [np.max(eval_episode_rewards)],
                        key_captured: episode_capture_counts,
                        key_capture_rate: [c / num_prey for c in episode_capture_counts],
                        key_prey_remaining: episode_prey_remaining,
                        key_capture_success: episode_capture_success,
                        key_predator_collisions: episode_predator_collision_counts,
                        key_max_capture_group: episode_max_capture_groups,
                    }

                    self.log_env(eval_env_infos, total_num_steps)
                    self._append_eval_predator_prey_stats(per_episode_rows)
                    print("faulty_node {} eval_average_episode_rewards is {}."
                          .format(faulty_node, np.mean(eval_episode_rewards)))
                    print(
                        "faulty_node {} avg captured_prey {:.3f}/{}, capture_success {:.1f}%, "
                        "avg prey_remaining {:.3f}, avg predator_collision_pairs {:.3f}, avg max_capture_group {:.3f}."
                        .format(
                            faulty_node,
                            np.mean(episode_capture_counts) if len(episode_capture_counts) > 0 else 0.0,
                            num_prey,
                            100.0 * (np.mean(episode_capture_success) if len(episode_capture_success) > 0 else 0.0),
                            np.mean(episode_prey_remaining) if len(episode_prey_remaining) > 0 else 0.0,
                            np.mean(episode_predator_collision_counts) if len(episode_predator_collision_counts) > 0 else 0.0,
                            np.mean(episode_max_capture_groups) if len(episode_max_capture_groups) > 0 else 0.0,
                        )
                    )

                    if self.use_wandb:
                        wandb.log(
                            {
                                "eval_average_episode_rewards": np.mean(eval_episode_rewards),
                                "eval_captured_prey": np.mean(episode_capture_counts) if len(episode_capture_counts) > 0 else 0.0,
                                "eval_capture_rate": np.mean(episode_capture_counts) / num_prey if len(episode_capture_counts) > 0 else 0.0,
                                "eval_prey_remaining": np.mean(episode_prey_remaining) if len(episode_prey_remaining) > 0 else 0.0,
                                "eval_capture_success_rate": np.mean(episode_capture_success) if len(episode_capture_success) > 0 else 0.0,
                                "eval_predator_collision_pairs": np.mean(episode_predator_collision_counts) if len(episode_predator_collision_counts) > 0 else 0.0,
                                "eval_max_capture_group": np.mean(episode_max_capture_groups) if len(episode_max_capture_groups) > 0 else 0.0,
                                "eval_num_predators": int(self.num_agents),
                                "eval_num_prey": int(num_prey),
                                "eval_faulty_node": int(faulty_node),
                            },
                            step=total_num_steps,
                        )
                        if len(per_episode_rows) > 0:
                            table_columns = list(per_episode_rows[0].keys())
                            table = wandb.Table(columns=table_columns)
                            for row in per_episode_rows:
                                table.add_data(*[row[c] for c in table_columns])
                            wandb.log(
                                {f"faulty_node_{faulty_node}/eval_predator_prey_stats_table": table},
                                step=total_num_steps,
                            )
                else:
                    key_reached_count = 'faulty_node_' + str(faulty_node) + '/eval_reached_goal_count'
                    key_collision_count = 'faulty_node_' + str(faulty_node) + '/eval_collision_count'
                    key_reached_rate = 'faulty_node_' + str(faulty_node) + '/eval_reached_goal_rate'
                    key_collision_rate = 'faulty_node_' + str(faulty_node) + '/eval_collision_rate'
                    eval_env_infos = {key_average: eval_episode_rewards,
                                      key_max: [np.max(eval_episode_rewards)],
                                      key_reached_count: episode_reached_counts,
                                      key_collision_count: episode_collision_counts,
                                      key_reached_rate: [c / max(self.num_agents, 1) for c in episode_reached_counts],
                                      key_collision_rate: [c / max(self.num_agents, 1) for c in episode_collision_counts]}

                    self.log_env(eval_env_infos, total_num_steps)
                    self._append_eval_team_stats(per_episode_rows)
                    print("faulty_node {} eval_average_episode_rewards is {}."
                          .format(faulty_node, np.mean(eval_episode_rewards)))
                    print(
                        "faulty_node {} avg reached_goals {:.3f}/{}, avg collisions {:.3f}/{}."
                        .format(
                            faulty_node,
                            np.mean(episode_reached_counts) if len(episode_reached_counts) > 0 else 0.0,
                            self.num_agents,
                            np.mean(episode_collision_counts) if len(episode_collision_counts) > 0 else 0.0,
                            self.num_agents,
                        )
                    )

                    if self.use_wandb:
                            wandb.log(
                                {
                                    "eval_average_episode_rewards": np.mean(eval_episode_rewards),
                                    "eval_reached_goal_count": np.mean(episode_reached_counts) if len(episode_reached_counts) > 0 else 0.0,
                                    "eval_collision_count": np.mean(episode_collision_counts) if len(episode_collision_counts) > 0 else 0.0,
                                    "eval_reached_goal_rate": np.mean(episode_reached_counts) / max(self.num_agents, 1) if len(episode_reached_counts) > 0 else 0.0,
                                    "eval_collision_rate": np.mean(episode_collision_counts) / max(self.num_agents, 1) if len(episode_collision_counts) > 0 else 0.0,
                                    "eval_team_size": int(self.num_agents),
                                    "eval_faulty_node": int(faulty_node),
                                },
                                step=total_num_steps,
                            )
                            self._update_aggregate_eval_stats(
                                faulty_node,
                                episode_reached_counts,
                                episode_collision_counts,
                            )
                            if len(per_episode_rows) > 0:
                                table_columns = list(per_episode_rows[0].keys())
                                table = wandb.Table(columns=table_columns)
                                for row in per_episode_rows:
                                    table.add_data(*[row[c] for c in table_columns])
                                wandb.log(
                                    {f"faulty_node_{faulty_node}/eval_team_stats_table": table},
                                    step=total_num_steps,
                                )
                            agg_reached_counts = self._agg_reached_counts_by_fault.get(faulty_node, episode_reached_counts)
                            agg_collision_counts = self._agg_collision_counts_by_fault.get(faulty_node, episode_collision_counts)
                            self._log_ci_bar_chart_to_wandb(
                                agg_reached_counts,
                                agg_collision_counts,
                                faulty_node,
                                total_num_steps,
                            )
                            self._log_ci_summary_table_to_wandb(
                                agg_reached_counts,
                                agg_collision_counts,
                                faulty_node,
                                total_num_steps,
                            )

                if self.reward_list is None:
                    self.reward_list = np.array(np.mean(eval_episode_rewards))
                else:
                    self.reward_list = np.hstack((self.reward_list, np.array(np.mean(eval_episode_rewards))))

                np.save(self.plt_name+'_eval.npy', self.reward_list)

                break

        # Save video
        '''if len(frames) > 0:
            #video_dir = os.path.join(self.run_dir, 'videos')
            video_dir = os.path.join('/home/shahils/Desktop/marl_ws/Multi-Agent-Transformer_old/mat/scripts/videos')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f'step_{total_num_steps}.mp4')
            imageio.mimsave(video_path, all_frames, fps=30)'''
