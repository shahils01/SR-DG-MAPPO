import torch
import numpy as np
import torch.nn.functional as F
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(
        self,
        args,
        num_agents,
        obs_space,
        cent_obs_space,
        act_space,
        env_name,
        use_value_entropy=True,
        device=torch.device("cpu"),
    ):
        self.device = torch.device(device)
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.n_embd = args.n_embd
        self.recurrent_N = args.recurrent_N
        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=self.device)
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name
        self.num_quants = args.n_quants

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=torch.float32, device=self.device)
        self.obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=torch.float32, device=self.device)
        self.edge_index = torch.zeros((self.episode_length + 1, self.n_rollout_threads, 2, num_agents*num_agents), dtype=torch.float32, device=self.device)
        self.adjcency_matrix = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, num_agents), dtype=torch.float32, device=self.device)

        self.rnn_states = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.n_embd),
            dtype=torch.float32, device=self.device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.num_quants), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros_like(self.value_preds)
        self.advantages = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=torch.float32, device=self.device)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = torch.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=torch.float32, device=self.device)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=torch.float32, device=self.device)
        self.action_log_probs = torch.zeros(
            (self.episode_length+1, self.n_rollout_threads, num_agents, act_shape), dtype=torch.float32, device=self.device)

        self.action_hats = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, args.n_embd), dtype=torch.float32, device=self.device)

        self.rewards = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=torch.float32, device=self.device)

        self.masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=torch.float32, device=self.device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)

        '''self.flat_params = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 68100), dtype=torch.float32, device=self.device)
        self.reconst_params = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 68100), dtype=torch.float32, device=self.device)
        self.params_mu = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, args.n_embd), dtype=torch.float32, device=self.device)
        self.params_logvar = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, args.n_embd), dtype=torch.float32, device=self.device)'''

        self.step = 0
        self.use_value_entropy = use_value_entropy

        if self.num_quants > 1:
            self.quantile_spacing = 1.0 / (self.num_quants - 1)

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None,
                 available_actions=None, edge_index=None, adjcency_matrix=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.clone()
        self.obs[self.step + 1] = obs.clone()
        self.rnn_states[self.step + 1] = rnn_states_actor.clone()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.clone()
        self.actions[self.step] = actions.clone()
        self.action_log_probs[self.step] = action_log_probs.clone()
        self.value_preds[self.step] = value_preds.clone()
        self.rewards[self.step] = rewards.clone()
        self.masks[self.step + 1] = masks.clone()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.clone()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.clone()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.clone()
        if edge_index is not None:
            self.edge_index[self.step + 1] = edge_index.clone()
        if adjcency_matrix is not None:
            self.adjcency_matrix[self.step + 1] = adjcency_matrix.clone()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None,
                     available_actions=None, action_hats=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.clone()
        self.obs[self.step] = obs.clone()
        self.rnn_states[self.step + 1] = rnn_states.clone()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.clone()
        self.actions[self.step] = actions.clone()
        self.action_log_probs[self.step] = action_log_probs.clone()
        self.value_preds[self.step] = value_preds.clone()
        self.rewards[self.step] = rewards.clone()
        self.masks[self.step + 1] = masks.clone()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.clone()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.clone()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.clone()
        if action_hats is not None:
            self.action_hats[self.step] = action_hats.clone()
        # if edge_index is not None:
        #     self.edge_index[self.step] = edge_index.clone()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].clone()
        self.obs[0] = self.obs[-1].clone()
        self.rnn_states[0] = self.rnn_states[-1].clone()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].clone()
        self.masks[0] = self.masks[-1].clone()
        self.bad_masks[0] = self.bad_masks[-1].clone()
        self.active_masks[0] = self.active_masks[-1].clone()
        self.edge_index[0] = self.edge_index[-1].clone()
        self.adjcency_matrix[0] = self.adjcency_matrix[-1].clone()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].clone()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].clone()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].coclonepy()
        self.masks[0] = self.masks[-1].clone()
        self.bad_masks[0] = self.bad_masks[-1].clone()

    def compute_entropy_from_inverse_cdf_numpy(self, inverse_cdfs):
        """
        Compute the entropy of random variables given their inverse CDFs (NumPy version).

        Parameters:
        - inverse_cdfs: np.ndarray of shape (40, 6, 64)
        The inverse CDFs for 6 random variables in 40 batches, each with 64 quantiles.

        Returns:
        - entropies: np.ndarray of shape (40, 6)
          The entropy for each random variable in each batch.
        """
        # Shape parameters
        batch_size, num_variables, num_quantiles = inverse_cdfs.shape

        # Quantile spacing (assume equally spaced quantiles)
        quantile_spacing = 1.0 / (num_quantiles - 1)

        # Compute differences between successive inverse CDF values
        differences = inverse_cdfs[:, :, 1:] - inverse_cdfs[:, :, :-1]  # Shape: (40, 6, 63)

        # Compute PDFs (add a small value to avoid division by zero)
        pdfs = quantile_spacing / (differences + 1e-8)  # Shape: (40, 6, 63)
        pdfs = np.clip(pdfs, 1e-8, None)  # Avoid log(0) by clamping

        # Compute entropy: -\int p(x) log p(x) dx (approximated as a sum)
        entropy = -np.sum(pdfs * np.log(pdfs) * quantile_spacing, axis=2)  # Shape: (40, 6)

        return np.expand_dims(entropy, axis=-1)

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = next_value
        gae = 0

        # - 0.95**(step+1) * torch.exp(self.action_log_probs[step])*self.rewards[step]

        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.wasserstein_like_distance(self.gamma * value_normalizer.denormalize(
                        self.value_preds[step + 1]) * self.masks[step + 1], value_normalizer.denormalize(self.value_preds[step]))

                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae #+ 0.1 * self.gae_lambda * np.clip(entropy_deltas, -delta, delta)

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                else:
                    delta = self.rewards[step] + self.wasserstein_like_distance(self.gamma * self.value_preds[step + 1] * \
                            self.masks[step + 1], self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = (gae - gae.mean()) / (gae.std() + 1e-8) #gae
                self.returns[step] = gae + self.value_preds[step]

    def wasserstein_like_distance(self, icdf1, icdf2):
        """
        Compute the Wasserstein distance between each pair of ICDF functions.

        Parameters:
        icdf1 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the first set of ICDFs.
        icdf2 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the second set of ICDFs.

        Returns:
        torch.Tensor: Tensor of shape [2048, 1] representing the Wasserstein distance for each pair of ICDFs.
        """
        # Compute the Wasserstein distance
        # Wasserstein distance between two distributions is the area between their CDFs
        # For ICDFs, this can be approximated by the average absolute difference between the ICDF values
        # distances = torch.sum((1/64)*(icdf1 - icdf2), dim=1, keepdim=True)\
        if self.use_value_entropy:
            del_icdf1 = (icdf1[:,:,1:] - icdf1[:,:,:-1])/self.quantile_spacing
            del_icdf2 = (icdf2[:,:,1:] - icdf2[:,:,:-1])/self.quantile_spacing

            icdf1_mids = (icdf1[:,:,1:] + icdf1[:,:,:-1])/2
            icdf2_mids = (icdf2[:,:,1:] + icdf2[:,:,:-1])/2

            distances = torch.mean((icdf1_mids - icdf2_mids) + 0.1*(torch.log(del_icdf1+1e-6)-torch.log(del_icdf2+1e-6)), dim=-1, keepdims=True)

        else:
            distances = torch.mean((icdf1 - icdf2), dim=-1)

        return distances

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        else:
            mini_batch_size = min(int(mini_batch_size), batch_size)
            num_mini_batch = max(1, batch_size // mini_batch_size)

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        edge_index = self.edge_index[:-1].reshape(-1, *self.edge_index.shape[2:])
        adjcency_matrix = self.adjcency_matrix[:-1].reshape(-1, *self.adjcency_matrix.shape[2:])
        adjcency_matrix = adjcency_matrix[rows, cols]
        next_obs = self.obs[1:].reshape(-1, *self.obs.shape[2:])
        next_obs = next_obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]

        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            edge_index_batch = edge_index[indices].reshape(-1, *edge_index.shape[2:])
            adjcency_matrix_batch = adjcency_matrix[indices].reshape(-1, *adjcency_matrix.shape[2:])
            next_obs_batch = next_obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])

            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None

            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, adjcency_matrix_batch, next_obs_batch, edge_index_batch
