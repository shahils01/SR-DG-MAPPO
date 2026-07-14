import wandb
import os
import numpy as np
import torch
from mat.utils.shared_buffer import SharedReplayBuffer
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from mat.utils.logging import SummaryWriter
from mat.utils.util import get_shape_from_obs_space

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        # self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.n_embd = self.all_args.n_embd

        act_space = self.envs.action_space[0]

        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(self.envs.observation_space[0])[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
        else:
            self.act_dim = act_space.shape[0]

        self.num_quants = self.all_args.n_quants

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name.startswith('mat'):
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        else:
            share_observation_space = self.envs.share_observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                         self.all_args.env_name,
                                         device=self.device)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            graph_context = (
                self.buffer.adjcency_matrix[-1]
                if self.algorithm_name == "sr_mappo"
                else None
            )
            next_values = self.trainer.policy.get_values(self.buffer.share_obs[-1],
                                                         self.buffer.obs[-1],
                                                         self.buffer.rnn_states_critic[-1],
                                                         self.buffer.masks[-1],
                                                         graph_context=graph_context)
        else:
            graph_context = (
                self.buffer.adjcency_matrix[-1]
                if self.algorithm_name == "sr_mappo"
                else None
            )
            next_values = self.trainer.policy.get_values(self.buffer.share_obs[-1],
                                                         self.buffer.obs[-1],
                                                         self.buffer.rnn_states_critic[-1],
                                                         self.buffer.masks[-1],
                                                         self.buffer.available_actions[-1],
                                                         graph_context=graph_context)

        # action_log, next_values, _ = self.trainer.policy.transformer(state, obs, action, available_actions, action_hats)

        next_values = next_values.reshape(self.n_rollout_threads, self.num_agents, -1)

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self, episode):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, episode, self.obs_dim)
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        allow_partial_restore = getattr(self.all_args, "allow_partial_restore", False)
        self.policy.restore(model_dir, allow_partial=allow_partial_restore)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
