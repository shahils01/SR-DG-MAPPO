import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act, discrete_decentralized_act, continuous_decentralized_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from mat.algorithms.utils.variationalPolicyEncoder import PolicyVAE
# from mat.algorithms.mat.algorithm.aero_gnn import AERO_GNN_Model as gnn
try:
    from mat.algorithms.mat.algorithm.aero_gnn import GNN_Model as gnn
except ModuleNotFoundError:
    from mat.algorithms.mat.algorithm.dense_gnn import GNN_Model as gnn
# from mat.algorithms.mat.algorithm.aero_gnn import MeanGNN_Model as gnn
# from mat.algorithms.mat.algorithm.aero_gnn import GATv2MultiHop as gat

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Encoder(nn.Module):

    def __init__(self, args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.use_centralized_critic = args.use_centralized_critic

        if args.iterations == 0:
            n_embd = 0

        '''self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim+n_embd),
                                         init_(nn.Linear(obs_dim+n_embd, n_embd), activate=True), nn.GELU())'''

        self.head_ = nn.ModuleList()
        for n in range(n_agent):
            if self.use_centralized_critic:
                critic = nn.Sequential(nn.LayerNorm(state_dim),
                                init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))
            else:
                critic = nn.Sequential(nn.LayerNorm(obs_dim+n_embd),
                                init_(nn.Linear(obs_dim+n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        v_loc = []
        rep = []
        # obs = obs.detach()
        # obs = self.obs_encoder(obs)
        # obs = torch.cat((obs,action_hat), axis=-1)
        for n in range(self.n_agent):
            if self.use_centralized_critic:
                x = state[:, n, :]
            else:
                x = obs[:, n, :]

            x = x.unsqueeze(1)
            rep_n = x
            v_loc_n = self.head_[n](rep_n[:,0,:])
            v_loc.append(v_loc_n)
            rep.append(rep_n)
        v_loc = torch.stack(v_loc, dim=1)
        rep = torch.stack(rep, dim=1)

        v_loc, _ = torch.sort(v_loc, dim=-1)

        return v_loc, rep

    def agent_forward(self, state, obs, agent_id):
        # obs = torch.cat((obs,action_hat), axis=-1)
        x = obs
        x = x.unsqueeze(1)
        v_loc = self.head_[agent_id](x[:,0,:])

        return v_loc

    def average_critic_parameters(self):
        """
        Averages the parameters of all critic networks in head_ and
        distributes the averaged parameters back to each critic.
        """
        with torch.no_grad():
            # Initialize dictionaries to store summed parameters
            avg_params = {}
            param_count = 0

            # First, sum up all parameters across all critics
            for critic in self.head_:
                for name, param in critic.named_parameters():
                    if param.requires_grad:
                        if name not in avg_params:
                            avg_params[name] = param.data.clone()
                        else:
                            avg_params[name] += param.data
                param_count += 1

            # Compute average
            for name in avg_params:
                avg_params[name] = avg_params[name] / param_count

            # Distribute averaged parameters back to all critics
            for critic in self.head_:
                for name, param in critic.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(avg_params[name])


class Decoder(nn.Module):

    def __init__(self, args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        print('n_agent = ', n_agent)
        print('action_dim = ', action_dim)
        print('obs_dim = ', obs_dim)

        self.mlp_ = nn.ModuleList()

        if args.iterations == 0:
            n_embd = 0

        for n in range(n_agent):
            actor = nn.Sequential(nn.LayerNorm(obs_dim+n_embd),
                                init_(nn.Linear(obs_dim+n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs_rep, obs):
        if torch.isnan(obs).any():
            print("Warning: NaNs in obs input to decoder")
            obs = obs.masked_fill(torch.isnan(obs), 0.0)

        logit = []
        for n in range(self.n_agent):
            x = obs[:, n, :]
            x = x.unsqueeze(1)
            logit_n = self.mlp_[n](x[:, 0, :])
            logit.append(logit_n)

        logit = torch.stack(logit, dim=1)
        return logit

    def agent_forward(self, obs_rep, obs, agent_id):
        if torch.isnan(obs).any():
            print("Warning: NaNs in obs input to decoder")
            obs = obs.masked_fill(torch.isnan(obs), 0.0)

        x = obs
        x = x.unsqueeze(1)
        logit = self.mlp_[agent_id](x[:, 0, :])

        return logit


class MultiAgentGnnTransformer(nn.Module):

    def __init__(self, args, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, num_quants=50):
        super(MultiAgentGnnTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self._use_CNN_for_pi = args.use_CNN_for_pi
        self._use_VAE_for_pi = args.use_VAE_for_pi
        self.n_embd = n_embd

        # GNN
        self.obs_encoder = gnn(args, obs_dim, n_embd, n_embd, n_agent)
        # self.policy_encoder = gnn(args, action_dim+n_agent, args.hid_dim, n_embd, n_agent)

        # Actor-Critic Networks
        self.encoder = Encoder(args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device)
        self.decoder = Decoder(args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                                   self.action_type, dec_actor=dec_actor, share_actor=share_actor)

        self.eye = torch.eye(self.n_agent, device=device).unsqueeze(0)
        self.eye = self.eye / torch.norm(self.eye, p='fro')  # Normalize entire matrix

        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None, obs_rep=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        v_loc, obs_rep = self.encoder(state, obs)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, batched_edge_index=None, obs_rep=None):
        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        # print('v_loc shape = ', v_loc.shape)

        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_decentralized_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_decentralized_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim,
                                                                             self.tpdv, deterministic)

        # action_logits = self.decoder(None,None,obs)
        # action_logits = torch.cat((F.gelu(action_logits), self.eye.repeat(action_logits.shape[0],1,1)), axis=-1)

        # output_action_hat = self.policy_encoder(action_logits, batched_edge_index)

        v_loc, obs_rep = self.encoder(state, obs)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)

        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot
