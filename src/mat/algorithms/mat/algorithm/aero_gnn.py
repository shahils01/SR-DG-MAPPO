import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class AERO_GNN_Model(MessagePassing):
    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__(node_dim=0, aggr='add')

        self.args = args
        self.num_nodes = num_agents
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = self.args.num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = self.args.iterations

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ModuleList()  # [n_agent] of [heads, hid_channels]
        self.hop_atts = nn.ModuleList()  # [K+1][n_agent] of [heads, hid_channels or 2*hid_channels]
        self.hop_biases = nn.ModuleList()  # same as above
        self.decay_weights = []

        # Per-agent MLP encoders
        self.agent_encoders = nn.ModuleList([
            nn.Sequential(
                Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'),
                nn.ELU(),
                nn.Dropout(self.args.dropout),
                *[nn.Sequential(
                    Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'),
                    nn.ELU(),
                    nn.Dropout(self.args.dropout)
                ) for _ in range(self.args.num_layers - 1)]
            )
            for _ in range(self.num_nodes)
        ])

        # Per-agent classifier heads
        self.node_classifier_heads = nn.ModuleList([
            Linear(self.heads * self.hid_channels, self.out_channels, bias=True, weight_initializer='glorot')
            for _ in range(self.num_nodes)
        ])

        # Init encoder & head weights
        for agent_enc in self.agent_encoders:
            for layer in agent_enc:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        for head in self.node_classifier_heads:
            torch.nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                torch.nn.init.zeros_(head.bias)

        # Hop attention & bias per hop and per agent
        for k in range(self.K + 1):
            edge_att_layer = nn.ParameterList()
            hop_att_layer = nn.ParameterList()
            hop_bias_layer = nn.ParameterList()

            for i in range(self.num_nodes):
                edge_att = nn.Parameter(torch.Tensor(self.heads, 2 * self.hid_channels))

                dim = self.hid_channels if k == 0 else 2 * self.hid_channels
                hop_att = nn.Parameter(torch.Tensor(self.heads, dim))
                hop_bias = nn.Parameter(torch.Tensor(self.heads))

                nn.init.xavier_uniform_(edge_att)
                nn.init.xavier_uniform_(hop_att)
                nn.init.zeros_(hop_bias)

                edge_att_layer.append(edge_att)
                hop_att_layer.append(hop_att)
                hop_bias_layer.append(hop_bias)

            if k < self.K + 1:
                self.atts.append(edge_att_layer)
            self.hop_atts.append(hop_att_layer)
            self.hop_biases.append(hop_bias_layer)

            # Precompute decay weight for hop k
            decay = np.log((self.args.lambd_gnn / (k + 1)) + (1 + 1e-6))
            self.decay_weights.append(decay)

    def reset_parameters(self):
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att)
        for att in self.hop_atts: glorot(att)
        for bias in self.hop_biases: ones(bias)

    def hid_feat_init(self, x):
        # x shape: (batch_size, num_agents, in_channels)
        x = self.dropout(x)
        h_list = []
        for agent_id in range(self.num_nodes):
            x_i = x[:, agent_id, :]  # shape: (batch_size, in_channels)
            x_i = self.agent_encoders[agent_id](x_i)  # (batch_size, hid_channels_)
            h_list.append(x_i.unsqueeze(1))
        x = torch.cat(h_list, dim=1)  # (batch_size, num_agents, hid_channels_)

        # Reshape to (batch_size, num_agents, heads, hid_channels)
        x = x.view(-1, self.num_nodes, self.heads, self.hid_channels)
        return x


    def aero_propagate(self, h, edge_index):
        # h shape: (batch_size, num_agents, heads, hid_channels)
        batch_size = h.size(0)
        self.k = 0

        edge_index_batch = edge_index

        # Initial hop attention
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[0]

        for k in range(self.K):
            self.k = k + 1

            # Flatten for batch processing
            h_flat = h.reshape(-1, self.heads, self.hid_channels)  # (batch_size * num_agents, heads, hid_channels)
            z_scale_flat = z_scale.reshape(-1, self.heads, self.hid_channels)

            # Prepare edge features
            row, col = edge_index_batch
            z_scale_i = z_scale_flat[row]
            z_scale_j = z_scale_flat[col]

            # Compute attention coefficients
            a_ij = self.edge_att_pred(z_scale_i, z_scale_j, edge_index_batch)

            # Prepare messages
            x_j = h_flat[col]
            messages = a_ij.unsqueeze(-1) * x_j

            # Aggregate messages
            out = torch.zeros_like(h_flat)
            out = scatter_add(messages, row, dim=0, out=out)

            # Reshape back
            h = out.view(batch_size, self.num_nodes, self.heads, self.hid_channels)

            # Update z and z_scale
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]

        return z

    def node_classifier(self, z):
        # z shape: (batch_size, num_agents, heads, hid_channels)
        batch_size, num_agents, _, _ = z.size()
        logits = []

        for agent_id in range(self.num_nodes):
            z_i = z[:, agent_id, :, :]  # (batch_size, heads, hid_channels)
            z_i = z_i.reshape(batch_size, -1)  # flatten heads
            z_i = self.elu(z_i)
            if self.args.add_dropout:
                z_i = self.dropout(z_i)
            z_i = self.node_classifier_heads[agent_id](z_i)

            if torch.isnan(z_i).any():
                print("Warning: NaNs in node_classifier output")

            logits.append(z_i.unsqueeze(1))

        z_out = torch.cat(logits, dim=1)  # (batch_size, num_agents, out_channels)
        return z_out # .clip(-1e6, 1e6)

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_agents, in_channels)
        # edge_index shape: (2, num_edges)
        h0 = self.hid_feat_init(x)  # (batch_size, num_agents, heads, hid_channels)
        z_k_max = self.aero_propagate(h0, edge_index)  # (batch_size, num_agents, heads, hid_channels)
        z_star = self.node_classifier(z_k_max)  # (batch_size, num_agents, out_channels)

        return z_star # .clip(-1e6, 1e6)

    def hop_att_pred(self, h, z_scale):
        # h shape: (batch_size, num_agents, heads, hid_channels) or similar
        if z_scale is None:
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        x.masked_fill_(torch.isnan(x), 0.0)

        # Compute attention for all batches and agents simultaneously
        g = self.elu(x)

        hop_att = torch.stack(list(self.hop_atts[self.k]), dim=0).unsqueeze(0)
        hop_bias = torch.stack(list(self.hop_biases[self.k]), dim=0).unsqueeze(0).unsqueeze(-1)

        g = (hop_att * g).sum(dim=-1, keepdim=True) + hop_bias

        if torch.isnan(g).any():
            print("Warning: NaNs in hop_att output while handling layer k = {self.k}")

        return g # torch.tanh(g)

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index_batch):
        # z_scale_i, z_scale_j shape: (batch_size * num_edges, heads, hid_channels)
        # edge_index_batch shape: (2, batch_size * num_edges)
        batch_size = z_scale_i.size(0)

        # edge attention (alpha_check_ij)
        # a_ij = z_scale_i + z_scale_j
        a_ij = torch.cat((z_scale_i, z_scale_j), dim=-1)
        a_ij = self.elu(a_ij)

        src_nodes = edge_index_batch[0]
        agent_indices = src_nodes % self.num_nodes
        # att_vec = self.atts[self.k-1][agent_indices]

        att_vec = torch.stack([self.atts[self.k-1][i] for i in agent_indices.tolist()], dim=0)

        a_ij = (att_vec * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij).clamp(max=1e6) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index_batch[0], edge_index_batch[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=batch_size * self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle zero degrees
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]

        if torch.isnan(a_ij).any():
            raise ValueError("NaNs detected in edge attention normalization while handling layer k+1 = {self.k}")

        return a_ij


class GNN_Model(MessagePassing):
    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__(node_dim=0, aggr='add')

        self.args = args
        self.num_nodes = num_agents
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = self.args.num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = self.args.iterations

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.decay_weights = []

        # Per-agent MLP encoders
        self.agent_encoders = nn.ModuleList([
            nn.Sequential(
                Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'),
                nn.ELU(),
                nn.Dropout(self.args.dropout),
                *[nn.Sequential(
                    Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'),
                    nn.ELU(),
                    nn.Dropout(self.args.dropout)
                ) for _ in range(self.args.num_layers - 1)]
            )
            for _ in range(self.num_nodes)
        ])

        # Per-agent classifier heads
        self.node_classifier_heads = nn.ModuleList([
            Linear(self.heads * self.hid_channels, self.out_channels, bias=True, weight_initializer='glorot')
            for _ in range(self.num_nodes)
        ])

        # Init encoder & head weights
        for agent_enc in self.agent_encoders:
            for layer in agent_enc:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        for head in self.node_classifier_heads:
            torch.nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                torch.nn.init.zeros_(head.bias)

        if self.args.algorithm_name == 'mappo_dgnn' or self.args.algorithm_name == 'mappo_dgnn_dsgd':
            self.atts = nn.ModuleList()  # [n_agent] of [heads, hid_channels]
            # Hop attention & bias per hop and per agent
            for k in range(self.K):
                edge_att_layer = nn.ParameterList()

                for i in range(self.num_nodes):
                    edge_att = nn.Parameter(torch.Tensor(self.heads, 2 * self.hid_channels))
                    nn.init.xavier_uniform_(edge_att)
                    edge_att_layer.append(edge_att)

                self.atts.append(edge_att_layer)

                # Precompute decay weight for hop k
                decay = np.log((self.args.lambd_gnn / (k + 1)) + (1 + 1e-6))
                self.decay_weights.append(decay)
        else:
            self.atts = nn.ModuleList()  # [1] of [heads, hid_channels] because shared Trunk
            edge_att_layer = nn.ParameterList()
            # Hop attention & bias per hop and per agent
            for k in range(self.K + 1):
                edge_att = nn.Parameter(torch.Tensor(self.heads, 2 * self.hid_channels))
                nn.init.xavier_uniform_(edge_att)
                edge_att_layer.append(edge_att)

                self.atts.append(edge_att_layer)

                # Precompute decay weight for hop k
                decay = np.log((self.args.lambd_gnn / (k + 1)) + (1 + 1e-6))
                self.decay_weights.append(decay)

    def reset_parameters(self):
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att)

    def hid_feat_init(self, x):
        # x shape: (batch_size, num_agents, in_channels)
        x = self.dropout(x)
        h_list = []
        for agent_id in range(self.num_nodes):
            x_i = x[:, agent_id, :]  # shape: (batch_size, in_channels)
            x_i = self.agent_encoders[agent_id](x_i)  # (batch_size, hid_channels_)
            h_list.append(x_i.unsqueeze(1))
        x = torch.cat(h_list, dim=1)  # (batch_size, num_agents, hid_channels_)

        # Reshape to (batch_size, num_agents, heads, hid_channels)
        x = x.view(-1, self.num_nodes, self.heads, self.hid_channels)
        return x


    def aero_propagate(self, h, edge_index):
        # h shape: (batch_size, num_agents, heads, hid_channels)
        batch_size = h.size(0)
        self.k = 0

        # Create batch-aware edge index
        edge_index_batch = edge_index

        # Initial hop attention
        z = h
        z_scale = z * self.decay_weights[0]

        for k in range(self.K):
            self.k = k + 1

            # Flatten for batch processing
            h_flat = h.reshape(-1, self.heads, self.hid_channels)  # (batch_size * num_agents, heads, hid_channels)
            z_scale_flat = z_scale.reshape(-1, self.heads, self.hid_channels)

            # Prepare edge features
            row, col = edge_index_batch
            z_scale_i = z_scale_flat[row]
            z_scale_j = z_scale_flat[col]

            # Compute attention coefficients
            a_ij = self.edge_att_pred(z_scale_i, z_scale_j, edge_index_batch)

            # Prepare messages
            x_j = h_flat[col]
            messages = a_ij.unsqueeze(-1) * x_j

            # Aggregate messages
            out = torch.zeros_like(h_flat)
            out = scatter_add(messages, row, dim=0, out=out)

            # Reshape back
            h = out.view(batch_size, self.num_nodes, self.heads, self.hid_channels)

            # Update z and z_scale
            z += h
            z_scale = z #* self.decay_weights[self.k]

        return z #.clip(-1e6, 1e6)

    def node_classifier(self, z):
        # z shape: (batch_size, num_agents, heads, hid_channels)
        batch_size, num_agents, _, _ = z.size()
        logits = []

        for agent_id in range(self.num_nodes):
            z_i = z[:, agent_id, :, :]  # (batch_size, heads, hid_channels)
            z_i = z_i.reshape(batch_size, -1)  # flatten heads
            z_i = self.elu(z_i)
            if self.args.add_dropout:
                z_i = self.dropout(z_i)
            z_i = self.node_classifier_heads[agent_id](z_i)

            if torch.isnan(z_i).any():
                print("Warning: NaNs in node_classifier output")

            logits.append(z_i.unsqueeze(1))

        z_out = torch.cat(logits, dim=1)  # (batch_size, num_agents, out_channels)
        return z_out # .clip(-1e6, 1e6)

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_agents, in_channels)
        # edge_index shape: (2, num_edges)
        h0 = self.hid_feat_init(x)  # (batch_size, num_agents, heads, hid_channels)
        z_k_max = self.aero_propagate(h0, edge_index)  # (batch_size, num_agents, heads, hid_channels)
        z_star = self.node_classifier(z_k_max)  # (batch_size, num_agents, out_channels)

        return z_star # .clip(-1e6, 1e6)

    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index_batch):
        # z_scale_i, z_scale_j shape: (batch_size * num_edges, heads, hid_channels)
        # edge_index_batch shape: (2, batch_size * num_edges)
        batch_size = z_scale_i.size(0)

        # edge attention (alpha_check_ij)
        # a_ij = z_scale_i + z_scale_j
        a_ij = torch.cat((z_scale_i, z_scale_j), dim=-1)
        a_ij = self.elu(a_ij)

        src_nodes = edge_index_batch[0]
        agent_indices = src_nodes % self.num_nodes
        # att_vec = self.atts[self.k-1][agent_indices]

        if self.args.algorithm_name == 'mappo_dgnn' or self.args.algorithm_name == 'mappo_dgnn_dsgd':
            att_vec = torch.stack([self.atts[self.k-1][i] for i in agent_indices.tolist()], dim=0)
        else:
            # att_vec = torch.stack([self.atts[self.k-1][0] for i in agent_indices.tolist()], dim=0)
            x = self.atts[self.k-1][0]   # shape: [d1, d2, ...]
            N = agent_indices.size(0)
            att_vec = x.unsqueeze(0).expand(N, *x.shape)

        a_ij = (att_vec * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij).clamp(max=1e6) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index_batch[0], edge_index_batch[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=batch_size * self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle zero degrees
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]

        if torch.isnan(a_ij).any():
            raise ValueError("NaNs detected in edge attention normalization while handling layer k+1 = {self.k}")

        return a_ij


class MeanGNN_Model(MessagePassing):
    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__(node_dim=0, aggr='add')

        self.args = args
        self.num_nodes = num_agents
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = self.args.num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = self.args.iterations

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self.dropout = nn.Dropout(self.args.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        # kept for compatibility but unused
        self.dense_lins = nn.ModuleList()

        # Per-agent MLP encoders
        self.agent_encoders = nn.ModuleList([
            nn.Sequential(
                Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'),
                nn.ELU(),
                nn.Dropout(self.args.dropout),
                *[nn.Sequential(
                    Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'),
                    nn.ELU(),
                    nn.Dropout(self.args.dropout)
                ) for _ in range(self.args.num_layers - 1)]
            )
            for _ in range(self.num_nodes)
        ])

        # Per-agent classifier heads
        self.node_classifier_heads = nn.ModuleList([
            Linear(self.heads * self.hid_channels, self.out_channels, bias=True, weight_initializer='glorot')
            for _ in range(self.num_nodes)
        ])

        # Init encoder & head weights
        for agent_enc in self.agent_encoders:
            for layer in agent_enc:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        for head in self.node_classifier_heads:
            torch.nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                torch.nn.init.zeros_(head.bias)

        # No attention parameters needed here
        self.atts = None

        # Optional: keep decay weights if you still want hop-dependent scaling
        self.decay_weights = [
            np.log((self.args.lambd_gnn / (k + 1)) + (1 + 1e-6))
            for k in range(self.K + 1)
        ]

    def reset_parameters(self):
        # You already initialize Linear layers in setup_layers
        # Nothing special to reset beyond that.
        pass

    def hid_feat_init(self, x):
        # x shape: (batch_size, num_agents, in_channels)
        x = self.dropout(x)
        h_list = []
        for agent_id in range(self.num_nodes):
            x_i = x[:, agent_id, :]  # (batch_size, in_channels)
            x_i = self.agent_encoders[agent_id](x_i)  # (batch_size, hid_channels_)
            h_list.append(x_i.unsqueeze(1))
        x = torch.cat(h_list, dim=1)  # (batch_size, num_agents, hid_channels_)

        # Reshape to (batch_size, num_agents, heads, hid_channels)
        x = x.view(-1, self.num_nodes, self.heads, self.hid_channels)
        return x

    def aero_propagate(self, h, edge_index):
        """
        Same interface as your original aero_propagate, but uses
        simple neighbor mean aggregation instead of attention.

        h: (batch_size, num_agents, heads, hid_channels)
        edge_index: (2, num_edges) over the *flattened* nodes
                    (batch_size * num_agents)
        """
        batch_size = h.size(0)
        self.k = 0

        edge_index_batch = edge_index  # assumed already batch-aware

        # Initial 'z' (accumulated representation)
        z = h

        for k in range(self.K):
            self.k = k + 1

            # Flatten for batch processing:
            # (batch_size * num_agents, heads, hid_channels)
            h_flat = h.reshape(-1, self.heads, self.hid_channels)

            row, col = edge_index_batch  # each of shape (num_edges_total,)

            # Messages are just neighbor features (no attention)
            x_j = h_flat[col]  # (num_edges_total, heads, hid_channels)

            # Sum aggregation
            out = torch.zeros_like(h_flat)
            out = scatter_add(x_j, row, dim=0, out=out)

            # Compute degree for mean aggregation
            deg = scatter_add(
                torch.ones_like(row, dtype=h_flat.dtype),
                row,
                dim=0,
                dim_size=batch_size * self.num_nodes
            )  # (batch_size * num_agents,)

            # Avoid division by zero
            deg = deg.view(-1, self.heads, 1).clamp(min=1.0)

            # Mean of neighbors
            out = out / deg  # (batch_size * num_agents, heads, hid_channels)

            # Reshape back
            h = out.view(batch_size, self.num_nodes, self.heads, self.hid_channels)

            # Accumulate across hops (you can drop this if you don't want residual across hops)
            z += h

        return z

    def node_classifier(self, z):
        # z shape: (batch_size, num_agents, heads, hid_channels)
        batch_size, num_agents, _, _ = z.size()
        logits = []

        for agent_id in range(self.num_nodes):
            z_i = z[:, agent_id, :, :]  # (batch_size, heads, hid_channels)
            z_i = z_i.reshape(batch_size, -1)  # flatten heads
            z_i = self.elu(z_i)
            if self.args.add_dropout:
                z_i = self.dropout(z_i)
            z_i = self.node_classifier_heads[agent_id](z_i)

            if torch.isnan(z_i).any():
                print("Warning: NaNs in node_classifier output")

            logits.append(z_i.unsqueeze(1))

        z_out = torch.cat(logits, dim=1)  # (batch_size, num_agents, out_channels)
        return z_out

    def forward(self, x, edge_index):
        # x shape: (batch_size, num_agents, in_channels)
        # edge_index shape: (2, num_edges) over flattened nodes
        h0 = self.hid_feat_init(x)          # (batch_size, num_agents, heads, hid_channels)
        z_k = self.aero_propagate(h0, edge_index)  # (batch_size, num_agents, heads, hid_channels)
        z_star = self.node_classifier(z_k)  # (batch_size, num_agents, out_channels)
        return z_star


class GATv2MultiHop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        self.out = GATv2Conv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        num_agents = x.shape[1]
        num_edges = edge_index.size(-1)

        x = x.view(num_agents,-1)

        # Create batch-aware edge index
        offset = torch.arange(0, batch_size * num_agents, num_agents,
                            device=x.device).repeat_interleave(num_edges)

        # Expand edge_index for all batches
        edge_index_batch = edge_index.repeat(1, batch_size) + offset.unsqueeze(0)

        for conv in self.convs:
            x = conv(x, edge_index_batch).relu()
        return self.out(x, edge_index_batch)