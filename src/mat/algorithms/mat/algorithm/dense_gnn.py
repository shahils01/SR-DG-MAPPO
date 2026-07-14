"""Dependency-free fallback for DG-MAPPO's multi-hop attention encoder."""

from __future__ import annotations

import torch
from torch import nn


class DenseGNNModel(nn.Module):
    """Per-agent attention message passing with the original encoder interface."""

    def __init__(self, args, in_channels, hid_channels, out_channels, num_agents):
        super().__init__()
        self.num_nodes = int(num_agents)
        self.hidden = int(hid_channels)
        self.K = int(args.iterations)
        self.dropout = nn.Dropout(float(args.dropout))
        self.agent_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels, self.hidden),
                    nn.ELU(),
                    nn.Linear(self.hidden, self.hidden),
                    nn.ELU(),
                )
                for _ in range(self.num_nodes)
            ]
        )
        self.node_classifier_heads = nn.ModuleList(
            [nn.Linear(self.hidden, out_channels) for _ in range(self.num_nodes)]
        )
        self.atts = nn.ModuleList(
            [
                nn.ParameterList(
                    [nn.Parameter(torch.empty(2 * self.hidden)) for _ in range(self.num_nodes)]
                )
                for _ in range(self.K)
            ]
        )
        for layer in self.atts:
            for attention in layer:
                nn.init.xavier_uniform_(attention.view(1, -1))

    def forward(self, x, edge_index):
        batch_size = x.shape[0]
        encoded = torch.stack(
            [encoder(self.dropout(x[:, i])) for i, encoder in enumerate(self.agent_encoders)],
            dim=1,
        )
        hidden = encoded.reshape(batch_size * self.num_nodes, self.hidden)
        row, col = edge_index.long()
        for hop in range(self.K):
            source_agent = row.remainder(self.num_nodes)
            attention_vectors = torch.stack(
                [self.atts[hop][int(i)] for i in source_agent.tolist()], dim=0
            )
            pairs = torch.cat((hidden[row], hidden[col]), dim=-1)
            scores = torch.nn.functional.softplus((pairs * attention_vectors).sum(dim=-1))
            degree = hidden.new_zeros(hidden.shape[0])
            degree.index_add_(0, row, scores)
            weights = scores / degree[row].clamp_min(1e-8)
            messages = hidden[col] * weights.unsqueeze(-1)
            aggregated = torch.zeros_like(hidden)
            aggregated.index_add_(0, row, messages)
            hidden = hidden + aggregated

        hidden = hidden.reshape(batch_size, self.num_nodes, self.hidden)
        return torch.stack(
            [head(hidden[:, i]) for i, head in enumerate(self.node_classifier_heads)],
            dim=1,
        )


GNN_Model = DenseGNNModel
