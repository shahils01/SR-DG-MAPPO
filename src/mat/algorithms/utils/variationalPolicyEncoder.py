import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyVAE(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, n_agent, latent_dim=64):
        """
        Variational Autoencoder for compressing policy network weights

        Args:
            latent_dim: Dimension of the latent space (default: 64)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_agent = n_agent

        # Get total number of parameters in the policy network
        #self.total_params = 1548
        #self.total_params = (obs_dim + 2*n_embd)*n_embd + n_embd + n_embd*n_embd + n_embd + n_embd*action_dim + action_dim
        self.total_params = action_dim*n_embd #21892 #68100

        self.encoders = nn.ModuleList()
        self.fc_mus = nn.ModuleList()
        self.fc_logvars = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in range(n_agent):
            # Encoder networks (q(z|x))
            encoder = nn.Sequential(
                nn.Linear(self.total_params, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
            )
            self.encoders.append(encoder)

            # Latent space means and log variances
            fc_mu = nn.Linear(512, latent_dim)
            self.fc_mus.append(fc_mu)

            fc_logvar = nn.Linear(512, latent_dim)
            self.fc_logvars.append(fc_logvar)

            # Decoder network (p(x|z))
            decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, self.total_params),
            )
            self.decoders.append(decoder)

    def encode(self, flat_weights):
        """Encode flattened weights into latent distribution parameters"""
        fc_mu = []
        fc_logvar = []
        for n in range(self.n_agent):
            h = self.encoders[n](flat_weights[n, :])
            fc_mu_n = self.fc_mus[n](h)
            fc_logvar_n = self.fc_logvars[n](h)
            fc_mu.append(fc_mu_n)
            fc_logvar.append(fc_logvar_n)

        fc_mu = torch.stack(fc_mu, dim=0)
        fc_logvar = torch.stack(fc_logvar, dim=0)

        return fc_mu, fc_logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector back into flattened weights"""
        out = []
        for n in range(self.n_agent):
            out_n = self.decoders[n](z[n,:])
            out.append(out_n)

        out = torch.stack(out, dim=0)

        return out

    def forward(self, flat_weights):
        """Full VAE forward pass"""
        mu, logvar = self.encode(flat_weights)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def flatten_weights(self, policy):
        """Flatten policy network weights into a single vector"""
        out = []
        for n in range(self.n_agent):
            out_n = torch.cat([p.data.view(-1) for p in policy[n].parameters()])#.view(1,-1)
            out.append(out_n)

        out = torch.stack(out, dim=0)

        return out.detach()#.squeeze()

    def reconstruct_policy(self, z):
        """Reconstruct policy network from latent vector"""
        flat_weights = self.decode(z)
        return self.unflatten_weights(flat_weights)

    def unflatten_weights(self, policy, flat_weights):
        """Unflatten batched weight vectors back into policy networks

        Args:
            policy: Template policy network (nn.Module)
            flat_weights: Tensor of shape (n_agent, total_params)

        Returns:
            List of reconstructed policy networks
        """
        policy_copies = []

        for agent_weights in flat_weights:  # Iterate over each agent's weights
            # Create a fresh copy of the policy network
            policy_copy = type(policy)(*list(policy.children()))
            pointer = 0

            # Reconstruct each parameter
            for param in policy_copy.parameters():
                num_params = param.numel()
                param.data = agent_weights[pointer:pointer+num_params].view(param.size())
                pointer += num_params

            policy_copies.append(policy_copy)

        return policy_copies