# SR-DG-MAPPO

**Symmetry-Reduced Distributed Graph MAPPO** is a trainable extension of
DG-MAPPO for studying rate-limited, frame-equivariant communication in
cooperative multi-agent reinforcement learning.

The repository now contains two connected layers:

- the original DG-MAPPO predator–prey environment, rollout buffer, PPO trainer,
  per-agent actor/critic heads, and distributed graph baseline;
- a learned product-VQ scene-map codec with analytic sender-to-receiver frame
  transport, multi-hop fusion, auxiliary reconstruction training, and exact
  communication-rate accounting.

The main design principle is:

> Learn scene content; compute known relative geometry analytically.

## Current milestone

The project can train MARL policies in the long-range continuous
predator–prey environment with either:

- `mappo_dgnn`: the ported DG-MAPPO graph-communication baseline;
- `sr_mappo`: PPO with differentiable symmetry-reduced quantized communication.

The rollout buffer stores raw decentralized observations and graph state.
`sr_mappo` recomputes communication inside each PPO minibatch, so gradients
reach the encoder, codebook, decoder, and policy. The simulator global state is
available to the MAPPO training interface but is not used as a reconstruction
target.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[marl]'
```

For tests and linting:

```bash
pip install -e '.[marl,dev]'
```

If `torch-geometric` is installed, `mappo_dgnn` uses the original AERO-GNN
encoder. Otherwise, the repository selects a dependency-free dense-attention
fallback with the same per-agent encoder interface.

## Train predator–prey

### Symmetry-reduced MAPPO

```bash
sr-dg-mappo-train \
  --algorithm_name sr_mappo \
  --env_device cuda \
  --num_env_steps 1000000 \
  --n_rollout_threads 64 \
  --experiment_name sr_seed1 \
  --seed 1
```

### DG-MAPPO baseline

```bash
sr-dg-mappo-train \
  --algorithm_name mappo_dgnn \
  --env_device cuda \
  --num_env_steps 1000000 \
  --n_rollout_threads 64 \
  --experiment_name dgnn_seed1 \
  --seed 1
```

Use `--no_cuda --env_device cpu` for a CPU run. Without installing the package,
the equivalent source-tree entry point is:

```bash
PYTHONPATH=src python scripts/train_predator_prey.py [arguments]
```

Checkpoints and TensorBoard logs are written beneath `results/`. Set
`SR_DG_MAPPO_RESULTS=/path/to/scratch/results` to redirect them on a cluster.
See [docs/marl-integration.md](docs/marl-integration.md) for the loss,
observation contract, Palmetto smoke command, and experiment plan.

## Communication configuration

The main `sr_mappo` controls are:

```text
--sr_comm_rounds
--sr_num_codebooks
--sr_codebook_size
--sr_latent_dim
--sr_reconstruction_coef
--sr_vq_coef
```

For `Q` product-codebook tokens and codebook size `K`, every message contains

```text
Q * ceil(log2(K)) bits.
```

The default `Q=4`, `K=16` payload is 16 bits. Relative pose is treated as
locally available side information and is not included in this payload; this
assumption must be changed for deployments that transmit pose explicitly.

## Standalone codec diagnostic

The earlier rate–distortion diagnostic remains available:

```bash
sr-dg-mappo-demo --steps 200 --rounds 2
```

The deterministic reference configuration produced a 16-bit message versus
192 bits for the uncompressed map, with frame-equivariance error below `5e-7`.
This diagnostic is not a policy-performance result.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite includes environment-contract tests, group and codec tests,
end-to-end PPO updates for both algorithms, and a gradient check proving that
the SR codebook participates in training.

## Research comparisons

The intended full study is:

1. original float-message DG-MAPPO;
2. non-equivariant quantized communication;
3. equivariant uncompressed communication;
4. full symmetry-reduced quantized communication;
5. no communication.

Current code establishes items 1 and 4. The remaining ablations are the next
experimental implementation milestone.

## Provenance and license

The `mat` training stack and long-range predator–prey environment are ported
from the companion DG-MAPPO repository and adapted for device portability and
SR communication. New and ported code is distributed under the MIT license.
