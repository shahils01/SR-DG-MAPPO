# MARL integration

## Data path

The ported long-range predator–prey environment emits one local observation
per predator. It contains the receiver's pose, masked relative predator slots,
masked relative prey slots, visibility, and alive flags. `sr_mappo` parses only
those local fields:

1. convert visible prey offsets from world axes into the sender's body frame;
2. encode the fixed-size local prey map with product vector quantization;
3. decode and transport neighbor maps into each receiver frame using relative
   orthogonal frames and positions;
4. confidence-fuse maps for the configured number of communication rounds;
5. project the fused map into the DG-MAPPO actor and critic feature width.

The graph adjacency comes from the environment's communication-radius graph.
The replay buffer stores raw observations and adjacency. It does not store
detached SR embeddings.

## Joint objective

Each PPO minibatch recomputes the communication graph and minimizes

```text
L = L_PPO
    + sr_reconstruction_coef * L_local_reconstruction
    + sr_vq_coef * L_product_VQ.
```

Logged SR metrics include:

- `sr_auxiliary_loss`;
- `sr_reconstruction_loss`;
- `sr_vq_loss`;
- `sr_codebook_perplexity`;
- `sr_map_coverage`;
- `sr_bits_per_message` and `sr_bits_per_agent`.

## Palmetto smoke run

After installing with `pip install -e '.[marl]'`, request a GPU interactively or
through a batch job and run:

```bash
export SR_DG_MAPPO_RESULTS="$SCRATCH/sr-dg-mappo-results"

sr-dg-mappo-train \
  --algorithm_name sr_mappo \
  --env_device cuda \
  --num_env_steps 2048 \
  --episode_length 64 \
  --env_episode_length 64 \
  --n_rollout_threads 8 \
  --ppo_epoch 2 \
  --mini_batch_size 256 \
  --n_embd 64 \
  --sr_hidden_dim 64 \
  --sr_latent_dim 32 \
  --sr_num_codebooks 4 \
  --sr_codebook_size 16 \
  --sr_comm_rounds 2 \
  --experiment_name palmetto_smoke \
  --seed 1
```

Success means one or more PPO updates complete, a `transformer_*.pt` checkpoint
appears under the results directory, and all reported SR metrics are finite.

## First controlled experiment

Run at least three matched seeds for `mappo_dgnn` and `sr_mappo`. Keep the
environment, network width, rollout count, training steps, and seed set fixed.
Compare capture success, episode return, collision count, inference coverage,
and bits per agent per step. Do not interpret a rate reduction as a control
improvement unless the return or capture metrics are competitive at the same
training budget.

## Current limitations

- The code assumes relative pose is available without communication charge.
- The square arena has exact `D4`, not unrestricted `SE(2)`, task symmetry.
- The current SR map has fixed prey identity slots and is not yet permutation
  invariant to prey relabeling.
- Only the original DG-MAPPO and full SR configurations are wired. Quantized
  non-equivariant, equivariant uncompressed, and no-communication ablations
  remain to be added.
- SMAC integration is not part of this milestone.
