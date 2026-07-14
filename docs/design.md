# Design notes

## Scope of the initial version

SR-DG-MAPPO now contains a trainable predator–prey MAPPO stack ported from
DG-MAPPO. The standalone diagnostic remains the smallest system that can
falsify the rate–distortion hypothesis, while `sr_mappo` evaluates whether the
same mechanism improves control at a fixed message budget.

Each agent maintains a fixed-size target map in its own coordinate frame:

```text
M_i = {(p_im, c_im)} for targets m = 1, ..., M,
```

where `p_im` is a two-dimensional relative position and `c_im` is confidence.
At every communication round, each agent encodes its current map with its own
product-VQ codec. Neighbors decode the map, analytically transport its points
into their own frames, and apply receiver-specific graph attention. Attention
logits use only invariant confidence, squared distance, and squared
disagreement; the weighted coordinate sum is performed after frame alignment.
The fused fixed-size map is re-encoded on the next round. Consequently, traffic
is fixed per directed edge per round rather than growing with the number of
message origins.

## Symmetry and gauge distinction

The code supports arbitrary orthogonal changes of global coordinates through
local-frame transport. This is coordinate, or gauge, equivariance and does not
assert that every environment has full Euclidean task symmetry.

For a bounded square predator-prey world, the exact spatial task group is
`D4`, not `SE(2)`. Arbitrary translations and rotations are broken by walls.
`D4` is therefore used for exact orbit metrics. A future benchmark without
symmetry-breaking boundaries is required for exact `SE(2)` task claims.

## Training information boundary

The codec is trained to reconstruct the sender's own local map. It does not use
the simulator global state as a training target. Ground-truth target positions
are used only for evaluation of distributed inference. This preserves the
fully distributed information boundary.

An oracle experiment may train against global state, but it must be labeled as
centralized auxiliary supervision and cannot be the main method.

## Rate accounting

The quantized codec reports the exact fixed token payload. The communicator
counts one payload per directed, non-self communication edge and round.

Not included by default:

- packet headers and error correction;
- relative-pose side information;
- D-GAT parameter consensus during training;
- retransmissions, contention, and channel failures.

These must be added for a physical-network claim. In DG-MAPPO, execution-time
feature traffic and training-time parameter-consensus traffic should be
reported separately.

## Integration with DG-MAPPO

The implemented integration is:

1. Replace the opaque float32 D-GAT message with codec tokens.
2. Keep the communication module inside the training computation graph; do not
   store only detached latents in the rollout buffer.
3. Feed the fused receiver-frame map, or an invariant readout of it, to each
   actor and critic alongside the raw local observation.
4. Optimize PPO loss, local reconstruction, and VQ commitment/codebook losses
   jointly. Frame equivariance is analytic for the transport layer and is
   monitored by tests rather than approximated by a learned penalty.
5. Use simulator global state only to report raw and quotient inference error.
6. Log message bits, bits per agent, reconstruction, codebook perplexity, and
   map coverage alongside PPO metrics.
7. Maintain per-agent codec, attention, readout, actor, and critic modules;
   update them with agent-local PPO gradients and then mix corresponding
   parameters over the communication graph using DG-MAPPO's D-SGD rule.

Sweeping codebook size and token count to estimate a return-versus-bits Pareto
frontier is now an experiment task rather than an architectural dependency.

## Theoretical target

Let `D_G(R, K)` be the expected quotient distortion achieved at bitrate `R`
after `K` communication rounds. The desired extension of the DG-MAPPO analysis
has the form

```text
||grad J_dist - grad J_cent|| <= L_PG D_G(R, K) + epsilon_sym,
```

where `epsilon_sym` captures task symmetry breaking. This connects actual
communication rate to value and policy-gradient approximation rather than
using latent dimension as a proxy for bandwidth.
