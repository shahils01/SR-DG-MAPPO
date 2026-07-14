# SR-DG-MAPPO

**Symmetry-Reduced Distributed Graph-Attention MAPPO** is an initial research
prototype for testing whether agents can infer global scene content while
transmitting fewer bits by communicating in local coordinate frames.

This repository deliberately isolates the communication hypothesis before it
is coupled to a full PPO implementation. It provides:

- analytic `O(2)` sender-to-receiver frame transport, including rotations and reflections;
- a product-vector-quantized scene-map codec with a fixed, auditable bit rate;
- an uncompressed matched-architecture baseline;
- fixed-size multi-hop map fusion over a dynamic communication graph;
- exact `D4` transformations for square-world symmetry tests;
- quotient-space reconstruction error, coverage, and communication-rate metrics;
- a synthetic predator-prey-style experiment that trains without global-state supervision.

The main design principle is:

> Learn the scene content; compute known relative geometry analytically.

A sender encodes target positions in its own local frame. The receiver decodes
the message and transports the result into its own frame using the relative
pose. Global rotations, translations, or reflections therefore do not need to
be relearned by the content codec.

## Status

This is a **communication and state-inference prototype**, not yet a complete
MAPPO trainer. It is intended to de-risk the codec, symmetry, and rate-distortion
parts before integration into DG-MAPPO.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
```

## Run the demonstration

```bash
sr-dg-mappo-demo --steps 200 --rounds 2
```

or without installation:

```bash
PYTHONPATH=src python -m sr_dg_mappo.demo --steps 200 --rounds 2
```

The command prints JSON containing:

- local reconstruction loss;
- receiver-frame global-map distortion and coverage;
- `D4` quotient reconstruction error;
- maximum equivariance error after a global transformation;
- bits per message and bits per agent per communication step;
- the corresponding uncompressed float32 baseline.

A deterministic 200-step CPU reference run with the default configuration
produced a 16-bit message versus 192 bits for the uncompressed baseline
(91.7% payload reduction), while the maximum frame-equivariance error remained
below `5e-7`. These are prototype diagnostics rather than MARL performance
claims; the learned codec trades the lower rate for non-zero reconstruction
distortion.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Core formulation

For sender `j` and receiver `i`, a point expressed in the sender frame is
transported as

```text
p_i = F_i^T (t_j + F_j p_j - t_i),
```

where `t` is position and `F` is an orthogonal local frame. The learned message
contains only a quantized description of `p_j` and its confidence. Relative pose
is treated as known side information and is not charged as message content in
the demo; an integration must charge it when the deployment cannot infer it
from sensing.

For `Q` product-codebook tokens and codebook size `K`, each fixed-rate message
uses

```text
Q * ceil(log2(K)) bits.
```

See [docs/design.md](docs/design.md) for assumptions, limitations, and the
planned DG-MAPPO integration.

## Research baselines

An eventual MARL evaluation should compare at matched total bits:

1. original float32 DG-MAPPO communication;
2. non-equivariant quantized DG-MAPPO;
3. equivariant but uncompressed communication;
4. symmetry-reduced quantized communication;
5. no communication.

This separation is essential: symmetry alone does not establish a bandwidth
gain, and quantization alone does not establish a symmetry benefit.

## License

MIT
