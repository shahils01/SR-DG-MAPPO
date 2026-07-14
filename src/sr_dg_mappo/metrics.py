"""Rate, distortion, orbit, and equivariance metrics."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from sr_dg_mappo.groups import apply_isometry, d4_matrices


def map_metrics(
    predicted_points: Tensor,
    predicted_confidence: Tensor,
    target_points: Tensor,
    threshold: float = 0.5,
    missing_penalty: float = 2.0,
) -> Dict[str, Tensor]:
    """Measure coordinate accuracy, target coverage, and confidence-aware distortion."""

    squared_distance = (predicted_points - target_points).square().sum(dim=-1)
    detected = predicted_confidence >= threshold
    detected_count = detected.sum().clamp_min(1)
    coordinate_rmse = torch.sqrt((squared_distance * detected).sum() / detected_count)
    coverage = detected.to(predicted_points.dtype).mean()
    soft_distortion = (
        predicted_confidence * squared_distance
        + (1.0 - predicted_confidence) * (missing_penalty**2)
    ).mean()
    return {
        "coordinate_rmse": coordinate_rmse,
        "coverage": coverage,
        "soft_distortion": soft_distortion,
    }


def d4_orbit_mse(
    prediction: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Return mean per-scene squared error minimized over the square's D4 orbit."""

    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must share shape [..., points, 2]")
    if prediction.ndim < 3:
        raise ValueError("an explicit batch dimension is required")
    matrices = d4_matrices(dtype=prediction.dtype, device=prediction.device)
    errors = []
    for matrix in matrices:
        transformed = apply_isometry(prediction, matrix)
        point_error = (transformed - target).square().sum(dim=-1)
        if weight is None:
            scene_error = point_error.mean(dim=-1)
        else:
            scene_error = (point_error * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1e-8)
        errors.append(scene_error)
    return torch.stack(errors, dim=-1).min(dim=-1).values.mean()


def equivariance_error(reference: Tensor, transformed_result: Tensor) -> Tensor:
    """Maximum absolute difference between gauge-fixed outputs."""

    if reference.shape != transformed_result.shape:
        raise ValueError("equivariance comparison shapes must match")
    return (reference - transformed_result).abs().max()
