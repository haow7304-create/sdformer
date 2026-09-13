"""Indicator-balanced loss for the V2 event forecasting datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


class GroupedForecastLoss(nn.Module):
    """Balance the 41 business indicators and constrain distributions.

    The model works in StandardScaler space.  Ordinary indicator errors are
    calculated there, while distribution constraints are calculated after
    converting predictions back to their 0..1 proportion space.
    """

    def __init__(
        self,
        preprocessor_path: str,
        distribution_constraint_weight: float = 0.1,
        validity_loss_weight: float = 0.25,
    ):
        super().__init__()
        path = Path(preprocessor_path)
        with path.open("r", encoding="utf-8") as handle:
            preprocessor = json.load(handle)
        schema_path = path.parent / "feature_schema.json"
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)

        columns = preprocessor["model_columns"]
        if columns != schema["model_columns"]:
            raise ValueError("preprocessor model columns do not match feature schema")
        column_to_index = {column: index for index, column in enumerate(columns)}

        constant_columns = set(preprocessor.get("constant_columns", {}))
        self.indicator_groups: List[List[int]] = []
        self.distribution_groups: List[Dict[str, object]] = []
        for indicator, indicator_columns in schema["indicator_to_columns"].items():
            indices = [
                column_to_index[column]
                for column in indicator_columns
                if column not in constant_columns
            ]
            if not indices:
                continue
            if schema["indicator_types"][indicator] == "distribution":
                validity_column = schema["validity_columns"][indicator]
                self.distribution_groups.append(
                    {
                        "indices": indices,
                        "validity_index": column_to_index[validity_column],
                    }
                )
            else:
                self.indicator_groups.append(indices)

        self.validity_indices = [
            column_to_index[column]
            for column in schema["validity_columns"].values()
            if column not in constant_columns
        ]
        self.distribution_constraint_weight = distribution_constraint_weight
        self.validity_loss_weight = validity_loss_weight
        self.register_buffer(
            "scaler_mean", torch.tensor(preprocessor["scaler"]["mean"]).float()
        )
        self.register_buffer(
            "scaler_scale", torch.tensor(preprocessor["scaler"]["scale"]).float()
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shapes differ: {prediction.shape} vs {target.shape}"
            )
        if prediction.shape[-1] != self.scaler_mean.numel():
            raise ValueError(
                f"expected {self.scaler_mean.numel()} channels, got {prediction.shape[-1]}"
            )

        losses: List[torch.Tensor] = []
        squared_error = (prediction - target).square()
        for indices in self.indicator_groups:
            losses.append(squared_error[..., indices].mean())

        constraint_losses: List[torch.Tensor] = []
        for group in self.distribution_groups:
            indices = group["indices"]
            validity_index = group["validity_index"]
            validity_raw = (
                target[..., validity_index] * self.scaler_scale[validity_index]
                + self.scaler_mean[validity_index]
            )
            valid = (validity_raw >= 0.5).to(prediction.dtype)

            group_mse = squared_error[..., indices].mean(dim=-1)
            losses.append(self._masked_mean(group_mse, valid))

            prediction_raw = (
                prediction[..., indices] * self.scaler_scale[indices]
                + self.scaler_mean[indices]
            )
            sum_error = (prediction_raw.sum(dim=-1) - 1.0).square()
            negative_error = torch.relu(-prediction_raw).square().mean(dim=-1)
            constraint_losses.append(
                self._masked_mean(sum_error + negative_error, valid)
            )

        business_loss = torch.stack(losses).mean()
        total_loss = business_loss

        if self.validity_indices:
            validity_loss = squared_error[..., self.validity_indices].mean()
            total_loss = total_loss + self.validity_loss_weight * validity_loss
        if constraint_losses:
            constraint_loss = torch.stack(constraint_losses).mean()
            total_loss = total_loss + (
                self.distribution_constraint_weight * constraint_loss
            )
        return total_loss
