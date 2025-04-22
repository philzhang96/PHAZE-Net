import torch
from InERTIA.region_weights import REGION_WEIGHTS
import numpy as np


class RegionAggregator:
    def __init__(self):
        self.region_weights = REGION_WEIGHTS  # Emotion-specific region weights

    def aggregate(
        self,
        region_outputs: dict,
        confidences: dict,
        emotion: str,
        motion_dict: dict = None,
        frame_index: int = None
    ) -> torch.Tensor:
        """
        Aggregates region-level outputs using emotion-specific region importance,
        modulated by region confidence and motion-based weighting.

        Args:
            region_outputs (dict): region name -> feature vector (torch.Tensor)
            confidences (dict): region name -> scalar confidence
            emotion (str): current predicted emotion
            motion_dict (dict): region name -> np.array of motion values
            frame_index (int): index into motion array

        Returns:
            torch.Tensor: Aggregated feature vector.
        """
        if emotion not in self.region_weights:
            raise ValueError(f"Unknown emotion '{emotion}' for region weighting.")

        weighted_sum = None
        total_weight = 0.0
        emotion_weights = self.region_weights[emotion]

        for region in region_outputs:
            if region not in confidences:
                continue  # Skip regions with no confidence available

            vector = region_outputs[region]
            confidence = confidences[region]

            base_region = region.split("::")[0]
            region_importance = emotion_weights.get(base_region, 1.0)
            adjusted_weight = confidence * region_importance

            if weighted_sum is None:
                weighted_sum = adjusted_weight * vector
            else:
                weighted_sum += adjusted_weight * vector

            total_weight += adjusted_weight


        if total_weight == 0:
            raise ValueError("Total weight for region aggregation is zero.")

        return weighted_sum / total_weight
