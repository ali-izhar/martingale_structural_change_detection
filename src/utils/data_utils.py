# src/utils/data_utils.py

"""Utilities for data processing, transformation, and visualization preparation."""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)


def normalize_features(
    features_numeric: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features by subtracting mean and dividing by standard deviation.

    Args:
        features_numeric: Array of numeric features with shape (n_samples, n_features)

    Returns:
        Tuple containing:
        - normalized_features: Normalized feature array
        - feature_means: Mean values used for normalization
        - feature_stds: Standard deviation values used for normalization
    """
    feature_means = np.mean(features_numeric, axis=0)
    feature_stds = np.std(features_numeric, axis=0)

    # Replace zero standard deviations with 1 to avoid division by zero
    feature_stds[feature_stds == 0] = 1.0

    normalized_features = (features_numeric - feature_means) / feature_stds

    return normalized_features, feature_means, feature_stds


def prepare_result_data(
    sequence_result: Dict[str, Any],
    features_numeric: np.ndarray,
    features_raw: List,
    trial_results: Optional[Dict[str, Any]] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Prepare and compile results data.

    Args:
        sequence_result: Dictionary containing graph sequence generation results
        features_numeric: Numeric feature values
        features_raw: Raw feature values
        trial_results: Detection trial results (optional)
        config: Configuration dictionary (optional)

    Returns:
        Dictionary containing compiled results
    """
    if config is None:
        config = {}

    # Start with basic information
    results = {
        "true_change_points": sequence_result.get("change_points", []),
        "model_name": sequence_result.get("model_name", ""),
        "params": config,
    }

    # Add data if configured to save
    output_config = config.get("output", {})
    if output_config.get("save_features", False):
        results.update(
            {
                "features_raw": features_raw,
                "features_numeric": features_numeric,
            }
        )

    # Add detection results if available
    if trial_results and output_config.get("save_martingales", False):
        if "aggregated" in trial_results:
            results.update(trial_results["aggregated"])
        # Add the individual trials data
        if "individual_trials" in trial_results:
            results["individual_trials"] = trial_results["individual_trials"]

    return results
