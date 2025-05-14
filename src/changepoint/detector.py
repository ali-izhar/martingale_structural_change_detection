# src/changepoint/detector.py

"""Change point detection using the martingale framework derived from conformal prediction.

This module implements online change point detection using exchangeability martingales
derived from conformal prediction. The detector monitors a sequence of observations
and reports changes when the martingale value exceeds a specified threshold.

Mathematical Framework:
---------------------
1. Conformal Prediction:
   - For a new observation xₙ, compute a nonconformity score and a corresponding p-value.
2. Martingale Construction:
   - Update the martingale: Mₙ = Mₙ₋₁ * g(pₙ), where g(p) is a valid betting function.
3. Change Detection:
   - Report a change when Mₙ > λ (detection threshold).

The detector supports two methods:
- Single-view: Processes a single feature sequence.
- Multiview: Processes multiple features independently and then combines evidence.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    final,
)

import logging
import numpy as np
import numpy.typing as npt
from numpy import floating, integer

from .betting import BettingFunctionConfig
from .martingale_base import MartingaleConfig
from .martingale_traditional import (
    compute_traditional_martingale,
    multiview_traditional_martingale,
)
from .distance import DistanceConfig
from .strangeness import StrangenessConfig

logger = logging.getLogger(__name__)

# Type definitions:
# ScalarType can be any floating point or integer numpy type.
ScalarType = TypeVar("ScalarType", bound=Union[floating, integer])
Array = npt.NDArray[np.float64]
DetectionMethod = Literal["single_view", "multiview"]


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the change point detector.

    Attributes:
        method: Detection method ('single_view' or 'multiview')
        threshold: Detection threshold for martingale values
        batch_size: Batch size for multiview processing
        reset: Whether to reset after detection
        max_window: Maximum window size for strangeness computation
        betting_func_config: Configuration for the betting function
        distance_measure: Distance metric for strangeness computation
        distance_p: Order parameter for Minkowski distance
        random_state: Random seed for general detector components
        betting_random_state: Optional separate random seed for betting functions
        strangeness_seed: Optional separate random seed for strangeness calculation
        pvalue_seed: Optional separate random seed for p-value computation
    """

    method: DetectionMethod = "multiview"
    threshold: float = 60.0
    batch_size: int = 1000
    reset: bool = True
    max_window: Optional[int] = None
    betting_func_config: Optional[BettingFunctionConfig] = None
    distance_measure: str = "euclidean"
    distance_p: float = 2.0
    random_state: Optional[int] = 42
    betting_random_state: Optional[int] = None
    strangeness_seed: Optional[int] = None
    pvalue_seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        # Ensure that the detection threshold is positive.
        if self.threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {self.threshold}")

        # Batch size must be at least 1.
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {self.batch_size}")
        # If specified, the max_window must be at least 1.
        if self.max_window is not None and self.max_window < 1:
            raise ValueError(
                f"Max window must be at least 1 if specified, got {self.max_window}"
            )
        # The distance order parameter must be positive.
        if self.distance_p <= 0:
            raise ValueError(
                f"Distance order parameter must be positive, got {self.distance_p}"
            )


class ChangePointDetector(Generic[ScalarType]):
    """Detector class for identifying change points in sequential data.

    The detector uses the martingale framework derived from conformal prediction
    to monitor a sequence of observations and report changes when the martingale
    value exceeds a specified threshold.

    Main Steps:
    1. Compute a strangeness measure for new points.
    2. Convert strangeness to a p-value via nonparametric ranking.
    3. Update a martingale using a chosen betting function.
    4. Report a change point if the martingale exceeds the threshold.

    Type Parameters:
    ---------------
    ScalarType : Union[floating, integer]
        The numpy dtype of the input data. Must be a floating point or integer type.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize the detector with configuration.

        Args:
            config: Detector configuration. If None, uses default configuration.
        """
        # Use provided configuration or fall back to the default settings.
        self.config = config or DetectorConfig()

        # If no betting function config is provided, set a default configuration.
        if self.config.betting_func_config is None:
            # Create default config, using betting_random_state if available
            self.config.betting_func_config = BettingFunctionConfig(
                name="power",
                params={"epsilon": 0.7},
                random_seed=self.config.betting_random_state,
            )
            logger.debug(
                f"No betting function config provided. Using default: {self.config.betting_func_config}"
            )
        elif (
            self.config.betting_random_state is not None
            and self.config.betting_func_config.random_seed is None
        ):
            # If betting_random_state is provided but not already set in the betting function config,
            # create a new betting function config with the random seed
            self.config.betting_func_config = BettingFunctionConfig(
                name=self.config.betting_func_config.name,
                params=self.config.betting_func_config.params,
                random_seed=self.config.betting_random_state,
            )

        # Initialize the internal state of the detector.
        self._reset_state()

        # Log configuration parameters for debugging purposes.
        logger.debug(f"Initialized {self.config.method} detector with:")
        logger.debug(f"  Threshold: {self.config.threshold}")
        logger.debug(f"  Batch size: {self.config.batch_size}")
        logger.debug(f"  Reset: {self.config.reset}")
        logger.debug(f"  Max window: {self.config.max_window}")
        logger.debug(
            f"  Distance: {self.config.distance_measure} (p={self.config.distance_p})"
        )
        logger.debug(f"  Core random state: {self.config.random_state}")
        logger.debug(f"  Betting function: {self.config.betting_func_config.name}")
        logger.debug(
            f"  Betting random state: {self.config.betting_func_config.random_seed}"
        )

    def _reset_state(self) -> None:
        """Reset the detector's internal state."""
        # Initialize state tracking
        self._traditional_state = None
        # Initialize the list to store detected change points.
        self._change_points: List[int] = []
        self._early_warnings: List[int] = []

    @property
    def change_points(self) -> List[int]:
        """Get the detected change points."""
        # Return a copy to prevent external modifications of the internal list.
        return self._change_points.copy()

    @property
    def early_warnings(self) -> List[int]:
        """Get the detected early warnings."""
        # Return a copy to prevent external modifications of the internal list.
        return self._early_warnings.copy()

    @final
    def run(
        self,
        data: Array,
        reset_state: bool = False,
    ) -> Dict[str, Any]:
        """Run the change point detection algorithm.

        The detector processes the input sequence and returns detected change points
        along with martingale values. For multiview detection, each feature is
        processed separately and evidence is combined.

        Args:
            data: Input sequence of shape (n_samples, n_features) for multiview
                 or (n_samples,) for single-view.
            reset_state: Whether to reset the detector state before running.

        Returns:
            Dictionary containing:
            - traditional_change_points: List of indices where traditional martingale detected changes
            - early_warnings: List of indices where early warning signals were detected
            - traditional_martingales: Array of traditional martingale values
            For multiview detection, also includes:
            - traditional_sum_martingales: Sum of traditional martingales across features
            - traditional_avg_martingales: Average of traditional martingales
            - individual_traditional_martingales: List of martingales for each feature

        Raises:
            ValueError: If input data is empty or invalid.
            RuntimeError: If detection fails.
        """
        try:
            # Reset state if requested
            if reset_state:
                self._reset_state()

            # Validate input dimensions and types.
            self._validate_input(data)

            # Run the appropriate detection method based on the configuration.
            if self.config.method == "single_view":
                results = self._run_single_view(data)
            else:  # For multiview detection.
                results = self._run_multiview(data)

            # Update internal change points and early warnings tracking
            self._change_points.extend(results.get("traditional_change_points", []))
            self._early_warnings.extend(results.get("early_warnings", []))

            return results

        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise

    def _validate_input(self, data: Array) -> None:
        """Validate input data dimensions and types.

        Args:
            data: Input sequence.

        Raises:
            ValueError: If validation fails.
        """
        if data.size == 0:
            raise ValueError("Empty input sequence")

        if self.config.method == "single_view":
            # Single-view data must be 1-dimensional.
            if len(data.shape) != 1:
                raise ValueError(
                    f"Single-view data must be 1-dimensional, got shape {data.shape}"
                )
        else:  # Multiview detection.
            # Multiview data should be 2D: (n_samples, n_features).
            if len(data.shape) != 2:
                raise ValueError(
                    f"Multiview data must have shape (n_samples, n_features), got {data.shape}"
                )

    def _run_single_view(self, data: Array) -> Dict[str, Any]:
        """Run single-view detection.

        Args:
            data: Input sequence.

        Returns:
            Detection results dictionary.
        """
        # Create martingale config from detector config
        martingale_config = MartingaleConfig(
            threshold=self.config.threshold,
            reset=self.config.reset,
            window_size=self.config.max_window,
            random_state=self.config.random_state,
            strangeness_seed=self.config.strangeness_seed,
            pvalue_seed=self.config.pvalue_seed,
            betting_func_config=self.config.betting_func_config,
            distance_measure=self.config.distance_measure,
            distance_p=self.config.distance_p,
            strangeness_config=StrangenessConfig(
                n_clusters=1,
                batch_size=None,
                random_state=self.config.strangeness_seed or self.config.random_state,
                distance_config=DistanceConfig(
                    metric=self.config.distance_measure,
                    p=self.config.distance_p,
                ),
            ),
        )

        # First, compute traditional martingale
        trad_results = compute_traditional_martingale(
            data=data.tolist(),  # Convert data to a list
            config=martingale_config,
            state=self._traditional_state,
        )

        # Update traditional state for future calls
        self._traditional_state = trad_results.get("state")
        # Merge both results
        combined_results = {
            "traditional_change_points": trad_results.get(
                "traditional_change_points", []
            ),
            "traditional_martingales": trad_results.get("traditional_martingales"),
        }

        return combined_results

    def _run_multiview(self, data: Array) -> Dict[str, Any]:
        """Run multiview detection.

        Args:
            data: Input sequence.

        Returns:
            Detection results dictionary.
        """
        # Split each feature (column) into a separate view.
        views = []
        for i in range(data.shape[1]):
            # Extract the i-th column and convert to list of scalars
            views.append([float(x) for x in data[:, i]])

        # Create martingale config from detector config
        martingale_config = MartingaleConfig(
            threshold=self.config.threshold,
            reset=self.config.reset,
            window_size=self.config.max_window,
            random_state=self.config.random_state,
            strangeness_seed=self.config.strangeness_seed,
            pvalue_seed=self.config.pvalue_seed,
            betting_func_config=self.config.betting_func_config,
            distance_measure=self.config.distance_measure,
            distance_p=self.config.distance_p,
            strangeness_config=StrangenessConfig(
                n_clusters=1,
                batch_size=None,
                random_state=self.config.strangeness_seed or self.config.random_state,
                distance_config=DistanceConfig(
                    metric=self.config.distance_measure,
                    p=self.config.distance_p,
                ),
            ),
        )

        # First, call the multiview traditional martingale
        trad_results = multiview_traditional_martingale(
            data=views,
            config=martingale_config,
            state=self._traditional_state,
            batch_size=self.config.batch_size,
        )

        # Update traditional state for future calls
        self._traditional_state = trad_results.get("state")

        # Merge results from both functions
        combined_results = {
            # Traditional results
            "traditional_change_points": trad_results.get(
                "traditional_change_points", []
            ),
            "traditional_sum_martingales": trad_results.get(
                "traditional_sum_martingales"
            ),
            "traditional_avg_martingales": trad_results.get(
                "traditional_avg_martingales"
            ),
            "individual_traditional_martingales": trad_results.get(
                "individual_traditional_martingales"
            ),
        }

        return combined_results
