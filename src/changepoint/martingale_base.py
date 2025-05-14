# src/changepoint/martingale_base.py

"""Base components and interfaces for martingale-based change detection."""

from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import logging
import numpy as np
from numpy import floating, integer

from .betting import BettingFunctionConfig
from .strangeness import StrangenessConfig


logger = logging.getLogger(__name__)

# Type definitions for scalars and arrays
ScalarType = TypeVar("ScalarType", bound=Union[floating, integer])
Array = np.ndarray
DataPoint = Union[List[float], np.ndarray]


@dataclass(frozen=True)
class MartingaleConfig:
    """Configuration for martingale computation.

    Attributes:
        threshold: Detection threshold for martingale values.
        reset: Whether to reset after detection.
        window_size: Maximum window size for strangeness computation.
        random_state: Random seed for general randomization.
        strangeness_seed: Optional separate random seed for strangeness calculation.
        pvalue_seed: Optional separate random seed for p-value computation.
        betting_func_config: Configuration for betting function.
        distance_measure: Distance metric for strangeness computation.
        distance_p: Order parameter for Minkowski distance.
        strangeness_config: Configuration for strangeness computation.
    """

    threshold: float
    reset: bool = True
    window_size: Optional[int] = None
    random_state: Optional[int] = None
    strangeness_seed: Optional[int] = None
    pvalue_seed: Optional[int] = None
    betting_func_config: Optional[BettingFunctionConfig] = None
    distance_measure: str = "euclidean"
    distance_p: float = 2.0
    strangeness_config: Optional[StrangenessConfig] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {self.threshold}")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError(
                f"Window size must be at least 1 if specified, got {self.window_size}"
            )
        if self.distance_p <= 0:
            raise ValueError(
                f"Distance order parameter must be positive, got {self.distance_p}"
            )


@dataclass
class MartingaleState:
    """Base state for single-view martingale computation.

    Attributes:
        window: Rolling window of past observations.
        traditional_martingale: Current traditional martingale value.
        saved_traditional: History of traditional martingale values.
        traditional_change_points: Indices where traditional martingale detected changes.
    """

    window: List[DataPoint] = field(default_factory=list)
    traditional_martingale: float = 1.0
    saved_traditional: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)

    def reset(self):
        """Reset martingale state after a detection event."""
        self.window.clear()
        self.traditional_martingale = 1.0
        # Append reset values to the history for continuity
        self.saved_traditional.append(1.0)


class MartingaleResult(Protocol):
    """Protocol defining the result format for martingale computations."""

    traditional_change_points: List[int]
    traditional_martingales: np.ndarray


@dataclass
class MultiviewMartingaleState:
    """Base state for multiview martingale computation.

    Attributes:
        windows: List of rolling windows for each feature.
        traditional_martingales: Current traditional martingale values per feature.
        traditional_sum: Sum of traditional martingales across features.
        traditional_avg: Average of traditional martingales.
        traditional_change_points: Indices where traditional martingale detected changes.
        individual_traditional: Martingale history for each individual feature.
        current_timestep: The current timestep being processed.
        has_detection: Flag indicating if a detection has occurred at the current timestep.
    """

    windows: List[List[DataPoint]] = field(default_factory=list)
    traditional_martingales: List[float] = field(default_factory=list)
    traditional_sum: List[float] = field(default_factory=lambda: [1.0])
    traditional_avg: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)
    individual_traditional: List[List[float]] = field(default_factory=list)
    current_timestep: int = 0
    has_detection: bool = False

    def __post_init__(self):
        """Initialize state lists if they are not already set."""
        if not self.windows:
            self.windows = []
        if not self.traditional_martingales:
            self.traditional_martingales = []
        if not self.individual_traditional:
            self.individual_traditional = []

    def record_traditional_values(
        self,
        timestep: int,
        traditional_values: List[float],
        is_detection: bool = False,
    ):
        """Record traditional martingale values at specific timestep.

        Args:
            timestep: The timestep to record values for
            traditional_values: List of traditional martingale values per feature
            is_detection: Whether this recording is for a detection event
        """
        self.current_timestep = timestep
        num_features = len(traditional_values)

        # Calculate sums and averages
        total_traditional = sum(traditional_values)
        avg_traditional = total_traditional / num_features

        # Ensure lists are long enough using manual extension to match martingale.py
        while len(self.traditional_sum) <= timestep:
            self.traditional_sum.append(1.0)
        while len(self.traditional_avg) <= timestep:
            self.traditional_avg.append(1.0)

        # Record traditional values
        self.traditional_sum[timestep] = total_traditional
        self.traditional_avg[timestep] = avg_traditional

        # Update individual traditional martingales
        for j in range(num_features):
            while len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            while len(self.individual_traditional[j]) <= timestep:
                self.individual_traditional[j].append(1.0)
            self.individual_traditional[j][timestep] = traditional_values[j]

        # If this is a detection event, mark it
        if is_detection:
            self.has_detection = True

    def reset(self, num_features: int):
        """Reset state for all features.

        Args:
            num_features: Number of features to initialize.
        """
        # Reset each feature's rolling window.
        self.windows = [[] for _ in range(num_features)]

        # Reset martingale values for each feature to 1.0.
        self.traditional_martingales = [1.0] * num_features

        # Reset detection flag
        self.has_detection = False

        # Add reset values to history for continuity
        current_t = self.current_timestep

        # Since we're resetting after the current timestep, we need to add reset values
        # for the next timestep
        next_t = current_t + 1

        # Update overall sum and average with the reset values
        while len(self.traditional_sum) <= next_t:
            self.traditional_sum.append(1.0)
        while len(self.traditional_avg) <= next_t:
            self.traditional_avg.append(1.0)

        self.traditional_sum[next_t] = float(num_features)
        self.traditional_avg[next_t] = 1.0

        # Reset individual martingale histories per feature
        for j in range(num_features):
            while len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            while len(self.individual_traditional[j]) <= next_t:
                self.individual_traditional[j].append(1.0)
            self.individual_traditional[j][next_t] = 1.0
