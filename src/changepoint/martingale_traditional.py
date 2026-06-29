# src/changepoint/martingale_traditional.py

"""Traditional martingale implementation for change point detection.
Traditional martingale uses the current observation and previous history."""

import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

from .martingale_base import (
    MartingaleConfig,
    MartingaleState,
    MultiviewMartingaleState,
    DataPoint,
)

from .betting import (
    create_betting_function,
)
from .strangeness import (
    strangeness_point,
    get_pvalue,
)


logger = logging.getLogger(__name__)


def compute_traditional_martingale(
    data: List[DataPoint],
    config: Optional[MartingaleConfig] = None,
    state: Optional[MartingaleState] = None,
) -> Dict[str, Any]:
    """Compute a traditional martingale for online change detection over a univariate data stream.

    Uses conformal p-values and a chosen strangeness measure to compute a traditional martingale
    that uses only the current observation with its history.

    Args:
        data: Sequential observations to monitor.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.

    Returns:
        Dictionary containing:
         - "traditional_change_points": List[int] of indices where martingale detected a change.
         - "traditional_martingales": np.ndarray of martingale values over time.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data:
        raise ValueError("Empty data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = MartingaleState()

    # Obtain the betting function callable based on the betting_func_config.
    betting_function = create_betting_function(config.betting_func_config)

    # NOTE: create the persistent smoothed-conformal RNG ONCE, before the
    # per-sample loop, seeded deterministically from the configured seed. θ_n is
    # then drawn fresh at every timestep (decorrelated across steps) while staying
    # bit-for-bit reproducible given the seed.
    pvalue_seed = config.pvalue_seed or config.random_state
    pvalue_rng = np.random.default_rng(pvalue_seed)

    # Log input dimensions and configuration details.
    logger.debug("Single-view Traditional Martingale Input Dimensions:")
    logger.debug(f"  Sequence length: {len(data)}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug("-" * 50)

    try:
        # Process each point in the data stream.
        for i, point in enumerate(data):
            # Maintain a rolling window if a window_size is set.
            if config.window_size and len(state.window) >= config.window_size:
                state.window = state.window[-config.window_size :]

            # Compute strangeness for the current observation.
            # If the window is empty, default strangeness is set to 0.
            if len(state.window) == 0:
                s_vals = [0.0]
            else:
                # Reshape the window data to 2D array (n_samples, 1)
                window_data = np.array(state.window + [point]).reshape(-1, 1)
                s_vals = strangeness_point(
                    window_data,
                    config=config.strangeness_config,
                    random_state=config.strangeness_seed or config.random_state,
                )

            # Compute conformal p-value using the strangeness scores.
            # Pass the persistent generator so θ_n is fresh each step (FIX 1).
            pvalue = get_pvalue(s_vals, rng=pvalue_rng)

            # Update traditional martingale using the betting function.
            prev_trad = state.traditional_martingale
            new_trad = betting_function(prev_trad, pvalue)

            # Store the updated martingale value before checking for change detection
            state.saved_traditional.append(new_trad)
            state.traditional_martingale = new_trad

            # Check if the updated traditional martingale exceeds the threshold.
            detected_trad = False
            if config.reset and new_trad > config.threshold:
                detected_trad = True
                # NOTE: record at the crossing index i (was i-1).
                # Volkhonskiy stopping rule τ = inf{n : M_n ≥ h}; no -1 offset.
                state.traditional_change_points.append(i)

            # Update window or reset state if a change is detected.
            if detected_trad:
                state.reset()
                # We already saved the martingale value that crossed the threshold,
                # so the reset happens after recording the detection
            else:
                state.window.append(point)

        # Return the computed martingale histories and detected change points.
        # NOTE: saved_traditional = [1.0(init), m_0, ..., m_{N-1}] now has
        # length N+1 exactly (reset no longer injects an extra baseline element),
        # so [1:1+N] is the length-N, time-aligned series: index t = martingale
        # after processing sample t (the crossing value at a detection step t).
        n_samples = len(data)
        return {
            "traditional_change_points": state.traditional_change_points,
            "traditional_martingales": np.array(
                state.saved_traditional[1 : 1 + n_samples], dtype=float
            ),
        }

    except Exception as e:
        logger.error(f"Traditional martingale computation failed: {str(e)}")
        raise RuntimeError(f"Traditional martingale computation failed: {str(e)}")


def multiview_traditional_martingale(
    data: List[List[DataPoint]],
    config: Optional[MartingaleConfig] = None,
    state: Optional[MultiviewMartingaleState] = None,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Compute a multivariate (multiview) traditional martingale test by aggregating evidence across features.

    For d features, each feature maintains its own martingale computed using the traditional update
    (current observation + history). The combined martingale is defined as:
         M_total(n) = sum_{j=1}^{d} M_j(n)
         M_avg(n) = M_total(n) / d
    A change is declared if M_total(n) exceeds the threshold.

    Args:
        data: List of feature sequences to monitor.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        batch_size: Size of batches for processing.
        silent: Whether to suppress logging of detection events.

    Returns:
        Dictionary containing change points and martingale values.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data or not data[0]:
        raise ValueError("Empty data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = MultiviewMartingaleState()
        state.reset(len(data))

    # Get the betting function based on the provided configuration.
    betting_function = create_betting_function(config.betting_func_config)

    # NOTE: create one persistent smoothed-conformal RNG PER FEATURE,
    # ONCE before the per-sample loop. Each feature j gets its own independent
    # stream via SeedSequence(seed).spawn(num_features) so the θ_n draws are
    # decorrelated across features yet fully reproducible given the seed.
    pvalue_seed = config.pvalue_seed or config.random_state
    num_features = len(data)
    pvalue_rngs = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence(pvalue_seed).spawn(num_features)
    ]

    # Log input dimensions and configuration details.
    logger.debug("Multiview Traditional Martingale Input Dimensions:")
    logger.debug(f"  Number of features: {len(data)}")
    logger.debug(f"  Sequence length per feature: {len(data[0])}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug("-" * 50)

    try:
        # num_features already computed above for the per-feature RNGs.
        num_samples = len(data[0])

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)
            logger.debug(
                f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
            )

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # Store previous traditional values before ANY updates within each timestep
                prev_traditional_t_minus_1 = state.traditional_martingales.copy()

                # Reset new_traditional for each timestep
                new_traditional = []

                # Update traditional martingale for each feature
                for j in range(num_features):
                    # Maintain rolling window for feature j if window_size is specified.
                    if (
                        config.window_size
                        and len(state.windows[j]) >= config.window_size
                    ):
                        state.windows[j] = state.windows[j][-config.window_size :]

                    # Compute strangeness for the current observation for feature j.
                    if not state.windows[j]:
                        s_vals = [0.0]
                    else:
                        window_data = np.array(state.windows[j] + [data[j][i]]).reshape(
                            -1, 1
                        )
                        s_vals = strangeness_point(
                            window_data,
                            config=config.strangeness_config,
                            random_state=config.strangeness_seed or config.random_state,
                        )
                    # Compute p-value and update traditional martingale for feature j using M_{t-1}.
                    # draw θ_n fresh from feature j's own persistent generator.
                    pv = get_pvalue(s_vals, rng=pvalue_rngs[j])
                    prev_val = prev_traditional_t_minus_1[j]  # Use M_{t-1}
                    new_val = betting_function(prev_val, pv)
                    new_traditional.append(new_val)

                    # Update running values for each feature
                    state.traditional_martingales[j] = new_val

                # Aggregate traditional martingale values across all features
                total_traditional = sum(new_traditional)

                # Record values first, so detection is stored at the correct timestep
                state.record_traditional_values(i, new_traditional, False)

                # Check if traditional martingale crosses threshold
                if total_traditional > config.threshold:
                    # NOTE: record at the crossing index i (was i-1).
                    state.traditional_change_points.append(i)

                    # Update the is_detection flag for this timestep
                    state.has_detection = True

                    # Reset state after detection
                    state.reset(num_features)
                else:
                    # No detection - update windows
                    for j in range(num_features):
                        state.windows[j].append(data[j][i])

            idx = batch_end

        # Return the aggregated results as numpy arrays.
        # NOTE: record_traditional_values(i, ...) writes index t = sample t
        # (index 0 = sample 0, overwriting the initial 1.0 placeholder), so the
        # time-aligned series is indices [0 : num_samples], length exactly N.
        # The previous `[1:]` slice dropped sample 0 (length N-1, off-by-one); a
        # trailing reset baseline (written at index num_samples only when a
        # detection lands on the last sample) is excluded by the [:num_samples] cap.
        return {
            "traditional_change_points": state.traditional_change_points,
            "traditional_sum_martingales": np.array(
                state.traditional_sum[:num_samples], dtype=float
            ),
            "traditional_avg_martingales": np.array(
                state.traditional_avg[:num_samples], dtype=float
            ),
            "individual_traditional_martingales": [
                np.array(m[:num_samples], dtype=float)
                for m in state.individual_traditional
            ],
        }

    except Exception as e:
        logger.error(f"Error in multiview traditional martingale computation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Multiview traditional martingale computation failed: {e}")
