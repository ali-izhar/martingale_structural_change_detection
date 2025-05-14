# src/algorithm.py

"""Core pipeline for forecast-based graph structural change detection."""

import logging
import yaml
import numpy as np
import time
import os

from src.changepoint import BettingFunctionConfig, ChangePointDetector, DetectorConfig
from src.configs import get_config, get_full_model_name
from src.graph import GraphGenerator, NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.utils import (
    normalize_features,
    OutputManager,
    prepare_result_data,
)

logger = logging.getLogger(__name__)


class GraphChangeDetection:
    """Pipeline for graph change point detection.

    This class implements a complete pipeline for detecting structural changes in graph
    sequences. The pipeline has
    the following stages:

    1. Graph sequence generation
    2. Feature extraction
    3. Change point detection
    4. (Optional) Data export

    Each step is implemented as a separate method, allowing for flexible execution
    and extension of the pipeline.
    """

    def __init__(self, config_path=None, config_dict=None):
        """Initialize the pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)

        Raises:
            ValueError: If neither config_path nor config_dict is provided
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

    def _load_config(self, config_path):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dict containing configuration
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_output_directory(self):
        """Create timestamped output directory with descriptive name."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        network_type = self.config["model"]["network"]
        betting_function = self.config["detection"]["betting_func_config"]["name"]
        distance_measure = self.config["detection"]["distance"]["measure"]

        self.config["output"]["directory"] = os.path.join(
            self.config["output"]["directory"],
            f"{network_type}_{distance_measure}_{betting_function}_{timestamp}",
        )

        os.makedirs(
            self.config["output"]["directory"],
            exist_ok=True,
        )
        logger.debug(f"Created output directory: {self.config['output']['directory']}")

    def _init_generator(self):
        """Initialize the graph sequence generator.

        Returns:
            GraphGenerator instance
        """
        network_type = self.config["model"]["network"]

        return GraphGenerator(network_type)

    def _init_detector(
        self,
        random_state=None,
        strangeness_seed=None,
        pvalue_seed=None,
        betting_seed=None,
    ):
        """Initialize the change point detector.

        Args:
            random_state: Optional random seed for general detector components
            strangeness_seed: Optional separate seed for strangeness calculation
            pvalue_seed: Optional separate seed for p-value computation
            betting_seed: Optional separate seed for betting function randomization

        Returns:
            ChangePointDetector instance
        """
        det_config = self.config["detection"]

        # Ensure random_state is compatible type
        if random_state is not None:
            random_state = int(random_state)

        # Get betting function config
        betting_func_name = det_config["betting_func_config"]["name"]
        betting_func_params = det_config["betting_func_config"].get(
            betting_func_name, {}
        )

        # Create proper BettingFunctionConfig
        betting_func_config = BettingFunctionConfig(
            name=betting_func_name,
            params=betting_func_params,
            random_seed=betting_seed,  # Pass separate betting seed
        )

        # Create detector config
        detector_config = DetectorConfig(
            method=self.config["model"]["type"],
            threshold=det_config["threshold"],
            batch_size=det_config["batch_size"],
            reset=det_config["reset"],
            max_window=det_config["max_window"],
            betting_func_config=betting_func_config,
            distance_measure=det_config["distance"]["measure"],
            distance_p=det_config["distance"]["p"],
            random_state=random_state,  # Main random state
            strangeness_seed=strangeness_seed,  # Seed for strangeness calculation
            pvalue_seed=pvalue_seed,  # Seed for p-value computation
        )

        return ChangePointDetector(detector_config)

    def _generate_sequence(self, generator):
        """Generate the graph sequence using model-specific configuration.

        Args:
            generator: GraphGenerator instance

        Returns:
            Dict containing generated graphs and true change points
        """
        model_name = get_full_model_name(self.config["model"]["network"])

        # Get model-specific configuration
        model_config = get_config(model_name)
        return generator.generate_sequence(model_config["params"].__dict__)

    def _extract_features(self, graphs):
        """Extract features from graph sequence.

        Args:
            graphs: List of adjacency matrices

        Returns:
            Tuple containing (numeric_features, raw_features)
        """

        feature_extractor = NetworkFeatureExtractor()
        features_raw = []
        features_numeric = []

        for adj_matrix in graphs:
            graph = adjacency_to_graph(adj_matrix)
            raw_features = feature_extractor.get_features(graph)
            numeric_features = feature_extractor.get_numeric_features(graph)
            features_raw.append(raw_features)
            features_numeric.append(
                [numeric_features[name] for name in self.config["features"]]
            )

        features_array = np.array(features_numeric)
        return features_array, features_raw

    def _run_detection_trials(self, features_normalized, true_change_points):
        """Run multiple trials of the detector.

        Args:
            features_normalized: Normalized feature vectors
            true_change_points: Ground truth change points

        Returns:
            Dict containing individual trial results and aggregated statistics

        Raises:
            RuntimeError: If all detection trials fail
        """
        trials_config = self.config["trials"]
        n_trials = trials_config["n_trials"]
        base_seed = trials_config["random_seeds"]

        # Handle random seeds for all randomized components
        if base_seed is None:
            # Generate completely random seeds
            random_seeds = np.random.randint(0, 2**31 - 1, size=n_trials)
        elif isinstance(base_seed, (int, float)):
            # Generate deterministic sequence of seeds from base seed
            rng = np.random.RandomState(int(base_seed))
            random_seeds = rng.randint(0, 2**31 - 1, size=n_trials)
        else:
            # Use provided list of seeds
            random_seeds = np.array(base_seed)

        logger.info(f"Running {n_trials} detection trials with varying algorithm seeds")
        logger.debug(
            f"Using random seeds: {random_seeds[:min(5, len(random_seeds))]}{' ...' if len(random_seeds) > 5 else ''}"
        )

        # Run individual trials
        individual_results = []
        for trial_idx, seed in enumerate(random_seeds):
            if trial_idx >= n_trials:
                break

            # Convert seed to integer if needed
            int_seed = int(seed) if seed is not None else None

            logger.info(
                f"Running trial {trial_idx + 1}/{n_trials} with seed {int_seed}"
            )

            # For all trials, use the actual data without modifications
            trial_features = features_normalized

            # Create different seeds for various components from the main seed
            # This creates controlled variation in different parts of the algorithm
            detector_seed = int_seed
            strangeness_seed = (int_seed + 1) % (
                2**31 - 1
            )  # Different seed for strangeness calculation
            pvalue_seed = (int_seed + 2) % (
                2**31 - 1
            )  # Different seed for p-value computation
            betting_seed = (int_seed + 3) % (
                2**31 - 1
            )  # Different seed for betting function

            # Initialize detector with this trial's seeds
            detector = self._init_detector(
                random_state=detector_seed,  # Main detector seed
                strangeness_seed=strangeness_seed,  # Seed for strangeness calculation
                pvalue_seed=pvalue_seed,  # Seed for p-value computation
                betting_seed=betting_seed,  # Seed for betting function
            )

            try:
                # Run detection using the original data but varied algorithm seeds
                detection_result = detector.run(
                    data=trial_features,
                    reset_state=True,
                )

                if detection_result is None:
                    logger.warning(
                        f"Trial {trial_idx + 1}/{n_trials} failed: No detection result"
                    )
                    continue

                individual_results.append(detection_result)

                # Log key results from this trial
                if "traditional_change_points" in detection_result:
                    trad_cp = detection_result.get("traditional_change_points", [])
                    logger.debug(
                        f"Trial {trial_idx + 1} results: Traditional CPs: {trad_cp}"
                    )

            except Exception as e:
                logger.error(f"Trial {trial_idx + 1}/{n_trials} failed: {str(e)}")
                continue

        if not individual_results:
            raise RuntimeError("All detection trials failed")

        logger.info(
            f"Completed {len(individual_results)}/{n_trials} trials successfully"
        )

        # Use the first trial's results for visualization
        aggregated_results = individual_results[0].copy()

        # Return combined results
        return {
            "individual_trials": individual_results,
            "aggregated": aggregated_results,
            "random_seeds": random_seeds.tolist(),
        }

    def _export_results_to_csv(self, trial_results, true_change_points):
        """Export detection results to CSV files.

        Args:
            trial_results: Dict containing detection trial results
            true_change_points: List of ground truth change points
        """
        try:
            csv_output_dir = os.path.join(self.config["output"]["directory"])
            output_manager = OutputManager(csv_output_dir, self.config)
            output_manager.export_to_csv(
                trial_results["aggregated"],
                true_change_points,
                individual_trials=trial_results["individual_trials"],
            )

        except Exception as e:
            logger.error(f"Failed to export results to CSV: {str(e)}")

    def run(self, save_csv=None):
        """Run the complete detection pipeline.

        Args:
            save_csv: Whether to save results to CSV files.
                      If None, uses config value.

        Returns:
            Dictionary containing all results

        Raises:
            RuntimeError: If the pipeline execution fails
        """
        # Use provided parameters or fall back to config values
        enable_csv_export = (
            save_csv
            if save_csv is not None
            else self.config["execution"].get("save_csv", True)
        )

        try:
            # Create output directory with descriptive name
            logger.info("STEP 1: Setting up output directory")
            self._setup_output_directory()

            # Run each pipeline stage
            logger.info("STEP 2: Generating graph sequence")
            generator = self._init_generator()
            sequence_result = self._generate_sequence(generator)
            graphs = sequence_result["graphs"]
            true_change_points = sequence_result["change_points"]
            logger.info(
                f"Generated sequence with {len(graphs)} graphs and {len(true_change_points)} change points at: {[int(cp) for cp in true_change_points]}"
            )

            logger.info("STEP 3: Extracting features")
            features_numeric, features_raw = self._extract_features(graphs)
            logger.info(
                f"Extracted {features_numeric.shape[1]} features across {features_numeric.shape[0]} timesteps"
            )

            # Normalize features ONCE before all trials
            logger.info("STEP 4: Normalizing features")
            features_normalized, feature_means, feature_stds = normalize_features(
                features_numeric
            )
            logger.debug(
                f"Feature means: {feature_means[:3]}{' ...' if len(feature_means) > 3 else ''}"
            )
            logger.debug(
                f"Feature std devs: {feature_stds[:3]}{' ...' if len(feature_stds) > 3 else ''}"
            )

            # Run detection trials
            logger.info("STEP 5: Running detection trials")
            trial_results = self._run_detection_trials(
                features_normalized, true_change_points
            )

            # Optional CSV export
            if enable_csv_export and trial_results["aggregated"]:
                logger.info("STEP 7: Exporting results to CSV")
                self._export_results_to_csv(trial_results, true_change_points)
            else:
                logger.info("STEP 7: CSV export disabled, skipping")

            # Compile final results
            logger.info("STEP 8: Preparing result data")
            results = prepare_result_data(
                sequence_result,
                features_numeric,
                features_raw,
                trial_results,
                self.config,
            )

            logger.info("Pipeline execution completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
