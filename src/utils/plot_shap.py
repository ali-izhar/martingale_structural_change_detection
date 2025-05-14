# src/utils/plot_shap.py

"""SHAP (SHapley Additive exPlanations) analysis for anomaly detection."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
from sklearn.linear_model import LinearRegression

# Feature display names for better visualization
FEATURE_NAMES = {
    "0": "Degree",
    "1": "Density",
    "2": "Clustering",
    "3": "Betweenness",
    "4": "Eigenvector",
    "5": "Closeness",
    "6": "Spectral",
    "7": "Laplacian",
}


def setup_plot_style():
    """Set up consistent plotting style for research visualizations."""
    plt.style.use("seaborn-v0_8-paper")
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "figure.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (12, 7),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
            "lines.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_data(file_path, sheet_name="Trial1"):
    """Load data from an Excel file. Returns a tuple of (DataFrame with data, list of change points, detection details DataFrame)."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully read sheet: {sheet_name} from {file_path}")

        # Extract change points from true_change_point column
        change_points = []
        if "true_change_point" in df.columns:
            # Get timesteps where true_change_point is not 0
            change_points = df.loc[
                df["true_change_point"] != 0, "timestep"
            ].values.tolist()
            print(f"Found {len(change_points)} change points in sheet")

        # Try to load detection details
        detection_details = None
        try:
            detection_df = pd.read_excel(file_path, sheet_name="Detection Details")

            # Filter for the current trial
            trial_num = int(sheet_name.replace("Trial", ""))
            detection_details = detection_df[
                (detection_df["Trial"] == trial_num)
                & (detection_df["Type"] == "Traditional")
            ]

            if not detection_details.empty:
                print(
                    f"Found {len(detection_details)} detection details for {sheet_name}"
                )
        except Exception as e:
            print(f"Warning: Could not read Detection Details: {str(e)}")
            detection_details = None

        return df, change_points, detection_details

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, [], None


def load_trial_data(file_path):
    """Load data from all trial sheets in an Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        tuple: (list of DataFrames for trials, list of change points, detection details DataFrame)
    """
    try:
        excel = pd.ExcelFile(file_path)
        trial_sheets = [
            sheet for sheet in excel.sheet_names if sheet.startswith("Trial")
        ]
        print(f"Found {len(trial_sheets)} trial sheets: {trial_sheets}")

        # Try to load detection details
        detection_details = None
        try:
            detection_details = pd.read_excel(file_path, sheet_name="Detection Details")
            detection_details = detection_details[
                detection_details["Type"] == "Traditional"
            ]
            print(f"Found {len(detection_details)} traditional detection details")
        except Exception as e:
            print(f"Warning: Could not read Detection Details: {str(e)}")

        # Load each trial sheet
        trial_dfs = []
        all_change_points = []

        for sheet in trial_sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            trial_dfs.append(df)

            # Extract change points from true_change_point column
            if "true_change_point" in df.columns:
                # Get timesteps where true_change_point is not 0
                sheet_change_points = df.loc[
                    df["true_change_point"] != 0, "timestep"
                ].values.tolist()
                all_change_points.extend(sheet_change_points)

        # Remove duplicates and sort
        change_points = sorted(set(all_change_points))
        print(
            f"Found {len(change_points)} unique change points across all trial sheets"
        )

        return trial_dfs, change_points, detection_details

    except Exception as e:
        print(f"Error loading trial data: {str(e)}")
        return [], [], None


def get_feature_cols(df, prefix="individual_traditional_martingales_feature"):
    """Extract feature column names from DataFrame based on prefix.

    Args:
        df: DataFrame containing the data
        prefix: Prefix of feature columns

    Returns:
        list: Feature column names
    """
    feature_cols = [col for col in df.columns if col.startswith(prefix)]

    # Sort by feature number
    feature_cols.sort(key=lambda x: int(x.split(prefix)[1]))

    return feature_cols


def get_display_names(feature_cols):
    """Convert feature column names to human-readable display names.

    Args:
        feature_cols: List of feature column names

    Returns:
        list: Display names for features
    """
    display_names = []
    for col in feature_cols:
        # Extract feature ID
        feature_id = col.split("feature")[1]
        # Get display name from mapping or use default
        display_name = FEATURE_NAMES.get(feature_id, f"Feature {feature_id}")
        display_names.append(display_name)

    return display_names


def compute_shap_values(
    df,
    feature_cols,
    sum_col,
    threshold=50.0,
    timesteps=None,
    detection_details=None,
    current_trial=None,
):
    """Compute SHAP values for feature importance analysis.

    Args:
        df: DataFrame containing martingale data
        feature_cols: List of column names for individual feature martingales
        sum_col: Column name for sum martingale
        threshold: Detection threshold value
        timesteps: Optional array of timesteps
        detection_details: DataFrame with detection information
        current_trial: Current trial number

    Returns:
        tuple: (SHAP values, feature contributions at detection points)
    """
    if timesteps is None:
        timesteps = np.arange(len(df))

    # Create feature matrix
    X = df[feature_cols].copy()

    # Handle NaN values by filling with zeros
    X = X.fillna(0)
    y = df[sum_col].fillna(0)

    # Check for any remaining NaN values
    if X.isna().any().any() or y.isna().any():
        print("Warning: After filling, some NaN values remain. These will be dropped.")
        # Drop rows with any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        print(f"Dropped {len(df) - len(X)} rows with NaN values")

    print(f"Shape of feature matrix after handling NaNs: {X.shape}")

    # Fit linear model to approximate the sum martingale
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Verify model accuracy
    predictions = model.predict(X)
    r2 = np.corrcoef(predictions, y)[0, 1] ** 2
    print(f"Linear model R² score: {r2:.6f}")

    # Compute SHAP values
    try:
        # Sample background data for the explainer
        background_indices = np.random.choice(
            len(X), size=min(100, len(X)), replace=False
        )
        background = X.iloc[background_indices]

        # Create explainer and compute SHAP values
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

        print("Successfully computed SHAP values using KernelExplainer")
    except Exception as e:
        print(f"Error computing SHAP values with KernelExplainer: {e}")
        print("Using feature values * coefficients as SHAP approximation")

        # Approximate SHAP values for linear model
        shap_values = np.zeros(X.shape)
        for i, col in enumerate(X.columns):
            shap_values[:, i] = X[col].values * model.coef_[i]

    # Find detection points from Detection Details or threshold crossings
    detection_indices = []

    # First try to use Detection Details if available
    if detection_details is not None and not detection_details.empty:
        trial_details = detection_details
        if current_trial is not None:
            trial_details = detection_details[
                detection_details["Trial"] == current_trial
            ]

        for _, row in trial_details.iterrows():
            detection_idx = row["Detection Index"]
            if detection_idx in df["timestep"].values:
                idx = df.index[df["timestep"] == detection_idx][0]
                detection_indices.append(idx)

    # If no Detection Details, check for threshold crossings
    if not detection_indices:
        for i in range(1, len(df)):
            if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
                detection_indices.append(i)

    # If still no detection points, find peaks after change points
    if not detection_indices:
        change_points = []
        if "true_change_point" in df.columns:
            change_points = df.loc[
                df["true_change_point"] != 0, "timestep"
            ].values.tolist()

        if change_points:
            for cp in change_points:
                # Find closest index to change point
                cp_idx = np.argmin(np.abs(timesteps - cp))

                # Define window after change point
                window_start = cp_idx
                window_end = min(len(df), cp_idx + 10)

                # Find index of maximum sum martingale
                if window_start < window_end:
                    max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                    detection_indices.append(max_idx)

    # Compute contributions at detection points
    contributions = []

    for idx in detection_indices:
        if 0 <= idx < len(df):
            detection_time = timesteps[idx]

            # Calculate feature values and contributions
            feature_values = df[feature_cols].iloc[idx].fillna(0).values
            total = sum(feature_values)

            if total > 0:
                contrib_data = {
                    "Feature": get_display_names(feature_cols),
                    "Martingale Value": feature_values,
                    "Contribution %": (feature_values / total) * 100,
                    "Detection Point": detection_time,
                }

                # Create DataFrame and sort by contribution
                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values("Contribution %", ascending=False)

                contributions.append(contrib_df)

                print(f"\nFeature contributions at detection point {detection_time}:")
                print(
                    f"{'Feature':<15} {'Martingale Value':<15} {'Contribution %':<15}"
                )
                print("-" * 50)

                for _, row in contrib_df.iterrows():
                    print(
                        f"{row['Feature']:<15} {row['Martingale Value']:<15.6f} {row['Contribution %']:<15.2f}"
                    )

    if contributions:
        all_contributions = pd.concat(contributions)
        return shap_values, all_contributions
    else:
        return shap_values, pd.DataFrame()


def plot_shap_summary(shap_values, X, feature_names, output_path):
    """Create SHAP summary plot to visualize feature importance.

    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix used for SHAP analysis
        feature_names: List of feature display names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {output_path}")


def plot_shap_waterfall(shap_values, X, feature_names, output_path, sample_idx=None):
    """Create SHAP waterfall plot for a specific sample.

    Args:
        shap_values: SHAP values from compute_shap_values
        X: Feature matrix used for SHAP analysis
        feature_names: List of feature display names
        output_path: Path to save the plot
        sample_idx: Index of sample to explain (defaults to highest sum of SHAP values)
    """
    if sample_idx is None:
        # Pick sample with highest sum of absolute SHAP values
        sample_idx = np.argmax(np.sum(np.abs(shap_values), axis=1))

    plt.figure(figsize=(10, 6))

    try:
        # Get expected value (base value)
        expected_value = np.mean(X.sum(axis=1).values)

        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=expected_value,
                data=X.iloc[sample_idx].values,
                feature_names=feature_names,
            ),
            show=False,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP waterfall plot for sample {sample_idx} to {output_path}")
    except Exception as e:
        print(f"Error creating waterfall plot: {e}")
        plt.close()


def plot_shap_over_time(
    shap_values,
    df,
    feature_cols,
    sum_col,
    change_points,
    threshold,
    output_path,
    detection_details=None,
    current_trial=None,
):
    """Create visualization of SHAP values over time with annotated change points.

    Args:
        shap_values: SHAP values for traditional martingales
        df: DataFrame containing martingale data
        feature_cols: List of column names for individual feature martingales
        sum_col: Column name for traditional sum martingale
        change_points: List of change points
        threshold: Detection threshold
        output_path: Path to save the plot
        detection_details: DataFrame with detection information
        current_trial: Current trial number
    """
    # Get display names for features
    feature_names = get_display_names(feature_cols)

    # Get timesteps
    timesteps = (
        df["timestep"].values if "timestep" in df.columns else np.arange(len(df))
    )

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_cols)))

    # Create figure with 2 panels (instead of 4)
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.1,
        },
    )

    # Panel 1: Traditional SHAP Values (Top)
    for i, col in enumerate(feature_cols):
        axs[0].plot(
            timesteps,
            shap_values[:, i],
            label=feature_names[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points
    if change_points:
        for cp in change_points:
            axs[0].axvline(x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.5)

    # Set titles and labels
    axs[0].set_title("Traditional Feature Contributions (SHAP Values)", fontsize=16)
    axs[0].set_ylabel("SHAP Value", fontsize=14)

    # Add legend
    axs[0].legend(loc="upper right", fontsize=11)

    # Add grid
    axs[0].grid(True, alpha=0.3)

    # Calculate feature contributions
    contributions = np.zeros((len(df), len(feature_cols)))
    detection_indices = []

    # Find detection points from Detection Details if available
    if detection_details is not None and not detection_details.empty:
        trial_details = detection_details
        if current_trial is not None:
            trial_details = detection_details[
                detection_details["Trial"] == current_trial
            ]

        for _, row in trial_details.iterrows():
            detection_idx = row["Detection Index"]
            if detection_idx in df["timestep"].values:
                idx = df.index[df["timestep"] == detection_idx][0]
                detection_indices.append(idx)
    # Otherwise find threshold crossings
    else:
        # Find detection points (threshold crossings)
        for i in range(1, len(df)):
            if df[sum_col].iloc[i - 1] <= threshold and df[sum_col].iloc[i] > threshold:
                detection_indices.append(i)

    # If no threshold crossings, find peaks after change points
    if not detection_indices and change_points:
        for cp in change_points:
            cp_idx = np.argmin(np.abs(timesteps - cp))
            window_start = cp_idx
            window_end = min(len(df), cp_idx + 10)

            if window_start < window_end:
                max_idx = df[sum_col].iloc[window_start:window_end].idxmax()
                detection_indices.append(max_idx)

    # Process detection points
    if detection_indices:
        for detection_idx in detection_indices:
            feature_values = df[feature_cols].iloc[detection_idx].values
            total = sum(feature_values)

            if total > 0:
                # Contributions as percentages
                for j, val in enumerate(feature_values):
                    contributions[detection_idx, j] = val / total

                # Add decaying contributions around detection point
                window = 2
                for i in range(
                    max(0, detection_idx - window),
                    min(len(df), detection_idx + window + 1),
                ):
                    if i != detection_idx:
                        decay = 0.2 ** abs(i - detection_idx)
                        vals = df[feature_cols].iloc[i].values
                        val_sum = sum(vals)
                        if val_sum > 0:
                            for j, val in enumerate(vals):
                                contributions[i, j] = (val / val_sum) * decay

    # Create legend labels with percentages
    feature_percentages = {}
    detection_timestamps = []

    if detection_indices and len(detection_indices) > 0:
        for i, idx in enumerate(detection_indices):
            if 0 <= idx < len(timesteps):
                detection_timestamps.append(timesteps[idx])
                for j, col in enumerate(feature_cols):
                    if j not in feature_percentages:
                        feature_percentages[j] = []
                    feature_percentages[j].append(contributions[idx, j] * 100)

    # Format legend labels
    legend_labels = []
    for i, name in enumerate(feature_names):
        if i in feature_percentages and len(feature_percentages[i]) > 0:
            if len(feature_percentages[i]) == 1:
                pct_str = f"{feature_percentages[i][0]:.1f}%"
            elif len(feature_percentages[i]) == 2:
                pct_str = f"{feature_percentages[i][0]:.1f}%, {feature_percentages[i][1]:.1f}%"
            else:
                pct_str = ", ".join([f"{pct:.1f}%" for pct in feature_percentages[i]])

            legend_labels.append(f"{name} ({pct_str})")
        else:
            legend_labels.append(name)

    # Plot feature contributions (Bottom panel)
    for i, col in enumerate(feature_cols):
        axs[1].plot(
            timesteps,
            contributions[:, i],
            label=legend_labels[i],
            color=colors[i],
            alpha=0.7,
            linewidth=1.2,
        )

    # Mark change points
    if change_points:
        for cp in change_points:
            axs[1].axvline(x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.5)

            # Add change point label
            axs[1].annotate(
                f"CP {cp}",
                xy=(cp, 0),
                xytext=(cp, -0.05),
                color="gray",
                fontweight="bold",
                fontsize=10,
                ha="center",
                va="top",
            )

    # Mark detection points
    for idx in detection_indices:
        if 0 <= idx < len(timesteps):
            dp = timesteps[idx]
            axs[1].axvline(
                x=dp, color="purple", linestyle=":", alpha=0.8, linewidth=1.5
            )

            # If we have detection details, add more information
            if detection_details is not None and not detection_details.empty:
                trial_details = detection_details
                if current_trial is not None:
                    trial_details = detection_details[
                        detection_details["Trial"] == current_trial
                    ]

                for _, row in trial_details.iterrows():
                    if row["Detection Index"] == dp:
                        nearest_cp = row["Nearest True CP"]
                        distance = row["Distance to CP"]
                        axs[1].annotate(
                            f"Detection (CP={nearest_cp}, d={distance})",
                            xy=(dp, 0.5),
                            xytext=(dp, 0.6),
                            color="purple",
                            fontweight="bold",
                            fontsize=10,
                            ha="center",
                            arrowprops=dict(arrowstyle="->", color="purple"),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        )
                        break
            else:
                # Simple detection label
                axs[1].annotate(
                    "Detection",
                    xy=(dp, 0.5),
                    xytext=(dp, 0.6),
                    color="purple",
                    fontweight="bold",
                    fontsize=10,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="purple"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    # Set titles and labels for bottom panel
    axs[1].set_title("Traditional Feature Contributions", fontsize=16)
    axs[1].set_xlabel("Timestep", fontsize=14)
    axs[1].set_ylabel("Feature Contribution (0-1)", fontsize=14)

    # Create custom legend with more space
    axs[1].legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        handlelength=1,
        handleheight=1.5,
        labelspacing=0.4,
    )

    # Add grid to bottom panel
    axs[1].grid(True, alpha=0.3)

    # Set common x-axis limits
    x_min, x_max = min(timesteps), max(timesteps)
    x_limits = (x_min, x_max + int(0.05 * (x_max - x_min)))

    # Reduce the number of x-axis ticks to avoid crowding
    max_ticks = 10  # Maximum number of ticks to show
    step = max(1, int((x_max - x_min) / max_ticks))
    ticks = list(range(int(x_min), int(x_max) + 1, step))

    for ax in axs:
        ax.set_xlim(x_limits)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(tick)) for tick in ticks])

    # Add a title with the trial number if available
    if current_trial is not None:
        plt.suptitle(
            f"Trial {current_trial} - Feature Contributions Analysis", fontsize=18
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP analysis plot to {output_path}")

    # Generate feature contribution report
    contributions_list = []

    if detection_indices:
        for idx in detection_indices:
            if 0 <= idx < len(df):
                detection_time = timesteps[idx]
                contrib_data = {
                    "Feature": feature_names,
                    "Martingale Value": df[feature_cols].iloc[idx].values,
                    "Contribution %": contributions[idx] * 100,
                    "Detection Point": detection_time,
                }
                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values("Contribution %", ascending=False)
                contributions_list.append(contrib_df)

    if contributions_list:
        return pd.concat(contributions_list)
    else:
        return pd.DataFrame()


def plot_feature_contributions(contributions_df, output_path):
    """Create horizontal bar chart of feature contributions.

    Args:
        contributions_df: DataFrame with feature contributions
        output_path: Path to save the plot
    """
    if contributions_df.empty:
        print("No feature contributions to plot")
        return

    # Group by feature and compute average contribution
    avg_contrib = (
        contributions_df.groupby("Feature")["Contribution %"].mean().reset_index()
    )
    avg_contrib = avg_contrib.sort_values(
        "Contribution %", ascending=True
    )  # Ascending for horizontal bars

    # Create colors based on contribution percentage
    colors = plt.cm.YlOrRd(avg_contrib["Contribution %"] / 100)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(avg_contrib["Feature"], avg_contrib["Contribution %"], color=colors)

    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center"
        )

    plt.xlabel("Average Contribution %")
    plt.title("Feature Contributions to Anomaly Detection")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature contributions plot to {output_path}")


def analyze_shap(
    file_path, sheet_name="Trial1", output_dir="results/shap", threshold=50.0
):
    """Main function to analyze SHAP values and create visualizations.

    Args:
        file_path: Path to Excel file
        sheet_name: Name of sheet to analyze
        output_dir: Directory to save output plots
        threshold: Detection threshold
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # Get current trial number
    current_trial = None
    if sheet_name.startswith("Trial"):
        try:
            current_trial = int(sheet_name.replace("Trial", ""))
        except ValueError:
            pass

    df, change_points, detection_details = load_data(file_path, sheet_name)

    if df is None:
        return

    # Get feature columns
    feature_cols = get_feature_cols(
        df, prefix="individual_traditional_martingales_feature"
    )

    if not feature_cols:
        print("No feature columns found in the data")
        return

    # Get display names
    feature_names = get_display_names(feature_cols)

    # Determine sum column name
    sum_col = "traditional_sum_martingales"

    if sum_col not in df.columns:
        print("Traditional sum martingale column not found in the data")
        return

    print(f"Analyzing {len(feature_cols)} features with {sum_col}")

    # Use provided threshold
    print(f"Using threshold: {threshold}")

    # Compute SHAP values
    shap_values, contributions_df = compute_shap_values(
        df=df,
        feature_cols=feature_cols,
        sum_col=sum_col,
        threshold=threshold,
        detection_details=detection_details,
        current_trial=current_trial,
    )

    # Create SHAP summary plot
    plot_shap_summary(
        shap_values=shap_values,
        X=df[feature_cols],
        feature_names=feature_names,
        output_path=os.path.join(output_dir, f"{sheet_name}_shap_summary.png"),
    )

    # Create SHAP over time plot
    contributions_df = plot_shap_over_time(
        shap_values=shap_values,
        df=df,
        feature_cols=feature_cols,
        sum_col=sum_col,
        change_points=change_points,
        threshold=threshold,
        output_path=os.path.join(output_dir, f"{sheet_name}_shap_analysis.png"),
        detection_details=detection_details,
        current_trial=current_trial,
    )

    # Create feature contributions plot
    if not contributions_df.empty:
        plot_feature_contributions(
            contributions_df=contributions_df,
            output_path=os.path.join(
                output_dir, f"{sheet_name}_feature_contributions.png"
            ),
        )

        # Save contributions to CSV
        contributions_df.to_csv(
            os.path.join(output_dir, f"{sheet_name}_feature_contributions.csv"),
            index=False,
        )
        print(
            f"Saved feature contributions to {os.path.join(output_dir, f'{sheet_name}_feature_contributions.csv')}"
        )

    print(f"SHAP analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SHAP values for anomaly detection"
    )
    parser.add_argument(
        "--file_path", "-f", type=str, required=True, help="Path to Excel file"
    )
    parser.add_argument(
        "--sheet_name", "-s", type=str, default="Trial1", help="Sheet name"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="results/shap", help="Output directory"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=50.0, help="Detection threshold"
    )

    args = parser.parse_args()

    analyze_shap(
        file_path=args.file_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
