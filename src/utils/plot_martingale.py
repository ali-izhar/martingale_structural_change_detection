# src/utils/plot_martingale.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


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
            "figure.titlesize": 18,
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


# Feature display names mapping
FEATURE_INFO = {
    "0": "Degree",
    "1": "Density",
    "2": "Clustering",
    "3": "Betweenness",
    "4": "Eigenvector",
    "5": "Closeness",
    "6": "Spectral",
    "7": "Laplacian",
}


def load_data(file_path, sheet_name="Trial1"):
    """Load martingale data from an Excel file. Return
    a dataframe with the martingale data and a list of change points."""
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

        return df, change_points, detection_details

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, [], None


def load_trial_data(file_path):
    """Load data from all trial sheets.

    Returns:
        trial_dfs: List of dataframes for each trial
        change_points: List of change points
        detection_details: DataFrame with detection information
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


def plot_individual_martingales(
    df,
    output_path,
    change_points=None,
    trial_dfs=None,
    detection_details=None,
    threshold=50.0,
    current_trial=None,
):
    """Create a grid of plots for individual feature martingales.

    Args:
        df: DataFrame containing martingale data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        trial_dfs: List of DataFrames for individual trials (for box plots)
        detection_details: DataFrame with detection information
        threshold: Detection threshold value
        current_trial: Current trial number
    """
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("individual_traditional_martingales_feature")
    ]

    if not feature_cols:
        print("No individual feature martingale columns found")
        return

    # Calculate grid dimensions
    n_features = len(feature_cols)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharex=True)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    x = df["timestep"].values

    # Determine full data range for consistent x-axis limits
    x_min, x_max = min(x), max(x)

    # Explicitly set the x-axis limits to show just a little beyond the max value
    x_limits = (x_min, x_max + int(0.05 * (x_max - x_min)))

    # Find global maximum for consistent y-axis limits
    y_max = 0
    for col in feature_cols:
        col_max = df[col].max()
        if col_max > y_max:
            y_max = col_max

    # Round up to nearest threshold
    y_max = ((y_max // threshold) + 1) * threshold

    # Find max feature martingale values at change points to determine importance
    feature_importance = {}
    if change_points and trial_dfs:
        for feature_id in range(len(feature_cols)):
            trad_col = f"individual_traditional_martingales_feature{feature_id}"
            max_val = 0

            for cp in change_points:
                # Look at window after change point
                for trial_df in trial_dfs:
                    cp_idx = np.argmin(np.abs(trial_df["timestep"].values - cp))
                    window_end = min(len(trial_df), cp_idx + 10)
                    if cp_idx < len(trial_df):
                        window = trial_df.iloc[cp_idx:window_end]
                        if trad_col in window.columns:
                            max_val = max(max_val, window[trad_col].max())

            feature_importance[feature_id] = max_val

    # If we have trial data, prepare for box plots
    has_trial_data = trial_dfs is not None and len(trial_dfs) > 0

    # Sample points for box plots (plotting at every timestep would be too crowded)
    if has_trial_data:
        # Find important points around change points and sample regularly elsewhere
        sample_points = []
        if change_points:
            window = 5  # Points before and after change points
            for cp in change_points:
                for i in range(max(0, cp - window), min(len(x), cp + window + 1)):
                    if i in x:
                        sample_points.append(i)

        # Add regular samples (about 10-15 points total)
        if len(x) > 0:
            step = max(1, len(x) // 10)
            for i in range(0, len(x), step):
                sample_points.append(x[i])

        # Remove duplicates and sort
        sample_points = sorted(set(sample_points))

    # Plot each feature
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            ax = axes[i]

            # Extract feature ID and get name
            feature_id = col.split("feature")[1]
            feature_name = FEATURE_INFO.get(feature_id, f"Feature {feature_id}")

            # Determine if this is an important feature
            is_important = False
            if feature_importance and int(feature_id) in feature_importance:
                importance_val = feature_importance[int(feature_id)]
                # Consider important if max value is at least 30% of the threshold
                is_important = importance_val > 15

            # Use background shading for important features
            if is_important:
                ax.set_facecolor(
                    "#f8f8ff"
                )  # Very light blue background for important features
                title_fontweight = "bold"
            else:
                title_fontweight = "normal"

            if has_trial_data:
                # Traditional martingales
                trad_col_base = (
                    f"individual_traditional_martingales_feature{feature_id}"
                )

                # Create box plot data for each sample point
                trad_data = []
                trad_positions = []

                for time_point in sample_points:
                    # Collect values across trials for this time point
                    trad_values = []

                    for trial_df in trial_dfs:
                        if "timestep" in trial_df.columns:
                            # Find exact or closest timestep
                            if time_point in trial_df["timestep"].values:
                                idx = trial_df.index[
                                    trial_df["timestep"] == time_point
                                ][0]
                                if trad_col_base in trial_df.columns:
                                    trad_values.append(trial_df.loc[idx, trad_col_base])

                    if trad_values:
                        trad_data.append(trad_values)
                        trad_positions.append(time_point)

                # Create box plots with narrower width for better visibility
                box_width = min(3.0, threshold / len(sample_points))

                if trad_data:
                    trad_boxes = ax.boxplot(
                        trad_data,
                        positions=trad_positions,
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor="#ADD8E6", color="#0000CD", alpha=0.7
                        ),  # Improved colors
                        whiskerprops=dict(color="#0000CD", linewidth=1.0),
                        medianprops=dict(color="#00008B", linewidth=1.5),
                        showfliers=False,
                        zorder=3,
                    )

                # Also plot the current trial data as a line
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=1.0, alpha=0.5, zorder=1
                )
            else:
                # Regular line plot (original behavior)
                ax.plot(
                    x, df[col].values, "#0000CD", linewidth=1.5, label="Traditional"
                )

            # Mark change points if provided
            if change_points:
                for cp in change_points:
                    # Add subtle background shading to highlight change points
                    ax.axvspan(cp - 1, cp + 1, color="gray", alpha=0.15, zorder=0)
                    ax.axvline(
                        x=cp, color="gray", linestyle="--", alpha=0.8, linewidth=1.2
                    )

            # Set title and labels
            ax.set_title(
                feature_name,
                fontsize=13,
                fontweight=title_fontweight,
                color="#444444" if not is_important else "#000066",
            )
            if i % n_cols == 0:
                ax.set_ylabel("Martingale Value", fontsize=12)
            if i >= n_features - n_cols:
                ax.set_xlabel("Time", fontsize=12)

            ax.set_ylim(0, y_max)
            ax.set_yticks(range(0, int(y_max) + 1, int(threshold)))
            ax.set_xlim(x_limits)

            # Reduce the number of x-ticks to avoid crowding
            max_ticks = 5  # Maximum number of ticks to show
            step = max(1, int((x_max - x_min) / max_ticks))
            ticks = list(range(int(x_min), int(x_max) + 1, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(tick)) for tick in ticks])

            # Add legend on first plot only
            if i == 0:
                if has_trial_data:
                    from matplotlib.patches import Patch

                    legend_elements = [
                        Patch(
                            facecolor="#ADD8E6",
                            edgecolor="#0000CD",
                            alpha=0.7,
                            label="Traditional",
                        ),
                    ]
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
                else:
                    ax.legend(loc="upper right", fontsize=10)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.95)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved individual martingales plot to {output_path}")


def plot_sum_martingales(
    df,
    output_path,
    change_points=None,
    threshold=50.0,
    trial_dfs=None,
    detection_details=None,
    current_trial=None,
):
    """Create a plot of sum martingales.

    Args:
        df: DataFrame containing martingale data
        output_path: Path to save the output image
        change_points: List of change points to mark on plots
        threshold: Detection threshold value
        trial_dfs: List of DataFrames for individual trials (for box plots)
        detection_details: DataFrame with detection information
        current_trial: Current trial number
    """
    # Find column name for sum martingales
    trad_sum_col = "traditional_sum_martingales"

    if trad_sum_col not in df.columns:
        print("Traditional sum martingale column not found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["timestep"].values

    # Set proper x-axis limits
    x_min, x_max = min(x), max(x)
    x_limits = (x_min, x_max + int(0.05 * (x_max - x_min)))

    # Check if we have trial data for box plots
    has_trial_data = trial_dfs is not None and len(trial_dfs) > 0

    # If we have change points, add subtle background highlighting
    if change_points:
        for cp in change_points:
            # Add light shading after change point to highlight detection region
            ax.axvspan(cp, min(cp + 10, x_max), color="#f5f5f5", zorder=0)

    if has_trial_data:
        # Sample points for box plots (plotting at every timestep would be too crowded)
        sample_points = []
        if change_points:
            window = 5  # Points before and after change points
            for cp in change_points:
                for i in range(max(0, cp - window), min(max(x) + 1, cp + window + 1)):
                    if i in x:
                        sample_points.append(i)

        # Add regular samples (about 10-15 points total)
        if len(x) > 0:
            step = max(1, len(x) // 15)
            regular_points = list(range(int(min(x)), int(max(x)) + 1, step))
            sample_points.extend(regular_points)

        # Remove duplicates and sort
        sample_points = sorted(set(sample_points))

        # Create box plot data for each sample point
        trad_data = []
        trad_positions = []

        for time_point in sample_points:
            # Collect values across trials for this time point
            trad_values = []

            for trial_df in trial_dfs:
                if "timestep" in trial_df.columns:
                    # Find rows with this timestep
                    matches = trial_df[trial_df["timestep"] == time_point]
                    if not matches.empty:
                        if trad_sum_col in trial_df.columns:
                            trad_values.append(matches[trad_sum_col].values[0])

            if trad_values:
                trad_data.append(trad_values)
                trad_positions.append(time_point)

        # Create box plots with improved styling
        box_width = min(3.0, 60 / len(sample_points))

        if trad_data:
            trad_boxes = ax.boxplot(
                trad_data,
                positions=trad_positions,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor="#ADD8E6", color="#0000CD", alpha=0.7),
                whiskerprops=dict(color="#0000CD", linewidth=1.0),
                medianprops=dict(color="#00008B", linewidth=1.5),
                showfliers=False,
                zorder=3,
            )

        # Add thin lines for current trial data
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )

        # Add legend with improved appearance
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="#ADD8E6", edgecolor="#0000CD", alpha=0.7, label="Traditional"
            ),
            Patch(facecolor="red", edgecolor="red", alpha=0.5, label="Threshold"),
            Patch(facecolor="gray", edgecolor="gray", alpha=0.5, label="Change Points"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    else:
        # Standard line plots (original behavior)
        ax.plot(
            x,
            df[trad_sum_col].values,
            "#0000CD",
            linewidth=2.0,
            label="Traditional Sum",
            zorder=5,
        )

        # Add legend
        ax.legend(loc="upper right", fontsize=12)

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label="Threshold",
        alpha=0.8,
        linewidth=1.5,
    )

    # Add detection points from Detection Details sheet
    if detection_details is not None and not detection_details.empty:
        # Filter for current trial if specified
        trial_details = detection_details
        if current_trial is not None:
            trial_details = detection_details[
                detection_details["Trial"] == current_trial
            ]

        for _, row in trial_details.iterrows():
            detection_idx = row["Detection Index"]
            nearest_cp = row["Nearest True CP"]
            distance = row["Distance to CP"]
            is_within_10 = row["Is Within 10 Steps"]

            # Check if this detection index is in our data range
            if detection_idx in df["timestep"].values:
                # Get value at detection point
                detection_value = df.loc[
                    df["timestep"] == detection_idx, trad_sum_col
                ].values[0]

                # Choose color based on detection quality
                marker_color = "#0000CD" if is_within_10 else "#FF4500"

                # Add detection marker
                ax.scatter(
                    [detection_idx],
                    [detection_value],
                    color=marker_color,
                    s=80,
                    zorder=10,
                    marker="o",
                )

                # Add annotation - position it much lower to avoid overlapping with peaks
                vertical_offset = threshold * 0.5  # Use 50% of threshold as offset
                ax.annotate(
                    f"Detection (CP={nearest_cp}, d={distance})",
                    xy=(detection_idx, detection_value),
                    xytext=(detection_idx, vertical_offset),
                    color="#00008B" if is_within_10 else "#FF4500",
                    fontweight="bold",
                    fontsize=10,
                    arrowprops=dict(
                        arrowstyle="->", color="#00008B" if is_within_10 else "#FF4500"
                    ),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    ha="center",
                )
    # Add regular detection markers if no Detection Details but traditional_detected is available
    elif "traditional_detected" in df.columns:
        detection_points = df.loc[df["traditional_detected"] == 1, "timestep"].values
        if len(detection_points) > 0:
            for detection_point in detection_points:
                detection_value = df.loc[
                    df["timestep"] == detection_point, trad_sum_col
                ].values[0]
                ax.scatter(
                    [detection_point],
                    [detection_value],
                    color="#0000CD",
                    s=80,
                    zorder=10,
                    marker="o",
                )

                # Position annotation lower with arrow pointing up
                vertical_offset = threshold * 0.5
                ax.annotate(
                    "Detection",
                    xy=(detection_point, detection_value),
                    xytext=(detection_point, vertical_offset),
                    color="#00008B",
                    fontweight="bold",
                    fontsize=11,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="#00008B"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )

    # Mark change points with cleaner labels
    if change_points:
        # Calculate appropriate label positions
        label_y_pos = -threshold * 0.05

        # Get change point groups to prevent label overlap
        cp_groups = []
        current_group = []
        min_distance = (
            15  # Minimum distance between change points before creating new group
        )

        for cp in sorted(change_points):
            if not current_group or (cp - current_group[-1]) < min_distance:
                current_group.append(cp)
            else:
                cp_groups.append(current_group)
                current_group = [cp]

        if current_group:
            cp_groups.append(current_group)

        # Add vertical lines and labels for change points
        for i, group in enumerate(cp_groups):
            for j, cp in enumerate(group):
                ax.axvline(
                    x=cp,
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label="Change Point" if i == 0 and j == 0 else "",
                )

                # For groups, stagger the labels vertically
                if len(group) > 1:
                    # Stagger labels so they don't overlap
                    offset = j * 0.05 * threshold
                    this_label_y_pos = label_y_pos - offset
                else:
                    this_label_y_pos = label_y_pos

                # Add change point label
                ax.annotate(
                    f"CP {cp}",
                    xy=(cp, 0),
                    xytext=(cp, this_label_y_pos),
                    color="gray",
                    fontweight="bold",
                    fontsize=10,
                    ha="center",
                    va="top",
                )

    # Reduce the number of x-axis ticks to avoid crowding
    max_ticks = 10  # Maximum number of ticks to show
    step = max(1, int((x_max - x_min) / max_ticks))
    ticks = list(range(int(x_min), int(x_max) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])

    ax.set_xlim(x_limits)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("Martingale Value", fontsize=14, fontweight="bold")

    # Add a title with the trial number if available
    if current_trial is not None:
        plt.title(
            f"Trial {current_trial} - Traditional Sum Martingales", fontsize=16, pad=10
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sum martingales plot to {output_path}")


def plot_martingales(
    file_path,
    sheet_name="Trial1",
    output_dir="results",
    threshold=50.0,
    use_boxplots=True,
):
    """Main function to plot martingale data from an Excel file.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to read (default is Trial1)
        output_dir: Directory to save output plots
        threshold: Detection threshold value
        use_boxplots: Whether to use box plots for showing distributions
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

    # Load trial data if box plots are requested
    trial_dfs = None
    all_detection_details = None
    if use_boxplots:
        trial_dfs, trial_change_points, all_detection_details = load_trial_data(
            file_path
        )
        # If we found change points in trials but not in the current sheet
        if not change_points and trial_change_points:
            change_points = trial_change_points

        # If we don't have detection details for this trial specifically, use the full set
        if detection_details is None or detection_details.empty:
            detection_details = all_detection_details

    # Plot individual martingales
    plot_individual_martingales(
        df=df,
        output_path=os.path.join(
            output_dir, f"{sheet_name}_individual_martingales.png"
        ),
        change_points=change_points,
        trial_dfs=trial_dfs if use_boxplots else None,
        detection_details=detection_details,
        threshold=threshold,
        current_trial=current_trial,
    )

    # Plot sum martingales
    plot_sum_martingales(
        df=df,
        output_path=os.path.join(output_dir, f"{sheet_name}_sum_martingales.png"),
        change_points=change_points,
        threshold=threshold,
        trial_dfs=trial_dfs if use_boxplots else None,
        detection_details=detection_details,
        current_trial=current_trial,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot martingale data from Excel file")
    parser.add_argument(
        "--file_path", "-f", type=str, required=True, help="Path to Excel file"
    )
    parser.add_argument(
        "--sheet_name", "-s", type=str, default="Trial1", help="Sheet name"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=50.0, help="Detection threshold"
    )
    parser.add_argument(
        "--no_boxplots", action="store_true", help="Disable box plots for distributions"
    )

    args = parser.parse_args()

    plot_martingales(
        file_path=args.file_path,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        threshold=args.threshold,
        use_boxplots=not args.no_boxplots,
    )
