"""
Create a combined figure with horizontal bars showing Elo confidence ranges
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Name mapping from old names to new names
NAME_MAPPING = {
    'Original Topics': 'Qwen3-Embedding + HDBSCAN',
    'MiniLM v2 Topics': 'MiniLM + HDBSCAN',
    'K-Means Topics': 'Qwen3-Embedding + K-Means',
    'Original Subject Field': 'Original Subject Field'
}

def load_results(filename):
    """Load Elo ranking results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_horizontal_elo(ax, results_file, title):
    """
    Plot Elo rankings with horizontal bars for confidence intervals.

    Args:
        ax: Matplotlib axis to plot on
        results_file: Path to results JSON file
        title: Subplot title
    """
    # Load results
    results = load_results(results_file)
    rankings = results['rankings']

    # Extract and map data
    systems = [NAME_MAPPING.get(r['name'], r['name']) for r in rankings]
    elos = [r['elo'] for r in rankings]
    ci_lowers = [r['ci_lower'] for r in rankings]
    ci_uppers = [r['ci_upper'] for r in rankings]

    # Reverse order so rank 1 is at the top
    systems = systems[::-1]
    elos = elos[::-1]
    ci_lowers = ci_lowers[::-1]
    ci_uppers = ci_uppers[::-1]

    # Calculate bar widths (confidence interval ranges)
    bar_widths = [ci_uppers[i] - ci_lowers[i] for i in range(len(elos))]
    bar_starts = ci_lowers

    # Define colors for each system (need to reverse to match new order)
    all_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    colors = all_colors[:len(systems)][::-1]

    # Create positions for bars
    y_pos = np.arange(len(systems))

    # Create horizontal bars for confidence intervals
    bars = ax.barh(y_pos, bar_widths, left=bar_starts, height=0.6,
                   color=colors, alpha=0.4, edgecolor='black', linewidth=1.2)

    # Plot points for actual Elo scores
    ax.scatter(elos, y_pos, color=colors, s=150, zorder=3,
               edgecolors='black', linewidth=2)

    # Add vertical line at base Elo (1000)
    ax.axvline(x=1000, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Base Elo (1000)')

    # Customize axes
    ax.set_xlabel('Elo Rating', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems)

    # Set x-axis limits with some padding
    x_min = min(ci_lowers) - 50
    x_max = max(ci_uppers) + 50
    ax.set_xlim(x_min, x_max)

    # Add grid for readability
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add value labels for Elo scores
    for i, (elo, ci_lower, ci_upper) in enumerate(zip(elos, ci_lowers, ci_uppers)):
        # Position label to the right of the confidence interval
        label_x = ci_upper + 10
        ax.text(label_x, i, f'{elo:.1f}',
                ha='left', va='center', fontweight='bold', fontsize=10)

    # Add legend
    ax.legend(loc='lower right', framealpha=0.9)

    # Invert y-axis so rank 1 is at the top
    ax.invert_yaxis()

def create_combined_horizontal_figure():
    """Create a combined figure with both 3-way and 4-way comparisons."""

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: 3-way comparison
    plot_horizontal_elo(
        ax1,
        'topic_0_comparison_results.json',
        'Three-Way Comparison'
    )

    # Right panel: 4-way comparison
    plot_horizontal_elo(
        ax2,
        'topic_4way_comparison_results.json',
        'Four-Way Comparison (with Subject Field)'
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = 'combined_elo_rankings.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved combined figure: {output_file}")

    plt.close()

if __name__ == '__main__':
    print("Creating combined horizontal Elo ranking figure...")
    create_combined_horizontal_figure()
    print("Done!")
