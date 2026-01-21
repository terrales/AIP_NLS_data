"""
Create a combined figure showing both 3-way and 4-way Elo comparisons side-by-side
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

def load_results(filename):
    """Load Elo ranking results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_elo_comparison(ax, results_file, title):
    """
    Plot Elo rankings with confidence intervals on a given axis.

    Args:
        ax: Matplotlib axis to plot on
        results_file: Path to results JSON file
        title: Subplot title
    """
    # Load results
    results = load_results(results_file)
    rankings = results['rankings']

    # Extract data
    systems = [r['name'] for r in rankings]
    elos = [r['elo'] for r in rankings]
    ci_lowers = [r['ci_lower'] for r in rankings]
    ci_uppers = [r['ci_upper'] for r in rankings]

    # Calculate error bars (distance from point estimate)
    yerr_lower = [elos[i] - ci_lowers[i] for i in range(len(elos))]
    yerr_upper = [ci_uppers[i] - elos[i] for i in range(len(elos))]
    yerr = [yerr_lower, yerr_upper]

    # Define colors for each system
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Create positions for bars
    x_pos = np.arange(len(systems))

    # Create bars
    bars = ax.bar(x_pos, elos, color=colors[:len(systems)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add error bars for confidence intervals
    ax.errorbar(x_pos, elos, yerr=yerr, fmt='none',
                ecolor='black', elinewidth=2, capsize=8, capthick=2)

    # Add horizontal line at base Elo (1000)
    ax.axhline(y=1000, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Base Elo (1000)')

    # Customize axes
    ax.set_ylabel('Elo Rating', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(systems, rotation=20, ha='right')

    # Set y-axis limits with some padding
    y_min = min(ci_lowers) - 50
    y_max = max(ci_uppers) + 50
    ax.set_ylim(y_min, y_max)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, (bar, elo, ci_upper) in enumerate(zip(bars, elos, ci_uppers)):
        # Position label above error bar
        label_y = ci_upper + 15
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{elo:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add rank numbers
    for i, bar in enumerate(bars):
        rank = i + 1
        ax.text(bar.get_x() + bar.get_width()/2, y_min + 20,
                f'#{rank}',
                ha='center', va='bottom', fontweight='bold',
                fontsize=12, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                         edgecolor='black', linewidth=1.5))

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

def create_combined_figure():
    """Create a combined figure with both 3-way and 4-way comparisons."""

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: 3-way comparison
    plot_elo_comparison(
        ax1,
        'topic_0_comparison_results.json',
        'Three-Way Comparison'
    )

    # Right panel: 4-way comparison
    plot_elo_comparison(
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
    print("Creating combined Elo ranking figure...")
    create_combined_figure()
    print("Done!")
