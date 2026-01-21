"""
Create a combined figure with voting_app style: clean bars with gradient and dots
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set clean, modern style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 11

# Name mapping
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

def plot_styled_elo(ax, results_file, title):
    """
    Plot Elo rankings with voting_app style.

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

    # Number of systems
    n = len(systems)
    y_pos = np.arange(n)

    # Set up colors - purple gradient like voting_app
    bar_color_start = '#667eea'  # Purple
    bar_color_end = '#764ba2'    # Darker purple
    point_color = '#333333'      # Dark gray/black

    # Calculate x-axis range
    x_min = min(ci_lowers) - 30
    x_max = max(ci_uppers) + 30
    x_range = x_max - x_min

    # Plot each confidence interval bar with gradient effect
    for i, (lower, upper, elo) in enumerate(zip(ci_lowers, ci_uppers, elos)):
        # Create gradient-like effect using multiple overlapping bars
        bar_width = upper - lower

        # Main bar with gradient color
        rect = mpatches.FancyBboxPatch(
            (lower, i - 0.15), bar_width, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=bar_color_start,
            edgecolor=bar_color_end,
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(rect)

        # Add inner gradient effect
        inner_rect = mpatches.FancyBboxPatch(
            (lower, i - 0.12), bar_width, 0.24,
            boxstyle="round,pad=0.01",
            facecolor=bar_color_end,
            edgecolor='none',
            alpha=0.3
        )
        ax.add_patch(inner_rect)

    # Plot points for actual Elo scores with white border (like voting_app)
    ax.scatter(elos, y_pos, color=point_color, s=120, zorder=10,
               edgecolors='white', linewidth=2.5)

    # Add inner darker dot for better visibility
    ax.scatter(elos, y_pos, color=point_color, s=60, zorder=11)

    # Vertical line at base Elo
    ax.axvline(x=1000, color='#cccccc', linestyle='--', linewidth=1.2,
               alpha=0.7, zorder=0)

    # Customize axes
    ax.set_xlabel('Elo Rating', fontweight='500')
    ax.set_title(title, fontweight='600', pad=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems, fontsize=10.5)
    ax.set_xlim(x_min, x_max)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')

    # Subtle grid
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8, color='#cccccc')
    ax.set_axisbelow(True)

    # Add Elo value labels to the right
    for i, (elo, ci_lower, ci_upper) in enumerate(zip(elos, ci_lowers, ci_uppers)):
        # Elo score label
        ax.text(x_max - 15, i, f'{elo:.1f}',
                ha='right', va='center', fontweight='600',
                fontsize=11, color='#333333')

        # CI range label (smaller, lighter)
        ci_text = f'±{(ci_upper - ci_lower) / 2:.1f}'
        ax.text(x_max - 15, i - 0.25, ci_text,
                ha='right', va='top', fontweight='400',
                fontsize=8.5, color='#888888', style='italic')

    # Set background color to match voting_app
    ax.set_facecolor('#fafafa')

def create_combined_styled_figure():
    """Create a combined figure with voting_app styling."""

    # Create figure with clean white background
    fig = plt.figure(figsize=(15, 6), facecolor='white')

    # Create subplots with some spacing
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25,
                          left=0.08, right=0.96, top=0.92, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Left panel: 3-way comparison
    plot_styled_elo(
        ax1,
        'topic_0_comparison_results.json',
        'Three-Way Comparison'
    )

    # Right panel: 4-way comparison
    plot_styled_elo(
        ax2,
        'topic_4way_comparison_results.json',
        'Four-Way Comparison'
    )

    # Save figure
    output_file = 'combined_elo_rankings.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved styled figure: {output_file}")

    plt.close()

if __name__ == '__main__':
    print("Creating combined Elo ranking figure with voting_app style...")
    create_combined_styled_figure()
    print("Done!")
