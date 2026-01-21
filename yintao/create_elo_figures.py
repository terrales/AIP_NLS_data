"""
Create publication-quality figures showing Elo rankings with confidence intervals
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

def load_results(filename):
    """Load Elo ranking results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_elo_figure(results_file, output_file, title, figsize=(10, 6)):
    """
    Create a figure showing Elo rankings with confidence intervals.

    Args:
        results_file: Path to results JSON file
        output_file: Path to save output PNG
        title: Figure title
        figsize: Figure size (width, height)
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

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_xlabel('Topic Modeling System', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(systems, rotation=15, ha='right')

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
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add rank numbers
    for i, bar in enumerate(bars):
        rank = i + 1
        ax.text(bar.get_x() + bar.get_width()/2, y_min + 20,
                f'#{rank}',
                ha='center', va='bottom', fontweight='bold',
                fontsize=13, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                         edgecolor='black', linewidth=1.5))

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

    # Add metadata text
    metadata_text = f"n = {results['total_votes']:,} pairwise votes | Judge: {results['model']}"
    fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10,
             style='italic', color='gray')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved figure: {output_file}")

    plt.close()

def create_comparison_figure(results_file, output_file, title, figsize=(12, 6)):
    """
    Create a detailed figure with Elo scores and win rates.

    Args:
        results_file: Path to results JSON file
        output_file: Path to save output PNG
        title: Figure title
        figsize: Figure size (width, height)
    """
    # Load results
    results = load_results(results_file)
    rankings = results['rankings']

    # Extract data
    systems = [r['name'] for r in rankings]
    elos = [r['elo'] for r in rankings]
    ci_lowers = [r['ci_lower'] for r in rankings]
    ci_uppers = [r['ci_upper'] for r in rankings]
    win_rates = [r['win_rate'] for r in rankings]

    # Calculate error bars
    yerr_lower = [elos[i] - ci_lowers[i] for i in range(len(elos))]
    yerr_upper = [ci_uppers[i] - elos[i] for i in range(len(elos))]
    yerr = [yerr_lower, yerr_upper]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Define colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Create positions
    x_pos = np.arange(len(systems))

    # === LEFT PLOT: Elo Ratings ===
    bars1 = ax1.bar(x_pos, elos, color=colors[:len(systems)],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.errorbar(x_pos, elos, yerr=yerr, fmt='none',
                 ecolor='black', elinewidth=2, capsize=8, capthick=2)
    ax1.axhline(y=1000, color='gray', linestyle='--', linewidth=1.5,
                alpha=0.5, label='Base Elo')

    ax1.set_ylabel('Elo Rating', fontweight='bold')
    ax1.set_xlabel('System', fontweight='bold')
    ax1.set_title('Elo Ratings with 95% CI', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(systems, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right')

    # Add Elo values on bars
    for bar, elo in zip(bars1, elos):
        ax1.text(bar.get_x() + bar.get_width()/2, elo + 5,
                f'{elo:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    # === RIGHT PLOT: Win Rates ===
    bars2 = ax2.bar(x_pos, win_rates, color=colors[:len(systems)],
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                alpha=0.5, label='50% (Random)')

    ax2.set_ylabel('Win Rate (%)', fontweight='bold')
    ax2.set_xlabel('System', fontweight='bold')
    ax2.set_title('Win Rates', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(systems, rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right')

    # Add win rate values on bars
    for bar, win_rate in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, win_rate + 2,
                f'{win_rate:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    # Overall title
    fig.suptitle(title, fontweight='bold', fontsize=16, y=0.98)

    # Add metadata
    metadata_text = f"n = {results['total_votes']:,} pairwise votes | Judge: {results['model']}"
    fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10,
             style='italic', color='gray')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved figure: {output_file}")

    plt.close()

def main():
    """Generate all figures."""
    print("Generating Elo ranking figures...")
    print()

    # Figure 1: 3-way comparison
    print("Creating Figure 1: Three-way comparison...")
    create_elo_figure(
        'topic_0_comparison_results.json',
        'figure1_3way_elo_rankings.png',
        'Three-Way Topic Modeling Comparison\n(topic_0 level)',
        figsize=(10, 7)
    )

    # Figure 2: 4-way comparison
    print("Creating Figure 2: Four-way comparison...")
    create_elo_figure(
        'topic_4way_comparison_results.json',
        'figure2_4way_elo_rankings.png',
        'Four-Way Topic Modeling Comparison\n(including Original Subject Field)',
        figsize=(11, 7)
    )

    # Bonus: Detailed comparison figures
    print("\nCreating detailed comparison figures...")
    create_comparison_figure(
        'topic_0_comparison_results.json',
        'figure1_detailed_3way.png',
        'Three-Way Topic Modeling Comparison',
        figsize=(14, 6)
    )

    create_comparison_figure(
        'topic_4way_comparison_results.json',
        'figure2_detailed_4way.png',
        'Four-Way Topic Modeling Comparison',
        figsize=(14, 6)
    )

    print("\n" + "="*70)
    print("All figures generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  • figure1_3way_elo_rankings.png - Simple 3-way comparison")
    print("  • figure2_4way_elo_rankings.png - Simple 4-way comparison")
    print("  • figure1_detailed_3way.png - Detailed 3-way (Elo + Win rates)")
    print("  • figure2_detailed_4way.png - Detailed 4-way (Elo + Win rates)")
    print()

if __name__ == '__main__':
    main()
